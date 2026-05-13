"""
Momentum-Integrated Adversarial Attack on Artist Fingerprints.

Adapted from the SD-MIAE (Style-Directed Momentum-Integrated Adversarial
Example) methodology. The attack finds a perturbation δ such that:

    model(source + δ) ≈ model(target)   [high cosine similarity]
    ‖δ‖∞ ≤ ε                            [imperceptibility constraint]

The proxy MLP stands in for a generative music model's encoder/decoder.
In a real deployment this would attack the latent conditioning vector of
a system like AudioCraft, MusicGen, or a Spotify-internal model.

Hyperparameters (matching SD-MIAE defaults):
    ε  = 0.05   (L∞ budget)
    μ  = 0.9    (momentum decay)
    α  = 0.01   (step size)
    T  = 20     (iterations)
"""

import json
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

# ── Attack pairs (source → target) ────────────────────────────────────────────
ATTACK_PAIRS = [
    ("Taylor Swift", "Olivia Rodrigo"),
    ("Drake", "21 Savage"),
    ("Billie Eilish", "Lorde"),
    ("The Weeknd", "Post Malone"),
    ("Kendrick Lamar", "J. Cole"),
]

# ── Hyperparameters ────────────────────────────────────────────────────────────
EPSILON = 0.05
MU = 0.9
ALPHA = 0.01
STEPS = 20
INPUT_DIM = 54
HIDDEN_DIM = 128
SEED = 42


# ── Proxy generative model ─────────────────────────────────────────────────────
class ProxyMLP(nn.Module):
    """
    Lightweight MLP proxy for a generative music model's style encoder.

    Architecture mirrors a typical conditioning network: two hidden layers
    with residual-style skip, followed by L2-normalized output so cosine
    similarity is the natural distance metric in the latent space.
    """

    def __init__(self, input_dim: int = INPUT_DIM, hidden_dim: int = HIDDEN_DIM):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.proj = nn.Linear(hidden_dim, input_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encoder(x)
        out = self.proj(h)
        return F.normalize(out, dim=-1)


def momentum_adversarial_attack(
    model: nn.Module,
    source: np.ndarray,
    target: np.ndarray,
    epsilon: float = EPSILON,
    mu: float = MU,
    alpha: float = ALPHA,
    steps: int = STEPS,
) -> tuple[np.ndarray, list[dict]]:
    """
    SD-MIAE: momentum-integrated sign-gradient attack.

    Gradient accumulates momentum to escape flat regions and avoid oscillation,
    analogous to Nesterov momentum in optimization. The sign update keeps the
    perturbation norm well-controlled within the L∞ ball.

    Returns:
        delta: perturbation array of shape (input_dim,)
        history: per-step metrics for analysis/visualization
    """
    model.eval()
    src = torch.FloatTensor(source).unsqueeze(0)
    tgt = torch.FloatTensor(target).unsqueeze(0)

    with torch.no_grad():
        target_emb = model(tgt)

    delta = torch.zeros_like(src, requires_grad=True)
    momentum = torch.zeros_like(src)

    history = []

    for step in range(steps):
        adv_emb = model(src + delta)

        # Maximize cosine similarity to target embedding
        cos_sim = F.cosine_similarity(adv_emb, target_emb)
        loss = -cos_sim.mean()

        loss.backward()

        with torch.no_grad():
            grad = delta.grad.data.clone()

            # L1-normalize gradient (standard MI-FGSM step)
            grad_l1 = grad.abs().mean()
            if grad_l1 > 1e-8:
                grad = grad / grad_l1

            # Momentum accumulation
            momentum = mu * momentum + grad

            # Sign update
            delta.data = delta.data - alpha * momentum.sign()

            # Project onto L∞ ball
            delta.data = torch.clamp(delta.data, -epsilon, epsilon)

            # Clip adversarial example to valid feature range [0, 1]
            adv_clipped = torch.clamp(src + delta.data, 0.0, 1.0)
            delta.data = adv_clipped - src

            # Recompute metrics on clean delta
            adv_emb_eval = model(src + delta.data)
            step_cos = F.cosine_similarity(adv_emb_eval, target_emb).item()
            step_linf = delta.data.abs().max().item()

            # Distance between source and target in latent space (for reference)
            src_emb = model(src)
            src_tgt_sim = F.cosine_similarity(src_emb, target_emb).item()

        history.append({
            "step": step,
            "loss": float(-cos_sim.mean().item()),
            "cosine_similarity": float(step_cos),
            "delta_linf": float(step_linf),
            "delta_l2": float(delta.data.norm(p=2).item()),
            "delta_mean_abs": float(delta.data.abs().mean().item()),
            "src_tgt_baseline": float(src_tgt_sim),
            "improvement": float(step_cos - src_tgt_sim),
        })

        delta.grad.zero_()

    return delta.detach().numpy().squeeze(), history


def compute_transfer_metrics(
    model: nn.Module,
    source: np.ndarray,
    target: np.ndarray,
    delta: np.ndarray,
) -> dict:
    """Compute final attack quality metrics."""
    model.eval()
    with torch.no_grad():
        src_t = torch.FloatTensor(source).unsqueeze(0)
        tgt_t = torch.FloatTensor(target).unsqueeze(0)
        adv_t = torch.FloatTensor(source + delta).unsqueeze(0).clamp(0, 1)

        src_emb = model(src_t)
        tgt_emb = model(tgt_t)
        adv_emb = model(adv_t)

        baseline_sim = F.cosine_similarity(src_emb, tgt_emb).item()
        attack_sim = F.cosine_similarity(adv_emb, tgt_emb).item()
        src_preservation = F.cosine_similarity(adv_emb, src_emb).item()

    return {
        "baseline_similarity": round(baseline_sim, 6),
        "attack_similarity": round(attack_sim, 6),
        "absolute_improvement": round(attack_sim - baseline_sim, 6),
        "source_preservation": round(src_preservation, 6),
        "delta_linf": round(float(np.abs(delta).max()), 6),
        "delta_l2": round(float(np.linalg.norm(delta)), 6),
        "constraint_satisfied": bool(np.abs(delta).max() <= EPSILON + 1e-6),
    }


def main():
    fp_path = Path("data/fingerprints.json")
    if not fp_path.exists():
        raise FileNotFoundError("Run fingerprint.py first to generate data/fingerprints.json")

    with open(fp_path) as f:
        fp_data = json.load(f)

    fingerprints = fp_data["fingerprints"]

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    model = ProxyMLP(input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM)

    # Load existing results to preserve detection data if already run
    results_path = Path("data/results.json")
    results = json.loads(results_path.read_text()) if results_path.exists() else {}
    results.setdefault("attacks", {})
    results["config"] = {
        "epsilon": EPSILON,
        "mu": MU,
        "alpha": ALPHA,
        "steps": STEPS,
        "input_dim": INPUT_DIM,
        "hidden_dim": HIDDEN_DIM,
        "seed": SEED,
    }

    print(f"Running SD-MIAE on {len(ATTACK_PAIRS)} pairs "
          f"(ε={EPSILON}, μ={MU}, α={ALPHA}, T={STEPS})\n")

    for source_name, target_name in ATTACK_PAIRS:
        if source_name not in fingerprints or target_name not in fingerprints:
            print(f"  SKIP {source_name} → {target_name} (missing fingerprint)")
            continue

        source_vec = np.array(fingerprints[source_name]["vector"], dtype=np.float32)
        target_vec = np.array(fingerprints[target_name]["vector"], dtype=np.float32)

        print(f"  Attacking {source_name:20s} → {target_name}...")
        delta, history = momentum_adversarial_attack(
            model, source_vec, target_vec,
            epsilon=EPSILON, mu=MU, alpha=ALPHA, steps=STEPS,
        )
        metrics = compute_transfer_metrics(model, source_vec, target_vec, delta)

        adv_vec = np.clip(source_vec + delta, 0.0, 1.0)

        pair_key = f"{source_name} → {target_name}"
        results["attacks"][pair_key] = {
            "source": source_name,
            "target": target_name,
            "metrics": metrics,
            "history": history,
            "adversarial_vector": adv_vec.tolist(),
            "delta": delta.tolist(),
        }

        print(f"    baseline sim={metrics['baseline_similarity']:.4f}  "
              f"attack sim={metrics['attack_similarity']:.4f}  "
              f"Δ={metrics['absolute_improvement']:+.4f}  "
              f"‖δ‖∞={metrics['delta_linf']:.4f}  "
              f"constraint={'✓' if metrics['constraint_satisfied'] else '✗'}")

    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
