"""
Success metrics framework for generative music systems.

Defines the measurement layer that a Data Scientist would own end-to-end
for an adversarial music evaluation system. Covers:

  1. Artist DNA Preservation Score (ADPS)    -  does generation stay true to
     the source artist's identity?
  2. Style Transfer Quality (STQ)            -  how faithfully does the model
     transfer a style while remaining imperceptible?
  3. Adversarial Robustness Index (ARI)      -  how resilient is the system
     to identity-spoofing attacks?
  4. Fairness Parity Score (FPS)             -  are all artists (across genre,
     listener-base size) protected equally?
  5. Artist Health Score (AHS)               -  composite metric for dashboards,
     weighting detection, fairness, and generation quality equally.

All metrics return values in [0, 1] where 1 is best unless noted.
"""

import json
import math
import numpy as np
from pathlib import Path
from typing import Optional
from scipy.spatial.distance import cosine as cosine_distance


# ── 1. Artist DNA Preservation Score ──────────────────────────────────────────

def artist_dna_preservation_score(
    original_vec: np.ndarray,
    generated_vec: np.ndarray,
) -> float:
    """
    Cosine similarity between original and generated artist embeddings.
    Score of 1.0 means perfect identity preservation; 0.0 means orthogonal.
    """
    sim = 1.0 - cosine_distance(original_vec, generated_vec)
    return float(np.clip(sim, 0.0, 1.0))


# ── 2. Style Transfer Quality ──────────────────────────────────────────────────

def style_transfer_quality(
    source_vec: np.ndarray,
    target_vec: np.ndarray,
    output_vec: np.ndarray,
    delta_vec: np.ndarray,
    epsilon: float = 0.05,
) -> dict:
    """
    Measures how well the attack transfers style while remaining imperceptible.

    Returns:
        transfer_fidelity: similarity of output to target
        imperceptibility:  1 - (||δ||∞ / ε), how far below budget
        stq:               harmonic mean of the two (composite score)
    """
    transfer_fidelity = float(
        np.clip(1.0 - cosine_distance(output_vec, target_vec), 0.0, 1.0)
    )
    # Use L2 norm vs theoretical max L2 budget for imperceptibility.
    # L∞ saturates at ε by design; L2 captures how CONCENTRATED the
    # perturbation is  -  a sparse attack is less perceptible than a dense one.
    dim = len(delta_vec)
    l2_max = epsilon * math.sqrt(dim)  # L∞-ball inscribes this L2 sphere
    l2_actual = float(np.linalg.norm(delta_vec))
    imperceptibility = float(np.clip(1.0 - l2_actual / l2_max, 0.0, 1.0))
    stq = (
        2 * transfer_fidelity * imperceptibility / (transfer_fidelity + imperceptibility)
        if (transfer_fidelity + imperceptibility) > 0
        else 0.0
    )
    return {
        "transfer_fidelity": round(transfer_fidelity, 4),
        "imperceptibility": round(imperceptibility, 4),
        "delta_l2": round(l2_actual, 4),
        "delta_l2_budget": round(l2_max, 4),
        "stq": round(stq, 4),
    }


# ── 3. Adversarial Robustness Index ───────────────────────────────────────────

def adversarial_robustness_index(detection_results: list[dict]) -> dict:
    """
    Aggregate robustness score from detection outcomes.

    detection_results: list of dicts with keys is_adversarial, confidence.
    All inputs are assumed to be adversarial (ground truth = True).

    Returns:
        detection_rate:  fraction of attacks detected
        mean_confidence: average detection confidence
        ari:             harmonic mean of the two
    """
    n = len(detection_results)
    if n == 0:
        return {"detection_rate": 0.0, "mean_confidence": 0.0, "ari": 0.0}

    detected = sum(1 for d in detection_results if d["is_adversarial"])
    detection_rate = detected / n
    mean_conf = np.mean([d["confidence"] for d in detection_results])

    ari = (
        2 * detection_rate * mean_conf / (detection_rate + mean_conf)
        if (detection_rate + mean_conf) > 0
        else 0.0
    )
    return {
        "detection_rate": round(detection_rate, 4),
        "mean_confidence": round(float(mean_conf), 4),
        "ari": round(float(ari), 4),
    }


# ── 4. Fairness Parity Score ───────────────────────────────────────────────────

def fairness_parity_score(
    per_artist_metrics: dict[str, float],
) -> dict:
    """
    Measures how equally protected all artists are from adversarial spoofing.

    per_artist_metrics: {artist_name: detection_confidence}

    Returns:
        min_protection:    worst-case artist protection
        max_protection:    best-case
        std_protection:    standard deviation (lower = more equitable)
        gini_coefficient:  inequality measure (0 = perfect equality)
        fps:               1 - gini_coefficient
    """
    values = list(per_artist_metrics.values())
    n = len(values)
    if n == 0:
        return {}

    sorted_v = sorted(values)
    cumsum = np.cumsum(sorted_v)
    gini = float(
        (2 * np.sum((np.arange(1, n + 1)) * sorted_v) - (n + 1) * np.sum(sorted_v))
        / (n * np.sum(sorted_v) + 1e-9)
    )
    return {
        "min_protection": round(min(values), 4),
        "max_protection": round(max(values), 4),
        "mean_protection": round(float(np.mean(values)), 4),
        "std_protection": round(float(np.std(values)), 4),
        "gini_coefficient": round(float(gini), 4),
        "fps": round(float(1.0 - gini), 4),
    }


# ── 5. Artist Health Score (composite) ───────────────────────────────────────

def artist_first_health_score(
    ari: float,
    fps: float,
    mean_stq: float,
    weights: tuple[float, float, float] = (0.4, 0.3, 0.3),
) -> float:
    """
    Composite dashboard metric combining robustness, fairness, and quality.

    Weights reflect artist-identity-first principles:
      - Robustness (ARI) weighted highest: core safety signal
      - Fairness (FPS): equitable protection across artists
      - Style quality (STQ): generation quality guard
    """
    w_ari, w_fps, w_stq = weights
    return round(w_ari * ari + w_fps * fps + w_stq * mean_stq, 4)


# ── Runner ─────────────────────────────────────────────────────────────────────

def main():
    results_path = Path("data/results.json")
    fp_path = Path("data/fingerprints.json")

    if not results_path.exists() or not fp_path.exists():
        raise FileNotFoundError("Run the full pipeline first (run_pipeline.py)")

    with open(results_path) as f:
        results = json.load(f)
    with open(fp_path) as f:
        fp_data = json.load(f)

    fingerprints = fp_data["fingerprints"]
    attacks = results.get("attacks", {})
    epsilon = results.get("config", {}).get("epsilon", 0.05)

    stq_scores = []
    detection_results = []
    per_artist_confidence = {}

    print("=" * 60)
    print("ADVERSARIAL MUSIC EVAL  -  METRICS REPORT")
    print("=" * 60)

    for pair_key, pair_data in attacks.items():
        source = pair_data["source"]
        target = pair_data["target"]

        src_vec = np.array(fingerprints[source]["vector"])
        tgt_vec = np.array(fingerprints[target]["vector"])
        adv_vec = np.array(pair_data["adversarial_vector"])
        delta = np.array(pair_data["delta"])

        adps = artist_dna_preservation_score(src_vec, adv_vec)
        stq = style_transfer_quality(src_vec, tgt_vec, adv_vec, delta, epsilon)
        stq_scores.append(stq["stq"])

        det = pair_data.get("detection", {})
        if det:
            detection_results.append(det)
            per_artist_confidence[source] = det.get("confidence", 0.0)

        print(f"\n{pair_key}")
        print(f"  ADPS (DNA preservation): {adps:.4f}")
        print(f"  STQ  (style quality):    {stq['stq']:.4f}  "
              f"[fidelity={stq['transfer_fidelity']:.3f}, "
              f"imperceptibility={stq['imperceptibility']:.3f}]")
        if det:
            print(f"  Detected: {'YES' if det['is_adversarial'] else 'NO'}  "
                  f"confidence={det.get('confidence', 0):.3f}")

    ari_result = adversarial_robustness_index(detection_results)
    fps_result = fairness_parity_score(per_artist_confidence)
    mean_stq = float(np.mean(stq_scores)) if stq_scores else 0.0
    afhs = artist_first_health_score(
        ari_result["ari"], fps_result.get("fps", 0.0), mean_stq
    )

    print(f"\n{'─'*60}")
    print(f"Adversarial Robustness Index (ARI): {ari_result['ari']:.4f}")
    print(f"  Detection rate:    {ari_result['detection_rate']:.1%}")
    print(f"  Mean confidence:   {ari_result['mean_confidence']:.4f}")
    print(f"\nFairness Parity Score (FPS):        {fps_result.get('fps', 0):.4f}")
    print(f"  Gini coefficient:  {fps_result.get('gini_coefficient', 0):.4f}")
    print(f"  Worst protected:   {fps_result.get('min_protection', 0):.4f}")
    print(f"\nMean Style Transfer Quality (STQ):  {mean_stq:.4f}")
    print(f"\n{'═'*60}")
    print(f"ARTIST HEALTH SCORE (AHS):          {afhs:.4f}")
    print(f"  (ARI×0.4 + FPS×0.3 + STQ×0.3)")
    print(f"{'═'*60}")

    # Save to results
    results["product_metrics"] = {
        "ari": ari_result,
        "fps": fps_result,
        "mean_stq": round(mean_stq, 4),
        "afhs": afhs,
        "per_pair_stq": {
            pair_key: stq_scores[i]
            for i, pair_key in enumerate(attacks.keys())
            if i < len(stq_scores)
        },
    }
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nMetrics saved to {results_path}")


if __name__ == "__main__":
    main()
