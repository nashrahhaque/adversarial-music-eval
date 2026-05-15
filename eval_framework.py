"""
Evaluation framework for generative music systems.

Provides the experimental infrastructure a Data Scientist would use to:

  1. A/B test framework       -  power analysis, metric sensitivity, MDE calculation
  2. Causal inference stub    -  DiD estimator for observational rollouts
  3. Generative quality rubric  -  multi-axis evaluation of music generation outputs
  4. Fairness audit           -  disparate impact and representation analysis
  5. Ecosystem impact model   -  estimate listener × artist value exchange

All statistical methods use standard frequentist + Bayesian approaches.
"""

import json
import math
import numpy as np
from dataclasses import dataclass, asdict
from pathlib import Path
from scipy import stats


# ─────────────────────────────────────────────────────────────────────────────
# 1. A/B TEST FRAMEWORK
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ABTestConfig:
    metric_name: str
    baseline_rate: float        # control group metric value
    mde: float                  # minimum detectable effect (relative)
    alpha: float = 0.05         # significance level
    power: float = 0.80         # desired statistical power
    test_type: str = "two-sided"  # "two-sided" or "one-sided"


def compute_sample_size(config: ABTestConfig) -> dict:
    """
    Compute required sample size per variant using the standard
    normal approximation for proportion / continuous metrics.
    """
    z_alpha = stats.norm.ppf(1 - config.alpha / (2 if config.test_type == "two-sided" else 1))
    z_beta = stats.norm.ppf(config.power)

    p1 = config.baseline_rate
    p2 = p1 * (1 + config.mde)
    p_bar = (p1 + p2) / 2

    # For proportions (e.g., engagement rate)
    n_per_variant = math.ceil(
        (z_alpha * math.sqrt(2 * p_bar * (1 - p_bar)) + z_beta * math.sqrt(
            p1 * (1 - p1) + p2 * (1 - p2)
        )) ** 2 / (p1 - p2) ** 2
    )

    DAU_SCALE = 25_000_000  # ~25M DAU eligible for music gen features
    runtime_days = math.ceil((n_per_variant * 2) / DAU_SCALE)

    return {
        "metric": config.metric_name,
        "baseline": config.baseline_rate,
        "treatment": round(p2, 4),
        "mde_relative": f"{config.mde:+.1%}",
        "n_per_variant": n_per_variant,
        "total_n": n_per_variant * 2,
        "runtime_days_at_scale": runtime_days,
        "alpha": config.alpha,
        "power": config.power,
    }


def design_generative_music_ab_test() -> list[dict]:
    """
    Design A/B tests for a hypothetical AI-generated playlist feature.
    Metrics designed for generative music platform evaluation.
    """
    configs = [
        ABTestConfig(
            metric_name="stream_completion_rate",
            baseline_rate=0.62,
            mde=0.03,         # detect 3% lift
            alpha=0.05,
            power=0.80,
        ),
        ABTestConfig(
            metric_name="artist_save_rate",           # artist-first signal
            baseline_rate=0.08,
            mde=0.10,
            alpha=0.05,
            power=0.80,
        ),
        ABTestConfig(
            metric_name="skip_rate",
            baseline_rate=0.22,
            mde=-0.05,        # detect 5% reduction
            alpha=0.05,
            power=0.80,
        ),
        ABTestConfig(
            metric_name="new_artist_discovery_rate",  # ecosystem health
            baseline_rate=0.15,
            mde=0.08,
            alpha=0.05,
            power=0.80,
        ),
    ]

    results = []
    for cfg in configs:
        result = compute_sample_size(cfg)
        results.append(result)

    return results


# ─────────────────────────────────────────────────────────────────────────────
# 2. CAUSAL INFERENCE  -  DiD ESTIMATOR
# ─────────────────────────────────────────────────────────────────────────────

def difference_in_differences(
    pre_control: np.ndarray,
    post_control: np.ndarray,
    pre_treatment: np.ndarray,
    post_treatment: np.ndarray,
) -> dict:
    """
    Standard 2x2 DiD estimator for observational rollout analysis.
    Assumes parallel trends in the pre-period.
    """
    control_change = np.mean(post_control) - np.mean(pre_control)
    treatment_change = np.mean(post_treatment) - np.mean(pre_treatment)
    att = treatment_change - control_change  # Average Treatment effect on Treated

    # Bootstrap confidence interval
    n_bootstrap = 2000
    rng = np.random.default_rng(42)
    att_boot = []
    for _ in range(n_bootstrap):
        pc = rng.choice(pre_control, len(pre_control))
        psc = rng.choice(post_control, len(post_control))
        pt = rng.choice(pre_treatment, len(pre_treatment))
        pst = rng.choice(post_treatment, len(post_treatment))
        att_boot.append((np.mean(pst) - np.mean(pt)) - (np.mean(psc) - np.mean(pc)))

    ci_lo, ci_hi = np.percentile(att_boot, [2.5, 97.5])
    p_val = float(np.mean(np.array(att_boot) <= 0) * 2)  # two-sided

    return {
        "att_estimate": round(float(att), 6),
        "ci_95_lo": round(float(ci_lo), 6),
        "ci_95_hi": round(float(ci_hi), 6),
        "p_value": round(float(p_val), 4),
        "significant": bool(p_val < 0.05),
        "control_trend": round(float(control_change), 6),
        "treatment_trend": round(float(treatment_change), 6),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 3. GENERATIVE QUALITY RUBRIC
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class GenerativeQualityScore:
    """
    Multi-axis rubric for evaluating generative music outputs.
    Designed to capture both model quality AND artist-first concerns.
    """
    artist_identity_fidelity: float   # [0,1] does it sound like the right artist?
    genre_coherence: float            # [0,1] does genre match expectation?
    audio_quality: float              # [0,1] technical audio quality
    novelty: float                    # [0,1] not a direct copy
    artist_consent_signal: float      # [0,1] artist opted in to this use case
    listener_satisfaction_proxy: float # [0,1] predicted skip rate inverse

    def composite(self, weights: dict[str, float] | None = None) -> float:
        if weights is None:
            weights = {
                "artist_identity_fidelity": 0.25,
                "genre_coherence": 0.15,
                "audio_quality": 0.15,
                "novelty": 0.10,
                "artist_consent_signal": 0.20,   # artist-first: high weight
                "listener_satisfaction_proxy": 0.15,
            }
        total = sum(getattr(self, k) * v for k, v in weights.items())
        return round(total, 4)


def simulate_quality_scores(n_samples: int = 100, seed: int = 42) -> list[dict]:
    """Simulate quality scores for a generative music model evaluation."""
    rng = np.random.default_rng(seed)
    scores = []
    for i in range(n_samples):
        s = GenerativeQualityScore(
            artist_identity_fidelity=float(rng.beta(7, 3)),
            genre_coherence=float(rng.beta(8, 2)),
            audio_quality=float(rng.beta(9, 2)),
            novelty=float(rng.beta(5, 3)),
            artist_consent_signal=float(rng.choice([0.0, 1.0], p=[0.15, 0.85])),
            listener_satisfaction_proxy=float(rng.beta(6, 3)),
        )
        scores.append({**asdict(s), "composite": s.composite()})
    return scores


# ─────────────────────────────────────────────────────────────────────────────
# 4. FAIRNESS AUDIT
# ─────────────────────────────────────────────────────────────────────────────

ARTIST_METADATA = {
    "Taylor Swift":   {"genre": "pop",      "tier": "mega",  "gender": "f"},
    "Drake":          {"genre": "hip-hop",  "tier": "mega",  "gender": "m"},
    "Billie Eilish":  {"genre": "alt-pop",  "tier": "mega",  "gender": "f"},
    "The Weeknd":     {"genre": "r&b",      "tier": "mega",  "gender": "m"},
    "Kendrick Lamar": {"genre": "hip-hop",  "tier": "mega",  "gender": "m"},
    "Olivia Rodrigo": {"genre": "pop-rock", "tier": "large", "gender": "f"},
    "21 Savage":      {"genre": "hip-hop",  "tier": "large", "gender": "m"},
    "Lorde":          {"genre": "alt-pop",  "tier": "mid",   "gender": "f"},
    "Post Malone":    {"genre": "pop",      "tier": "mega",  "gender": "m"},
    "J. Cole":        {"genre": "hip-hop",  "tier": "large", "gender": "m"},
}


def fairness_audit(per_artist_metrics: dict[str, float]) -> dict:
    """
    Disparate impact analysis across genre and tier groups.
    Uses the 4/5ths rule: a group is disadvantaged if its mean metric
    is < 80% of the best-performing group's mean.
    """
    from collections import defaultdict

    groups = defaultdict(list)
    for artist, metric in per_artist_metrics.items():
        meta = ARTIST_METADATA.get(artist, {})
        groups[f"genre:{meta.get('genre', 'unknown')}"].append(metric)
        groups[f"tier:{meta.get('tier', 'unknown')}"].append(metric)
        groups[f"gender:{meta.get('gender', 'unknown')}"].append(metric)

    group_means = {g: float(np.mean(vals)) for g, vals in groups.items()}
    best_mean = max(group_means.values()) if group_means else 1.0

    audit = {}
    for group, mean in group_means.items():
        impact_ratio = mean / best_mean if best_mean > 0 else 1.0
        audit[group] = {
            "mean": round(mean, 4),
            "n": len(groups[group]),
            "impact_ratio": round(impact_ratio, 4),
            "disparate_impact": bool(impact_ratio < 0.80),
        }

    return audit


# ─────────────────────────────────────────────────────────────────────────────
# 5. ECOSYSTEM IMPACT MODEL
# ─────────────────────────────────────────────────────────────────────────────

def ecosystem_impact_estimate(
    detection_rate: float,
    n_daily_generations: int = 5_000_000,
    artist_royalty_per_stream: float = 0.004,
    attack_royalty_redirect_pct: float = 0.30,
) -> dict:
    """
    Estimate the financial and ecosystem impact of adversarial identity attacks
    at platform scale.

    If an adversarial attack redirects listener engagement from artist A to
    artist B (by spoofing A's style in a generative model), the royalty
    flow is distorted. This model quantifies the protected value.
    """
    n_attacks_daily = n_daily_generations * 0.01  # assume 1% attack attempt rate
    n_undetected = n_attacks_daily * (1 - detection_rate)
    n_detected = n_attacks_daily * detection_rate

    royalty_at_risk = n_attacks_daily * artist_royalty_per_stream * attack_royalty_redirect_pct
    royalty_protected = n_detected * artist_royalty_per_stream * attack_royalty_redirect_pct
    royalty_lost = n_undetected * artist_royalty_per_stream * attack_royalty_redirect_pct

    return {
        "daily_generations": n_daily_generations,
        "estimated_daily_attacks": int(n_attacks_daily),
        "detected_attacks": int(n_detected),
        "missed_attacks": int(n_undetected),
        "royalty_at_risk_daily_usd": round(royalty_at_risk, 2),
        "royalty_protected_daily_usd": round(royalty_protected, 2),
        "royalty_lost_daily_usd": round(royalty_lost, 2),
        "annual_protected_usd": round(royalty_protected * 365, 0),
    }


# ─────────────────────────────────────────────────────────────────────────────
# RUNNER
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print("GENERATIVE MUSIC EVALUATION FRAMEWORK")
    print("Adversarial Music Evaluation System")
    print("=" * 65)

    # A/B test designs
    print("\n── A/B Test Power Analysis ──────────────────────────────────")
    ab_results = design_generative_music_ab_test()
    for r in ab_results:
        print(f"  {r['metric']:35s}  n={r['n_per_variant']:>8,}  "
              f"runtime={r['runtime_days_at_scale']}d  "
              f"MDE={r['mde_relative']}")

    # Simulated DiD
    print("\n── Causal Inference (DiD on simulated rollout) ──────────────")
    rng = np.random.default_rng(42)
    pre_ctrl = rng.normal(0.62, 0.05, 5000)
    post_ctrl = rng.normal(0.63, 0.05, 5000)
    pre_trt = rng.normal(0.62, 0.05, 5000)
    post_trt = rng.normal(0.67, 0.05, 5000)   # true effect: +4pp
    did = difference_in_differences(pre_ctrl, post_ctrl, pre_trt, post_trt)
    print(f"  ATT estimate: {did['att_estimate']:+.4f}  "
          f"95% CI [{did['ci_95_lo']:+.4f}, {did['ci_95_hi']:+.4f}]  "
          f"p={did['p_value']:.4f}  sig={'✓' if did['significant'] else '✗'}")

    # Generative quality rubric
    print("\n── Generative Quality Rubric (n=100 simulated outputs) ─────")
    quality_scores = simulate_quality_scores(n_samples=100)
    composites = [s["composite"] for s in quality_scores]
    consent_ok = [s for s in quality_scores if s["artist_consent_signal"] > 0.5]
    print(f"  Mean composite score:   {np.mean(composites):.4f}")
    print(f"  P10/P50/P90:            {np.percentile(composites, 10):.3f} / "
          f"{np.percentile(composites, 50):.3f} / "
          f"{np.percentile(composites, 90):.3f}")
    print(f"  Artist consent rate:    {len(consent_ok)/len(quality_scores):.1%}")

    # Fairness audit (using simulated per-artist confidence)
    print("\n── Fairness Audit (simulated protection scores) ─────────────")
    rng2 = np.random.default_rng(99)
    sim_conf = {a: float(rng2.beta(7, 2)) for a in ARTIST_METADATA}
    audit = fairness_audit(sim_conf)
    for group, data in sorted(audit.items()):
        flag = " ⚠ DISPARATE IMPACT" if data["disparate_impact"] else ""
        print(f"  {group:25s}  mean={data['mean']:.3f}  "
              f"ratio={data['impact_ratio']:.3f}{flag}")

    # Ecosystem impact
    print("\n── Ecosystem Impact Model ───────────────────────────────────")
    impact = ecosystem_impact_estimate(detection_rate=0.80)
    print(f"  Daily generations:        {impact['daily_generations']:>12,}")
    print(f"  Estimated daily attacks:  {impact['estimated_daily_attacks']:>12,}")
    print(f"  Royalty at risk/day:      ${impact['royalty_at_risk_daily_usd']:>11,.2f}")
    print(f"  Royalty protected/day:    ${impact['royalty_protected_daily_usd']:>11,.2f}")
    print(f"  Annual protected value:   ${impact['annual_protected_usd']:>11,.0f}")

    # Save
    results_path = Path("data/results.json")
    if results_path.exists():
        with open(results_path) as f:
            results = json.load(f)
    else:
        results = {}

    results["eval_framework"] = {
        "ab_tests": ab_results,
        "did_example": did,
        "quality_rubric_summary": {
            "n_samples": len(quality_scores),
            "mean_composite": round(float(np.mean(composites)), 4),
            "p10": round(float(np.percentile(composites, 10)), 4),
            "p50": round(float(np.percentile(composites, 50)), 4),
            "p90": round(float(np.percentile(composites, 90)), 4),
            "consent_rate": round(len(consent_ok) / len(quality_scores), 4),
        },
        "fairness_audit": audit,
        "ecosystem_impact": impact,
    }

    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nEvaluation framework results saved to {results_path}")


if __name__ == "__main__":
    main()
