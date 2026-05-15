"""
Statistical adversarial anomaly detector.

Uses per-dimension z-score deviation from the clean embedding distribution
to flag adversarial examples. An embedding is flagged as adversarial if its
mean absolute z-score exceeds the detection threshold (default 2.0).

Also computes:
  - Per-dimension flagging (which feature dimensions were most perturbed)
  - Confidence score calibrated to the z-score magnitude
  - Precision / recall / F1 over the attack set (ground truth: all attacked)
  - Dimension-level attribution (which fingerprint dims are most exploited)

Detection results are merged into data/results.json.
"""

import json
import math
import numpy as np
from pathlib import Path
from scipy import stats as scipy_stats

THRESHOLD = 2.0  # mean |z| above which we flag as adversarial

ARTISTS = [
    "Taylor Swift", "Drake", "Billie Eilish", "The Weeknd",
    "Kendrick Lamar", "Olivia Rodrigo", "21 Savage", "Lorde",
    "Post Malone", "J. Cole",
]

DIM_LAYOUT = {
    "tags": (0, 30),
    "listeners": (30, 31),
    "playcount": (31, 32),
    "engagement": (32, 33),
    "tag_diversity": (33, 34),
    "similar_artists": (34, 44),
    "track_stats": (44, 49),
    "tag_weight_stats": (49, 54),
}


def z_scores(embedding: np.ndarray, dist_mean: np.ndarray, dist_std: np.ndarray) -> np.ndarray:
    """
    Per-dimension z-scores measuring how far an embedding deviates from
    the clean distribution mean, scaled by each dimension's natural variance.

    Dimensions with near-zero natural variance (std < 0.002) are the most
    sensitive: ANY perturbation there stands out strongly. We floor std at
    0.002 so those dims contribute a large but finite signal rather than
    infinite z-scores. The floor is chosen so that the maximum L∞ budget
    (ε=0.05) on a near-constant dim produces z = 0.05/0.002 = 25, a clear
    anomaly that pushes the mean above the 2.0 threshold.
    """
    safe_std = np.where(dist_std >= 0.002, dist_std, 0.002)
    return np.abs((embedding - dist_mean) / safe_std)


def detect(
    embedding: np.ndarray,
    dist_mean: np.ndarray,
    dist_std: np.ndarray,
    threshold: float = THRESHOLD,
) -> dict:
    zs = z_scores(embedding, dist_mean, dist_std)
    mean_z = float(zs.mean())
    max_z = float(zs.max())
    flagged_dim_count = int((zs > threshold).sum())
    top5_dims = zs.argsort()[-5:][::-1].tolist()

    # Confidence: sigmoid-scaled distance above threshold
    confidence = float(1 / (1 + math.exp(-2.0 * (mean_z - threshold))))

    # Per-region z-scores
    region_zscores = {}
    for region, (start, end) in DIM_LAYOUT.items():
        region_zscores[region] = float(zs[start:end].mean())

    return {
        "is_adversarial": bool(mean_z > threshold),
        "mean_z_score": round(mean_z, 6),
        "max_z_score": round(max_z, 6),
        "flagged_dim_count": flagged_dim_count,
        "top5_perturbed_dims": top5_dims,
        "confidence": round(confidence, 6),
        "region_z_scores": {k: round(v, 4) for k, v in region_zscores.items()},
    }


def compute_detection_metrics(detections: list[dict], ground_truth_positive: list[bool]) -> dict:
    """Precision, recall, F1 and AUROC approximation over the detection results."""
    preds = [d["is_adversarial"] for d in detections]
    tp = sum(p and g for p, g in zip(preds, ground_truth_positive))
    fp = sum(p and not g for p, g in zip(preds, ground_truth_positive))
    fn = sum(not p and g for p, g in zip(preds, ground_truth_positive))
    tn = sum(not p and not g for p, g in zip(preds, ground_truth_positive))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    # Rank by confidence for AUROC approximation
    scores = [d["confidence"] for d in detections]
    if len(set(ground_truth_positive)) == 2:
        from sklearn.metrics import roc_auc_score
        try:
            auroc = float(roc_auc_score(ground_truth_positive, scores))
        except Exception:
            auroc = None
    else:
        auroc = None

    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "auroc": round(auroc, 4) if auroc is not None else None,
    }


def attribution_analysis(attack_results: dict, dist_mean: np.ndarray, dist_std: np.ndarray) -> dict:
    """Which fingerprint regions are most exploited across all attacks?"""
    region_z_totals = {r: 0.0 for r in DIM_LAYOUT}
    count = 0
    for pair_data in attack_results.values():
        adv_vec = np.array(pair_data["adversarial_vector"], dtype=np.float32)
        zs = z_scores(adv_vec, dist_mean, dist_std)
        for region, (start, end) in DIM_LAYOUT.items():
            region_z_totals[region] += float(zs[start:end].mean())
        count += 1

    if count == 0:
        return {}
    return {r: round(v / count, 4) for r, v in region_z_totals.items()}


def main():
    results_path = Path("data/results.json")
    fp_path = Path("data/fingerprints.json")

    if not results_path.exists():
        raise FileNotFoundError("Run attack.py first to generate data/results.json")
    if not fp_path.exists():
        raise FileNotFoundError("Run fingerprint.py first to generate data/fingerprints.json")

    with open(results_path) as f:
        results = json.load(f)
    with open(fp_path) as f:
        fp_data = json.load(f)

    fingerprints = fp_data["fingerprints"]

    # Build clean distribution from all 10 artist embeddings
    clean_vecs = np.array([
        fingerprints[a]["vector"]
        for a in ARTISTS
        if a in fingerprints
    ], dtype=np.float32)

    dist_mean = clean_vecs.mean(axis=0)
    dist_std = clean_vecs.std(axis=0)

    print(f"Clean distribution: {len(clean_vecs)} artists, "
          f"mean_norm={np.linalg.norm(dist_mean):.4f}, "
          f"mean_std={dist_std.mean():.4f}\n")

    # ── Detect on clean embeddings (should mostly not be flagged) ─────────────
    clean_detections = []
    print("Clean embeddings:")
    for artist in ARTISTS:
        if artist not in fingerprints:
            continue
        vec = np.array(fingerprints[artist]["vector"], dtype=np.float32)
        result = detect(vec, dist_mean, dist_std, threshold=THRESHOLD)
        clean_detections.append(result)
        flag = "⚠ FLAGGED" if result["is_adversarial"] else "  clean"
        print(f"  {artist:22s} {flag}  mean_z={result['mean_z_score']:.3f}")

    # ── Detect on adversarial embeddings ──────────────────────────────────────
    attack_detections = []
    print("\nAdversarial embeddings:")
    for pair_key, pair_data in results.get("attacks", {}).items():
        adv_vec = np.array(pair_data["adversarial_vector"], dtype=np.float32)
        result = detect(adv_vec, dist_mean, dist_std, threshold=THRESHOLD)
        attack_detections.append(result)
        results["attacks"][pair_key]["detection"] = result
        flag = "⚠ FLAGGED" if result["is_adversarial"] else "  missed"
        print(f"  {pair_key:40s} {flag}  mean_z={result['mean_z_score']:.3f}  "
              f"conf={result['confidence']:.3f}")

    # ── Overall detection metrics ─────────────────────────────────────────────
    all_detections = clean_detections + attack_detections
    ground_truth = [False] * len(clean_detections) + [True] * len(attack_detections)
    metrics = compute_detection_metrics(all_detections, ground_truth)

    print(f"\nDetection metrics (threshold={THRESHOLD}):")
    print(f"  Precision={metrics['precision']:.3f}  Recall={metrics['recall']:.3f}  "
          f"F1={metrics['f1']:.3f}  AUROC={metrics['auroc']}")

    # ── Attribution analysis ──────────────────────────────────────────────────
    attribution = attribution_analysis(results.get("attacks", {}), dist_mean, dist_std)
    print(f"\nMost exploited fingerprint regions (mean |z|):")
    max_score = max(attribution.values(), default=1.0)
    for region, score in sorted(attribution.items(), key=lambda x: -x[1]):
        bar_len = int((score / max_score) * 30)
        bar = "█" * bar_len
        print(f"  {region:20s} {score:.3f}  {bar}")

    # ── Save ──────────────────────────────────────────────────────────────────
    results["detection"] = {
        "threshold": THRESHOLD,
        "distribution": {
            "mean": dist_mean.tolist(),
            "std": dist_std.tolist(),
            "n_artists": len(clean_vecs),
        },
        "clean_embeddings": {
            artist: clean_detections[i]
            for i, artist in enumerate(a for a in ARTISTS if a in fingerprints)
        },
        "metrics": metrics,
        "attribution": attribution,
    }

    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nDetection results saved to {results_path}")


if __name__ == "__main__":
    main()
