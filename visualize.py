"""
Visualization suite for adversarial music evaluation.

Generates publication-quality figures saved to figures/:
  1. fingerprint_radar.png    -  per-artist DNA radar charts
  2. pca_fingerprints.png     -  2D PCA of the fingerprint space
  3. similarity_heatmap.png   -  pairwise cosine similarity matrix
  4. attack_trajectories.png  -  cosine similarity vs iteration for each attack
  5. detection_summary.png    -  z-score distributions: clean vs adversarial
  6. attribution_bars.png     -  which fingerprint regions are most exploited
  7. metrics_dashboard.png    -  composite metrics summary panel
"""

import json
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from pathlib import Path
from sklearn.decomposition import PCA
from scipy.spatial.distance import cosine as cosine_dist

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
})

CLEAN_GREEN   = "#1A5C2A"
DARK_BG       = "#191414"
ATTACK_RED = "#E53935"
CLEAN_BLUE = "#1565C0"
PALETTE = [
    "#1DB954", "#E53935", "#1565C0", "#F9A825",
    "#6A1B9A", "#00838F", "#EF6C00", "#2E7D32",
    "#AD1457", "#37474F",
]

ARTISTS = [
    "Taylor Swift", "Drake", "Billie Eilish", "The Weeknd",
    "Kendrick Lamar", "Olivia Rodrigo", "21 Savage", "Lorde",
    "Post Malone", "J. Cole",
]

DIM_LAYOUT = {
    "Tags": (0, 30),
    "Listeners": (30, 31),
    "Playcount": (31, 32),
    "Engagement": (32, 33),
    "Tag Diversity": (33, 34),
    "Similar Artists": (34, 44),
    "Track Stats": (44, 49),
    "Tag Weights": (49, 54),
}


def load_data() -> tuple[dict, dict]:
    with open("data/fingerprints.json") as f:
        fp_data = json.load(f)
    with open("data/results.json") as f:
        results = json.load(f)
    return fp_data, results


def get_vectors(fp_data: dict) -> tuple[list[str], np.ndarray]:
    names = [a for a in ARTISTS if a in fp_data["fingerprints"]]
    vecs = np.array([fp_data["fingerprints"][a]["vector"] for a in names])
    return names, vecs


# ── 1. Fingerprint Radar Charts ───────────────────────────────────────────────

def plot_fingerprint_radar(fp_data: dict):
    """One radar chart per artist showing 8 fingerprint region averages."""
    names, vecs = get_vectors(fp_data)
    regions = list(DIM_LAYOUT.keys())
    n_regions = len(regions)

    fig, axes = plt.subplots(2, 5, figsize=(18, 7),
                             subplot_kw={"projection": "polar"})
    fig.patch.set_facecolor(DARK_BG)
    fig.suptitle("Artist DNA Fingerprints", color="white", fontsize=16, fontweight="bold", y=1.01)

    angles = np.linspace(0, 2 * math.pi, n_regions, endpoint=False).tolist()
    angles += angles[:1]

    for idx, (artist, vec) in enumerate(zip(names, vecs)):
        ax = axes[idx // 5][idx % 5]
        ax.set_facecolor("#222222")

        region_vals = []
        for region, (s, e) in DIM_LAYOUT.items():
            region_vals.append(float(vec[s:e].mean()))
        region_vals += region_vals[:1]

        ax.plot(angles, region_vals, color=PALETTE[idx], linewidth=2)
        ax.fill(angles, region_vals, color=PALETTE[idx], alpha=0.25)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(
            [r[:6] for r in regions],
            color="white", fontsize=6
        )
        ax.set_yticks([0.2, 0.4, 0.6, 0.8])
        ax.set_yticklabels([], color="gray")
        ax.tick_params(colors="white")
        ax.set_title(artist.split()[-1], color="white", fontsize=9, pad=8)
        ax.spines["polar"].set_color("#444444")
        ax.grid(color="#444444", linewidth=0.5)

    plt.tight_layout()
    plt.savefig("figures/fingerprint_radar.png", bbox_inches="tight",
                facecolor=DARK_BG)
    plt.close()
    print("  Saved figures/fingerprint_radar.png")


# ── 2. PCA of Fingerprint Space ───────────────────────────────────────────────

def plot_pca(fp_data: dict, results: dict):
    names, vecs = get_vectors(fp_data)
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(vecs)
    var_exp = pca.explained_variance_ratio_

    fig, ax = plt.subplots(figsize=(9, 7))
    ax.set_facecolor("#F5F5F5")

    for i, (name, (x, y)) in enumerate(zip(names, coords)):
        ax.scatter(x, y, color=PALETTE[i], s=180, zorder=3, edgecolors="white", linewidth=1.5)
        short = name.split()[0] if len(name) > 10 else name
        ax.annotate(short, (x, y), textcoords="offset points",
                    xytext=(8, 5), fontsize=8, color="#333333")

    # Draw attack arrows
    for pair_key, pair_data in results.get("attacks", {}).items():
        src, tgt = pair_data["source"], pair_data["target"]
        if src in names and tgt in names:
            si, ti = names.index(src), names.index(tgt)
            ax.annotate(
                "", xy=coords[ti], xytext=coords[si],
                arrowprops=dict(arrowstyle="->", color=ATTACK_RED,
                                lw=1.5, linestyle="dashed"),
            )

    ax.set_xlabel(f"PC1 ({var_exp[0]:.1%} variance)", fontsize=11)
    ax.set_ylabel(f"PC2 ({var_exp[1]:.1%} variance)", fontsize=11)
    ax.set_title("Artist Fingerprint Space (PCA)\nDashed arrows = adversarial attack pairs",
                 fontsize=12, fontweight="bold")

    legend = [
        mpatches.Patch(color=PALETTE[i], label=names[i])
        for i in range(len(names))
    ]
    ax.legend(handles=legend, loc="lower right", fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig("figures/pca_fingerprints.png", bbox_inches="tight")
    plt.close()
    print("  Saved figures/pca_fingerprints.png")


# ── 3. Cosine Similarity Heatmap ──────────────────────────────────────────────

def plot_similarity_heatmap(fp_data: dict):
    import matplotlib.colors as mcolors
    names, vecs = get_vectors(fp_data)
    n = len(names)
    sim_mat = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            sim_mat[i, j] = 1.0 - cosine_dist(vecs[i], vecs[j])

    fig, ax = plt.subplots(figsize=(9, 8))
    cmap = matplotlib.colormaps["RdYlGn"]
    im = ax.imshow(sim_mat, cmap=cmap, vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, label="Cosine Similarity")

    short_names = [n.split()[0] for n in names]
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(short_names, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(short_names, fontsize=9)

    for i in range(n):
        for j in range(n):
            val = sim_mat[i, j]
            color = "white" if val < 0.5 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=8, color=color)

    ax.set_title("Artist Fingerprint Pairwise Cosine Similarity",
                 fontsize=13, fontweight="bold", pad=12)
    plt.tight_layout()
    plt.savefig("figures/similarity_heatmap.png", bbox_inches="tight")
    plt.close()
    print("  Saved figures/similarity_heatmap.png")


# ── 4. Attack Trajectories ────────────────────────────────────────────────────

def plot_attack_trajectories(results: dict):
    attacks = results.get("attacks", {})
    if not attacks:
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, (pair_key, pair_data) in enumerate(attacks.items()):
        history = pair_data.get("history", [])
        if not history:
            continue
        steps = [h["step"] for h in history]
        sims = [h["cosine_similarity"] for h in history]
        baseline = history[0]["src_tgt_baseline"]
        label = f"{pair_data['source'].split()[0]} → {pair_data['target'].split()[0]}"

        ax.plot(steps, sims, color=PALETTE[i], linewidth=2, label=label, marker="o",
                markersize=3, zorder=3)
        ax.axhline(baseline, color=PALETTE[i], linewidth=0.8, linestyle=":", alpha=0.5)

    ax.axhline(0.9, color="gray", linewidth=1, linestyle="--", alpha=0.7)
    ax.text(0.5, 0.915, "Target sim = 0.9", transform=ax.get_xaxis_transform(),
            color="gray", fontsize=8)

    ax.set_xlabel("Attack Iteration", fontsize=11)
    ax.set_ylabel("Cosine Similarity (adv → target)", fontsize=11)
    ax.set_title("SD-MIAE Attack Trajectories\n"
                 "(dotted baseline = pre-attack source→target similarity)",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9, loc="lower right")
    ax.set_xlim(0, max(h["step"] for pair_data in attacks.values()
                       for h in pair_data.get("history", [{"step": 0}])))
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("figures/attack_trajectories.png", bbox_inches="tight")
    plt.close()
    print("  Saved figures/attack_trajectories.png")


# ── 5. Detection Z-Score Distributions ───────────────────────────────────────

def plot_detection_summary(results: dict):
    detection = results.get("detection", {})
    clean_results = detection.get("clean_embeddings", {})
    attacks = results.get("attacks", {})

    clean_z = [v["mean_z_score"] for v in clean_results.values()]
    adv_z = [
        pair_data["detection"]["mean_z_score"]
        for pair_data in attacks.values()
        if "detection" in pair_data
    ]

    if not clean_z or not adv_z:
        print("  Skipping detection plot (no detection data)")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: distribution comparison
    ax = axes[0]
    bins = np.linspace(0, max(max(clean_z), max(adv_z)) * 1.1, 20)
    ax.hist(clean_z, bins=bins, color=CLEAN_BLUE, alpha=0.7, label="Clean", density=True)
    ax.hist(adv_z, bins=bins, color=ATTACK_RED, alpha=0.7, label="Adversarial", density=True)
    ax.axvline(detection.get("threshold", 2.0), color="black", linewidth=2,
               linestyle="--", label=f"Threshold = {detection.get('threshold', 2.0)}")
    ax.set_xlabel("Mean |z-score|", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.set_title("Z-Score Distribution:\nClean vs Adversarial", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)

    # Right: per-pair bar chart
    ax = axes[1]
    pair_labels = [
        f"{pd['source'].split()[0]}→{pd['target'].split()[0]}"
        for pd in attacks.values()
        if "detection" in pd
    ]
    pair_z = [pd["detection"]["mean_z_score"] for pd in attacks.values() if "detection" in pd]
    colors = [ATTACK_RED if z > detection.get("threshold", 2.0) else CLEAN_BLUE for z in pair_z]

    bars = ax.barh(pair_labels, pair_z, color=colors, edgecolor="white", linewidth=0.5)
    ax.axvline(detection.get("threshold", 2.0), color="black", linewidth=2,
               linestyle="--", label="Detection threshold")
    ax.set_xlabel("Mean |z-score|", fontsize=11)
    ax.set_title("Per-Attack Detection Score", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)

    for bar, val in zip(bars, pair_z):
        ax.text(val + 0.02, bar.get_y() + bar.get_height() / 2,
                f"{val:.2f}", va="center", fontsize=9)

    plt.tight_layout()
    plt.savefig("figures/detection_summary.png", bbox_inches="tight")
    plt.close()
    print("  Saved figures/detection_summary.png")


# ── 6. Attribution Bars ───────────────────────────────────────────────────────

def plot_attribution(results: dict):
    attribution = results.get("detection", {}).get("attribution", {})
    if not attribution:
        print("  Skipping attribution plot (no attribution data)")
        return

    regions = list(attribution.keys())
    scores = [attribution[r] for r in regions]

    fig, ax = plt.subplots(figsize=(9, 5))
    colors = [ATTACK_RED if s == max(scores) else CLEAN_GREEN for s in scores]
    bars = ax.bar(regions, scores, color=colors, edgecolor="white", linewidth=0.5)
    ax.set_ylabel("Mean |z-score| across attacks", fontsize=11)
    ax.set_title("Fingerprint Region Attribution\n"
                 "(which dimensions are most exploited by SD-MIAE)",
                 fontsize=12, fontweight="bold")
    ax.set_xticklabels(regions, rotation=25, ha="right", fontsize=9)

    for bar, val in zip(bars, scores):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{val:.3f}", ha="center", fontsize=8)

    plt.tight_layout()
    plt.savefig("figures/attribution_bars.png", bbox_inches="tight")
    plt.close()
    print("  Saved figures/attribution_bars.png")


# ── 7. Metrics Dashboard ──────────────────────────────────────────────────────

def plot_metrics_dashboard(results: dict):
    pm = results.get("product_metrics", {})
    if not pm:
        print("  Skipping dashboard (run metrics.py first)")
        return

    ari = pm.get("ari", {})
    fps = pm.get("fps", {})
    afhs = pm.get("afhs", 0)
    mean_stq = pm.get("mean_stq", 0)

    fig = plt.figure(figsize=(14, 8))
    gs = GridSpec(2, 4, figure=fig, hspace=0.45, wspace=0.4)

    def gauge(ax, value, title, color, low=0.5, high=0.8):
        theta = np.linspace(0, math.pi, 200)
        ax.plot(np.cos(theta), np.sin(theta), color="#DDDDDD", linewidth=8)
        pct = min(value, 1.0)
        theta_fill = np.linspace(0, math.pi * pct, 200)
        ax.plot(np.cos(theta_fill), np.sin(theta_fill), color=color, linewidth=8)
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-0.2, 1.2)
        ax.axis("off")
        ax.text(0, -0.15, f"{value:.3f}", ha="center", va="center",
                fontsize=20, fontweight="bold", color=color)
        ax.text(0, 0.55, title, ha="center", va="center",
                fontsize=10, fontweight="bold", color="#333333", wrap=True)

    ax1 = fig.add_subplot(gs[0, 0])
    gauge(ax1, afhs, "Artist\nHealth Score", CLEAN_GREEN)

    ax2 = fig.add_subplot(gs[0, 1])
    gauge(ax2, ari.get("ari", 0), "Adversarial\nRobustness (ARI)", CLEAN_BLUE)

    ax3 = fig.add_subplot(gs[0, 2])
    gauge(ax3, fps.get("fps", 0), "Fairness\nParity (FPS)", "#F9A825")

    ax4 = fig.add_subplot(gs[0, 3])
    gauge(ax4, mean_stq, "Style Transfer\nQuality (STQ)", "#6A1B9A")

    # Detection breakdown bar
    ax5 = fig.add_subplot(gs[1, :2])
    det_metrics = results.get("detection", {}).get("metrics", {})
    if det_metrics:
        bar_labels = ["Precision", "Recall", "F1"]
        bar_vals = [det_metrics.get("precision", 0),
                    det_metrics.get("recall", 0),
                    det_metrics.get("f1", 0)]
        bars = ax5.bar(bar_labels, bar_vals, color=[CLEAN_BLUE, ATTACK_RED, CLEAN_GREEN],
                       width=0.5, edgecolor="white")
        ax5.set_ylim(0, 1.1)
        ax5.set_title("Detection Performance", fontsize=11, fontweight="bold")
        ax5.set_ylabel("Score")
        for bar, val in zip(bars, bar_vals):
            ax5.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                     f"{val:.2f}", ha="center", fontsize=11, fontweight="bold")

    # Per-pair attack similarity improvement
    ax6 = fig.add_subplot(gs[1, 2:])
    attacks = results.get("attacks", {})
    if attacks:
        pair_labels = [
            f"{pd['source'].split()[0]}→{pd['target'].split()[0]}"
            for pd in attacks.values()
        ]
        improvements = [
            pd["metrics"].get("absolute_improvement", 0)
            for pd in attacks.values()
        ]
        colors = [ATTACK_RED if v > 0.1 else CLEAN_GREEN for v in improvements]
        ax6.barh(pair_labels, improvements, color=colors, edgecolor="white")
        ax6.axvline(0, color="black", linewidth=1)
        ax6.set_xlabel("Cosine Similarity Improvement", fontsize=10)
        ax6.set_title("Attack Effectiveness per Pair", fontsize=11, fontweight="bold")

    fig.suptitle("Adversarial Music Evaluation  -  Metrics Dashboard",
                 fontsize=14, fontweight="bold", y=1.01)

    plt.savefig("figures/metrics_dashboard.png", bbox_inches="tight")
    plt.close()
    print("  Saved figures/metrics_dashboard.png")


# ─────────────────────────────────────────────────────────────────────────────

def main():
    Path("figures").mkdir(exist_ok=True)

    for path in ["data/fingerprints.json", "data/results.json"]:
        if not Path(path).exists():
            raise FileNotFoundError(f"Run the full pipeline first: missing {path}")

    fp_data, results = load_data()

    print("Generating figures...")
    plot_fingerprint_radar(fp_data)
    plot_pca(fp_data, results)
    plot_similarity_heatmap(fp_data)
    plot_attack_trajectories(results)
    plot_detection_summary(results)
    plot_attribution(results)
    plot_metrics_dashboard(results)

    figs = list(Path("figures").glob("*.png"))
    print(f"\nGenerated {len(figs)} figures in figures/")


if __name__ == "__main__":
    main()
