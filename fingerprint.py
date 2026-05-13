"""
Artist DNA fingerprint builder.

Encodes Last.fm artist data into a 54-dimensional numpy vector:
  [0:30]  Multi-hot tag vector over the top-30 global tags
  [30:32] Log-normalized listener count, log-normalized play count
  [32]    Engagement ratio (playcount / listeners), normalized
  [33]    Tag diversity (unique tags / 15)
  [34:44] Similar-artist overlap: for each of the 10 canonical artists,
          the similarity score (0 if not present)
  [44:49] Track statistics (mean/std/max play count + track count + mean listeners),
          all log-normalized
  [49:54] Tag-weight statistics (mean weight, std weight, top weight,
          weight entropy, weighted unique-tag count), normalized

Saves fingerprints and the tag vocabulary to data/fingerprints.json.
"""

import json
import math
import numpy as np
from pathlib import Path

ARTISTS = [
    "Taylor Swift", "Drake", "Billie Eilish", "The Weeknd",
    "Kendrick Lamar", "Olivia Rodrigo", "21 Savage", "Lorde",
    "Post Malone", "J. Cole",
]

FINGERPRINT_DIM = 54

# ── Dimension layout ──────────────────────────────────────────────────────────
DIM_TAGS_START = 0
DIM_TAGS_END = 30          # 30 dims
DIM_LISTENERS = 30         # 1 dim
DIM_PLAYCOUNT = 31         # 1 dim
DIM_ENGAGEMENT = 32        # 1 dim
DIM_TAG_DIVERSITY = 33     # 1 dim
DIM_SIMILAR_START = 34     # 10 dims
DIM_SIMILAR_END = 44
DIM_TRACK_START = 44       # 5 dims
DIM_TRACK_END = 49
DIM_TAGWEIGHT_START = 49   # 5 dims
DIM_TAGWEIGHT_END = 54


def log_norm(value: float, scale: float = 1e7) -> float:
    return math.log1p(max(value, 0)) / math.log1p(scale)


def build_tag_vocabulary(artists_data: dict, top_k: int = 30) -> list[str]:
    """Collect global tag counts and return the top-k tags."""
    tag_totals: dict[str, float] = {}
    for artist_data in artists_data.values():
        for tag in artist_data.get("tags", []):
            name = tag["name"]
            tag_totals[name] = tag_totals.get(name, 0) + tag["count"]
    sorted_tags = sorted(tag_totals.items(), key=lambda x: -x[1])
    return [t[0] for t in sorted_tags[:top_k]]


def tag_entropy(weights: list[float]) -> float:
    total = sum(weights)
    if total == 0:
        return 0.0
    probs = [w / total for w in weights if w > 0]
    return -sum(p * math.log2(p) for p in probs) / math.log2(max(len(probs), 2))


def build_fingerprint(
    artist_data: dict,
    vocab: list[str],
    artist_index: dict[str, int],
) -> np.ndarray:
    vec = np.zeros(FINGERPRINT_DIM, dtype=np.float32)

    # ── [0:30] Multi-hot tags ─────────────────────────────────────────────────
    tag_weight_map = {t["name"]: t["count"] for t in artist_data.get("tags", [])}
    max_tag_count = max(tag_weight_map.values(), default=1)
    for i, tag in enumerate(vocab):
        if tag in tag_weight_map:
            vec[DIM_TAGS_START + i] = tag_weight_map[tag] / max_tag_count

    # ── [30] Listener count (log-normalized) ──────────────────────────────────
    vec[DIM_LISTENERS] = log_norm(artist_data.get("listeners", 0), scale=1e8)

    # ── [31] Play count (log-normalized) ─────────────────────────────────────
    vec[DIM_PLAYCOUNT] = log_norm(artist_data.get("playcount", 0), scale=1e10)

    # ── [32] Engagement ratio ─────────────────────────────────────────────────
    listeners = max(artist_data.get("listeners", 1), 1)
    playcount = artist_data.get("playcount", 0)
    vec[DIM_ENGAGEMENT] = min(playcount / listeners / 500.0, 1.0)

    # ── [33] Tag diversity ────────────────────────────────────────────────────
    vec[DIM_TAG_DIVERSITY] = len(artist_data.get("tags", [])) / 15.0

    # ── [34:44] Similar-artist overlap ───────────────────────────────────────
    similar_map = {
        s["name"].lower(): s["match"]
        for s in artist_data.get("similar_artists", [])
    }
    for art, idx in artist_index.items():
        score = similar_map.get(art.lower(), 0.0)
        vec[DIM_SIMILAR_START + idx] = float(score)

    # ── [44:49] Track statistics ──────────────────────────────────────────────
    tracks = artist_data.get("top_tracks", [])
    if tracks:
        playcounts = [t.get("playcount", 0) for t in tracks]
        track_listeners = [t.get("listeners", 0) for t in tracks]
        vec[DIM_TRACK_START + 0] = log_norm(np.mean(playcounts), scale=1e7)
        vec[DIM_TRACK_START + 1] = log_norm(np.std(playcounts), scale=1e6)
        vec[DIM_TRACK_START + 2] = log_norm(np.max(playcounts), scale=1e7)
        vec[DIM_TRACK_START + 3] = min(len(tracks) / 10.0, 1.0)
        vec[DIM_TRACK_START + 4] = log_norm(np.mean(track_listeners), scale=1e6)

    # ── [49:54] Tag-weight statistics ─────────────────────────────────────────
    tag_weights = [t["count"] for t in artist_data.get("tags", [])]
    if tag_weights:
        total_w = sum(tag_weights)
        vec[DIM_TAGWEIGHT_START + 0] = np.mean(tag_weights) / 100.0
        vec[DIM_TAGWEIGHT_START + 1] = np.std(tag_weights) / 100.0
        vec[DIM_TAGWEIGHT_START + 2] = max(tag_weights) / 100.0
        vec[DIM_TAGWEIGHT_START + 3] = tag_entropy(tag_weights)
        vec[DIM_TAGWEIGHT_START + 4] = min(len(tag_weights) / 15.0, 1.0)

    return np.clip(vec, 0.0, 1.0)


def main():
    artists_path = Path("data/artists.json")
    if not artists_path.exists():
        raise FileNotFoundError("Run fetch_artists.py first to generate data/artists.json")

    with open(artists_path) as f:
        artists_data = json.load(f)

    vocab = build_tag_vocabulary(artists_data, top_k=30)
    artist_index = {name: i for i, name in enumerate(ARTISTS)}

    print(f"Tag vocabulary ({len(vocab)} tags): {vocab[:10]} ...")

    fingerprints = {}
    for artist in ARTISTS:
        if artist not in artists_data or "error" in artists_data[artist]:
            print(f"  SKIP {artist} (no data)")
            continue
        vec = build_fingerprint(artists_data[artist], vocab, artist_index)
        fingerprints[artist] = {
            "vector": vec.tolist(),
            "dim": int(vec.shape[0]),
            "norm_l2": float(np.linalg.norm(vec)),
        }
        print(f"  {artist:20s}  ||v||={np.linalg.norm(vec):.4f}  "
              f"sparsity={float((vec == 0).mean()):.2%}")

    output = {
        "vocabulary": vocab,
        "artist_index": artist_index,
        "fingerprints": fingerprints,
        "dim": FINGERPRINT_DIM,
        "dim_layout": {
            "tags_0_30": "multi-hot tag weights (top-30 global tags)",
            "listeners_30": "log-normalized listener count",
            "playcount_31": "log-normalized play count",
            "engagement_32": "playcount/listeners ratio, normalized",
            "tag_diversity_33": "unique tag fraction",
            "similar_34_44": "similar-artist match scores for each of the 10 artists",
            "track_stats_44_49": "track playcount/listener statistics, log-normalized",
            "tag_weight_stats_49_54": "tag weight distribution statistics",
        },
    }

    out_path = "data/fingerprints.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nSaved {len(fingerprints)} fingerprints ({FINGERPRINT_DIM}-dim) to {out_path}")


if __name__ == "__main__":
    main()
