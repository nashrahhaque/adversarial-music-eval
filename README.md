# Adversarial Music Evaluation
### SD-MIAE Applied to Generative Music Identity Protection

> A research prototype demonstrating how adversarial ML methodology
> can protect artist identity in AI-powered music generation systems —
> built for Spotify's Artist-First AI Music Lab.

---

## Motivation

As generative AI music systems scale to hundreds of millions of listeners,
a new attack surface emerges: **adversarial style injection**. An attacker
can craft a small perturbation δ to an artist's conditioning fingerprint
such that a generative model produces output that *sounds like a different
artist* — silently redirecting royalty flow, distorting artist–fan
connections, and violating the artist's creative identity.

This project adapts **SD-MIAE** (Style-Directed Momentum-Integrated
Adversarial Example attack, from my IEEE publication) to the music domain,
building an end-to-end pipeline for:

1. **Artist DNA fingerprinting** from real streaming data (Last.fm)
2. **Adversarial identity spoofing** using momentum-integrated gradient attacks
3. **Statistical anomaly detection** to flag perturbed embeddings
4. **Product-grade metrics** measuring robustness, fairness, and generation quality
5. **A/B test and causal inference frameworks** for responsible deployment

---

## Pipeline Overview

```
Last.fm API
    │
    ▼
fetch_artists.py ──► data/artists.json
    │                 (tags, tracks, listeners, similar artists)
    ▼
fingerprint.py ───► data/fingerprints.json
    │                 (54-dim artist DNA vectors)
    ▼
attack.py ────────► data/results.json (attacks section)
    │                 (SD-MIAE on 5 artist pairs)
    ▼
detect.py ────────► data/results.json (detection section)
    │                 (z-score anomaly detection)
    ▼
metrics.py ───────► data/results.json (product_metrics section)
    │                 (ADPS, STQ, ARI, FPS, AFHS)
    ▼
eval_framework.py ► data/results.json (eval_framework section)
    │                 (A/B tests, DiD, fairness audit, ecosystem impact)
    ▼
visualize.py ─────► figures/
                      (7 publication-quality figures)
```

One-command run:
```bash
python run_pipeline.py
# or, if you already have data/artists.json:
python run_pipeline.py --skip-fetch
```

---

## Artist DNA Fingerprinting (54 dimensions)

Each artist is encoded into a **54-dimensional vector** from Last.fm data:

| Dims    | Feature Group        | Description                                    |
|---------|----------------------|------------------------------------------------|
| 0–29    | Tag embedding        | Multi-hot over top-30 global tags, weighted    |
| 30      | Listener reach       | Log-normalized monthly listener count          |
| 31      | Play volume          | Log-normalized total play count                |
| 32      | Engagement ratio     | Plays-per-listener, normalized                 |
| 33      | Tag diversity        | Unique tag breadth (0–1)                       |
| 34–43   | Similar-artist graph | Similarity scores to each of the 10 artists   |
| 44–48   | Track statistics     | Mean/std/max plays, track count, mean listeners|
| 49–53   | Tag weight stats     | Tag weight distribution (mean, std, entropy…)  |

This fingerprint serves as a proxy for the **conditioning vector** a
generative model would use to steer musical style — the same vector
an adversary would perturb.

---

## SD-MIAE Attack

**Objective:** find δ such that:
- `model(source + δ) ≈ model(target)` — high cosine similarity in latent space
- `‖δ‖∞ ≤ ε = 0.05` — imperceptibility constraint

**Algorithm** (momentum-integrated sign gradient):
```
g₀ = 0
for t = 1..T:
    L = −cos_sim(model(x + δ), model(xₜ))
    ĝₜ = ∇L / ‖∇L‖₁
    gₜ = μ · gₜ₋₁ + ĝₜ            # momentum accumulation
    δ  = δ − α · sign(gₜ)          # sign update
    δ  = clip(δ, −ε, ε)            # L∞ projection
```

| Param | Value | Rationale |
|-------|-------|-----------|
| ε     | 0.05  | 5% max feature perturbation — imperceptible to metadata filters |
| μ     | 0.9   | Strong momentum to escape flat cosine-similarity landscapes    |
| α     | 0.01  | Conservative step size for stable convergence                  |
| T     | 20    | Sufficient for >90% similarity in 4/5 pairs                    |

**Attack pairs:**

| Source           | Target         | Rationale                          |
|------------------|----------------|------------------------------------|
| Taylor Swift     | Olivia Rodrigo | Pop adjacency, generational bridge |
| Drake            | 21 Savage      | Hip-hop collaboration network      |
| Billie Eilish    | Lorde          | Alt-pop sonic sisters              |
| The Weeknd       | Post Malone    | R&B/pop-trap overlap               |
| Kendrick Lamar   | J. Cole        | Lyrical rap peer set               |

Pairs are chosen to be *plausible but distinct* — similar enough to test
the attack, distinct enough that success demonstrates real capability.

---

## Detection

The anomaly detector uses **per-dimension z-score deviation** from the
clean embedding distribution:

```
z_d = |x_d − μ_d| / σ_d      for each dimension d
flag if mean(z) > 2.0
```

Confidence is calibrated as:
```
confidence = sigmoid(2 × (mean_z − threshold))
```

**Detection metrics** include precision, recall, F1, and AUROC over
the joint clean + adversarial embedding set.

**Attribution analysis** identifies which fingerprint regions (tags,
similar-artist graph, track stats, etc.) are most exploited — directly
informing which features to harden in a production system.

---

## Product Metrics

Five metrics designed for a Data Scientist to *own end-to-end*:

| Metric | Abbr | Formula | Measures |
|--------|------|---------|---------|
| Artist DNA Preservation Score | ADPS | cos_sim(original, generated) | Identity fidelity |
| Style Transfer Quality | STQ | H-mean(fidelity, imperceptibility) | Attack quality |
| Adversarial Robustness Index | ARI | H-mean(detection_rate, confidence) | System safety |
| Fairness Parity Score | FPS | 1 − Gini(per-artist protection) | Equitable protection |
| Artist-First Health Score | AFHS | 0.4×ARI + 0.3×FPS + 0.3×STQ | Composite dashboard KPI |

AFHS is designed for a **real-time monitoring dashboard** — a single
number that surfaces when any of the three pillars (safety, fairness,
quality) degrades.

---

## Evaluation Framework

Beyond the core attack/detect loop, `eval_framework.py` provides:

### A/B Test Power Analysis
Pre-computed sample sizes and runtimes for four generative music metrics
at Spotify scale (~25M daily eligible users):

| Metric                   | n/variant | Runtime  | MDE   |
|--------------------------|-----------|----------|-------|
| Stream completion rate   | ~65,000   | <1 day   | +3%   |
| Artist save rate         | ~25,000   | <1 day   | +10%  |
| Skip rate                | ~45,000   | <1 day   | −5%   |
| New artist discovery     | ~30,000   | <1 day   | +8%   |

At Spotify's scale, most experiments can reach significance within a
single day — but **novelty effects** and **carryover bias** still
require minimum 1–2 week holdouts.

### Causal Inference
Difference-in-Differences estimator with bootstrap confidence intervals
for observational rollout analysis — for when randomization isn't possible
(e.g., model updates that can't be cleanly A/B split).

### Fairness Audit
Disparate impact analysis using the **4/5ths rule** across genre, tier,
and gender groups. Surfaces whether the detector protects all artists
equitably — a critical artist-first requirement.

### Ecosystem Impact Model
Estimates the **royalty-at-risk** if adversarial identity spoofing goes
undetected at scale:
- At 5M daily generations with 1% attack rate and 80% detection:
  ~$4K/day in redirected royalties would be caught, ~$1K/day missed
- Annual protected artist royalty value: ~$1.5M (conservative estimate)

---

## Figures

| File | Description |
|------|-------------|
| `fingerprint_radar.png` | Per-artist DNA radar charts across 8 feature regions |
| `pca_fingerprints.png` | 2D PCA of artist fingerprint space with attack arrows |
| `similarity_heatmap.png` | Pairwise cosine similarity matrix |
| `attack_trajectories.png` | SD-MIAE convergence curves per pair |
| `detection_summary.png` | Z-score distributions: clean vs adversarial |
| `attribution_bars.png` | Which fingerprint regions are most exploited |
| `metrics_dashboard.png` | Composite metrics summary panel |

---

## Setup

```bash
# Create virtual environment
python -m venv .venv && source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Set API key (or leave hardcoded default)
export LASTFM_API_KEY="your_key_here"

# Run full pipeline
python run_pipeline.py

# Or step by step
python fetch_artists.py
python fingerprint.py
python attack.py
python detect.py
python metrics.py
python eval_framework.py
python visualize.py
```

---

## Connection to SD-MIAE (IEEE Paper)

The original SD-MIAE paper demonstrates adversarial attacks on image
classifiers with style-directed objectives. This project adapts the
core methodology to audio/music fingerprint spaces:

| Original (Image Domain) | This Project (Music Domain) |
|-------------------------|------------------------------|
| Pixel perturbation δ | Fingerprint feature perturbation δ |
| Classifier logits | Generative model latent embedding |
| Style loss (Gram matrix) | Cosine similarity to target artist |
| L∞ pixel budget | L∞ feature budget (ε = 0.05) |
| Momentum gradient sign | Momentum gradient sign (identical) |

The key insight transfers directly: **momentum accumulation prevents
oscillation in flat loss landscapes**, which is especially important when
the source and target embeddings are in similar regions of the space
(e.g., hip-hop artists with overlapping tag distributions).

---

## Artist-First Design Principles

Every design choice in this system reflects Spotify's stated principles:

- **Choice**: the fairness audit ensures no artist is disproportionately
  vulnerable to spoofing attacks
- **Transparency**: attribution analysis explains *which* features are
  perturbed, enabling artist-legible explanations
- **Fair compensation**: the ecosystem impact model quantifies royalty
  protection in dollar terms
- **Artist–fan connection**: ADPS directly measures whether generation
  preserves the artist identity that fans came to experience

---

*Built as a technical demonstration for the Data Scientist role,
Spotify Artist-First AI Music Lab.*
