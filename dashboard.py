"""
Streamlit dashboard for adversarial-music-eval.

Presents the full SD-MIAE pipeline results in an interactive UI
that Spotify reviewers can explore without running any code.

Deploy to Streamlit Community Cloud:
    1. Push repo to GitHub
    2. Go to share.streamlit.io → New app → select this file
    3. Add LASTFM_API_KEY to Secrets if you want live fetching

Or run locally:
    streamlit run dashboard.py
"""

import json
import math
import pathlib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.decomposition import PCA
from scipy.spatial.distance import cosine as cosine_dist

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Adversarial Music Eval",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded",
)

SPOTIFY_GREEN = "#1DB954"
ATTACK_RED = "#E53935"
CLEAN_BLUE = "#1565C0"
BG = "#191414"
CARD_BG = "#121212"

ARTISTS = [
    "Taylor Swift", "Drake", "Billie Eilish", "The Weeknd",
    "Kendrick Lamar", "Olivia Rodrigo", "21 Savage", "Lorde",
    "Post Malone", "J. Cole",
]

PALETTE = px.colors.qualitative.Plotly

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

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .stApp { background-color: #191414; color: #FFFFFF; }
    .metric-card {
        background: #282828;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        border: 1px solid #333;
    }
    .metric-value { font-size: 2.2rem; font-weight: 700; color: #1DB954; }
    .metric-label { font-size: 0.85rem; color: #B3B3B3; margin-top: 4px; }
    .section-header {
        font-size: 1.4rem;
        font-weight: 700;
        color: #FFFFFF;
        border-bottom: 2px solid #1DB954;
        padding-bottom: 8px;
        margin: 24px 0 16px 0;
    }
    .attack-pair-card {
        background: #282828;
        border-radius: 8px;
        padding: 16px;
        margin: 8px 0;
        border-left: 3px solid #E53935;
    }
</style>
""", unsafe_allow_html=True)


# ── Data loading ──────────────────────────────────────────────────────────────

@st.cache_data
def load_data():
    data_dir = pathlib.Path(__file__).parent / "data"
    fp_path = data_dir / "fingerprints.json"
    res_path = data_dir / "results.json"

    if not fp_path.exists() or not res_path.exists():
        return None, None

    with open(fp_path) as f:
        fp_data = json.load(f)
    with open(res_path) as f:
        results = json.load(f)
    return fp_data, results


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.image(
        "https://storage.googleapis.com/pr-newsroom-wp/1/2018/11/Spotify_Logo_RGB_White.png",
        width=130,
    )
    st.markdown("## Adversarial Music Eval")
    st.markdown(
        "**SD-MIAE** applied to generative music identity protection.\n\n"
        "Adapted from a published IEEE adversarial-attack methodology to "
        "detect style-spoofing attacks on artist DNA fingerprints."
    )
    st.markdown("---")
    st.markdown("**Navigation**")
    page = st.radio(
        "",
        ["Overview", "Artist Fingerprints", "Attack Analysis",
         "Detection Results", "Product Metrics", "Eval Framework"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.markdown(
        "[GitHub](https://github.com) · "
        "[IEEE Paper](#) · "
        "[Run Pipeline](https://github.com)"
    )

# ── Load ──────────────────────────────────────────────────────────────────────

fp_data, results = load_data()

if fp_data is None:
    st.error(
        "No data found. Run `python run_pipeline.py` to generate data/, "
        "then restart the dashboard."
    )
    st.stop()

fingerprints = fp_data["fingerprints"]
vocab = fp_data["vocabulary"]
attacks = results.get("attacks", {})
detection = results.get("detection", {})
pm = results.get("product_metrics", {})
ef = results.get("eval_framework", {})

artist_vecs = {a: np.array(fingerprints[a]["vector"]) for a in ARTISTS if a in fingerprints}

# ─────────────────────────────────────────────────────────────────────────────
# PAGE: OVERVIEW
# ─────────────────────────────────────────────────────────────────────────────

if page == "Overview":
    st.markdown("# Adversarial Music Evaluation")
    st.markdown(
        "An end-to-end research prototype demonstrating how adversarial ML "
        "methodology protects **artist identity** in AI-powered generative music systems."
    )

    # Top KPI row
    c1, c2, c3, c4, c5 = st.columns(5)
    kpi_pairs = [
        ("Artist-First Health Score", f"{pm.get('afhs', 0):.3f}", SPOTIFY_GREEN),
        ("Detection F1", f"{detection.get('metrics', {}).get('f1', 0):.3f}", SPOTIFY_GREEN),
        ("Robustness (ARI)", f"{pm.get('ari', {}).get('ari', 0):.3f}", CLEAN_BLUE),
        ("Fairness (FPS)", f"{pm.get('fps', {}).get('fps', 0):.3f}", "#F9A825"),
        ("Style Quality (STQ)", f"{pm.get('mean_stq', 0):.3f}", "#AB47BC"),
    ]
    for col, (label, value, color) in zip([c1, c2, c3, c4, c5], kpi_pairs):
        col.markdown(
            f'<div class="metric-card">'
            f'<div class="metric-value" style="color:{color}">{value}</div>'
            f'<div class="metric-label">{label}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.markdown("---")

    col_l, col_r = st.columns([2, 1])

    with col_l:
        st.markdown('<div class="section-header">Pipeline Overview</div>', unsafe_allow_html=True)
        st.markdown("""
| Step | Module | What it does |
|------|--------|-------------|
| 1 | `fetch_artists.py` | Last.fm API → tags, tracks, listeners for 10 artists |
| 2 | `fingerprint.py` | 54-dim artist DNA vector (tags · stats · graph) |
| 3 | `attack.py` | SD-MIAE adversarial attack on 5 artist pairs |
| 4 | `detect.py` | Z-score anomaly detection (threshold = 2.0) |
| 5 | `metrics.py` | ADPS · STQ · ARI · FPS · AFHS product metrics |
| 6 | `eval_framework.py` | A/B tests · causal inference · fairness · ecosystem impact |
        """)

    with col_r:
        st.markdown('<div class="section-header">Attack Pairs</div>', unsafe_allow_html=True)
        for pair_key, pd_ in attacks.items():
            sim = pd_["metrics"]["attack_similarity"]
            delta = pd_["metrics"]["absolute_improvement"]
            det = pd_.get("detection", {})
            flag = "⚠️ DETECTED" if det.get("is_adversarial") else "✓ missed"
            st.markdown(
                f'<div class="attack-pair-card">'
                f'<b>{pd_["source"].split()[0]} → {pd_["target"].split()[0]}</b><br>'
                f'sim={sim:.3f} ({delta:+.3f})  {flag}'
                f'</div>',
                unsafe_allow_html=True,
            )

    # Artist listener breakdown
    st.markdown('<div class="section-header">Artist Dataset</div>', unsafe_allow_html=True)
    with open(pathlib.Path(__file__).parent / "data" / "artists.json") as f:
        artists_raw = json.load(f)

    df_artists = pd.DataFrame([
        {
            "Artist": a,
            "Listeners": artists_raw[a].get("listeners", 0),
            "Playcount": artists_raw[a].get("playcount", 0),
            "Top Tags": ", ".join(t["name"] for t in artists_raw[a].get("tags", [])[:3]),
        }
        for a in ARTISTS if a in artists_raw and "error" not in artists_raw[a]
    ])
    df_artists["Listeners (M)"] = (df_artists["Listeners"] / 1e6).round(2)
    df_artists["Plays (B)"] = (df_artists["Playcount"] / 1e9).round(2)

    fig = px.bar(
        df_artists, x="Artist", y="Listeners (M)",
        color="Listeners (M)", color_continuous_scale="Greens",
        title="Monthly Listeners (Last.fm)",
        template="plotly_dark",
    )
    fig.update_layout(showlegend=False, paper_bgcolor=CARD_BG, plot_bgcolor=CARD_BG,
                      coloraxis_showscale=False)
    st.plotly_chart(fig, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: ARTIST FINGERPRINTS
# ─────────────────────────────────────────────────────────────────────────────

elif page == "Artist Fingerprints":
    st.markdown("# Artist DNA Fingerprints")
    st.markdown(
        "Each artist is encoded as a **54-dimensional vector** derived from Last.fm data. "
        "The fingerprint acts as the conditioning vector for a generative music model."
    )

    tab1, tab2, tab3 = st.tabs(["Radar Charts", "PCA Space", "Similarity Heatmap"])

    with tab1:
        selected = st.multiselect(
            "Select artists to compare",
            ARTISTS,
            default=ARTISTS[:4],
        )
        if selected:
            regions = list(DIM_LAYOUT.keys())
            fig = go.Figure()
            for i, artist in enumerate(selected):
                if artist not in artist_vecs:
                    continue
                vec = artist_vecs[artist]
                vals = [float(vec[s:e].mean()) for s, e in DIM_LAYOUT.values()]
                vals += vals[:1]
                angles = [r for r in regions] + [regions[0]]
                fig.add_trace(go.Scatterpolar(
                    r=vals,
                    theta=angles,
                    fill="toself",
                    name=artist,
                    opacity=0.7,
                    line=dict(color=PALETTE[i % len(PALETTE)], width=2),
                ))
            fig.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                showlegend=True,
                title="Artist DNA Radar (per region mean)",
                template="plotly_dark",
                paper_bgcolor=CARD_BG,
            )
            st.plotly_chart(fig, use_container_width=True)

        # Fingerprint dim layout table
        st.markdown("#### 54-Dimension Layout")
        dim_df = pd.DataFrame([
            {"Dims": f"{s}–{e-1}", "Region": region, "Description": desc}
            for (region, (s, e)), desc in zip(
                DIM_LAYOUT.items(),
                [
                    "Multi-hot tag weights over top-30 global tags",
                    "Log-normalized monthly listener count",
                    "Log-normalized total play count",
                    "Plays-per-listener ratio, normalized",
                    "Unique tag breadth (0–1)",
                    "Similarity scores to each of the 10 canonical artists",
                    "Track play/listener stats, log-normalized",
                    "Tag weight distribution statistics",
                ],
            )
        ])
        st.dataframe(dim_df, use_container_width=True, hide_index=True)

    with tab2:
        names = list(artist_vecs.keys())
        vecs = np.array([artist_vecs[a] for a in names])
        pca = PCA(n_components=2, random_state=42)
        coords = pca.fit_transform(vecs)
        var_exp = pca.explained_variance_ratio_

        fig = go.Figure()
        for i, (name, (x, y)) in enumerate(zip(names, coords)):
            fig.add_trace(go.Scatter(
                x=[x], y=[y],
                mode="markers+text",
                marker=dict(size=16, color=PALETTE[i % len(PALETTE)]),
                text=[name.split()[0]],
                textposition="top center",
                name=name,
            ))

        # Attack arrows
        for _, pd_ in attacks.items():
            src, tgt = pd_["source"], pd_["target"]
            if src in names and tgt in names:
                si, ti = names.index(src), names.index(tgt)
                fig.add_annotation(
                    x=coords[ti][0], y=coords[ti][1],
                    ax=coords[si][0], ay=coords[si][1],
                    xref="x", yref="y", axref="x", ayref="y",
                    arrowhead=3, arrowcolor=ATTACK_RED,
                    arrowwidth=2, arrowsize=1.2,
                )

        fig.update_layout(
            xaxis_title=f"PC1 ({var_exp[0]:.1%} var)",
            yaxis_title=f"PC2 ({var_exp[1]:.1%} var)",
            title="Fingerprint PCA Space — red arrows = attack direction",
            template="plotly_dark",
            paper_bgcolor=CARD_BG,
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        names = list(artist_vecs.keys())
        vecs = np.array([artist_vecs[a] for a in names])
        sim_mat = np.array([
            [1.0 - cosine_dist(vecs[i], vecs[j]) for j in range(len(names))]
            for i in range(len(names))
        ])
        short = [n.split()[0] for n in names]
        fig = px.imshow(
            sim_mat,
            x=short, y=short,
            color_continuous_scale="RdYlGn",
            zmin=0, zmax=1,
            text_auto=".2f",
            title="Pairwise Cosine Similarity",
            template="plotly_dark",
        )
        fig.update_layout(paper_bgcolor=CARD_BG)
        st.plotly_chart(fig, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: ATTACK ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

elif page == "Attack Analysis":
    st.markdown("# SD-MIAE Attack Analysis")
    st.markdown(
        "Momentum-integrated sign-gradient attack adapted from the IEEE SD-MIAE paper. "
        "Finds δ such that `model(source + δ) ≈ model(target)` with `‖δ‖∞ ≤ 0.05`."
    )

    # Hyperparams banner
    cfg = results.get("config", {})
    c1, c2, c3, c4 = st.columns(4)
    for col, (k, v) in zip([c1, c2, c3, c4], [("ε (budget)", cfg.get("epsilon")),
                                                 ("μ (momentum)", cfg.get("mu")),
                                                 ("α (step size)", cfg.get("alpha")),
                                                 ("T (steps)", cfg.get("steps"))]):
        col.metric(k, v)

    st.markdown("---")

    # Convergence trajectories
    st.markdown("#### Convergence Trajectories")
    fig = go.Figure()
    for i, (pair_key, pd_) in enumerate(attacks.items()):
        history = pd_.get("history", [])
        if not history:
            continue
        steps = [h["step"] for h in history]
        sims = [h["cosine_similarity"] for h in history]
        baseline = history[0]["src_tgt_baseline"]
        label = f"{pd_['source'].split()[0]} → {pd_['target'].split()[0]}"
        fig.add_trace(go.Scatter(
            x=steps, y=sims,
            mode="lines+markers",
            name=label,
            line=dict(color=PALETTE[i % len(PALETTE)], width=2),
            marker=dict(size=4),
        ))
        fig.add_hline(
            y=baseline, line_dash="dot",
            line_color=PALETTE[i % len(PALETTE)],
            opacity=0.4,
            annotation_text=f"baseline {label[:5]}",
            annotation_position="right",
        )
    fig.update_layout(
        xaxis_title="Attack Iteration",
        yaxis_title="Cosine Similarity (adv → target)",
        title="SD-MIAE Convergence (dotted = pre-attack baseline)",
        template="plotly_dark",
        paper_bgcolor=CARD_BG,
        yaxis=dict(range=[0, 1.05]),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Per-pair results table
    st.markdown("#### Per-Pair Results")
    rows = []
    for pair_key, pd_ in attacks.items():
        m = pd_["metrics"]
        det = pd_.get("detection", {})
        rows.append({
            "Pair": pair_key,
            "Baseline Sim": round(m["baseline_similarity"], 4),
            "Attack Sim": round(m["attack_similarity"], 4),
            "Improvement": f"{m['absolute_improvement']:+.4f}",
            "‖δ‖∞": round(m["delta_linf"], 4),
            "‖δ‖₂": round(m["delta_l2"], 4),
            "Constraint ✓": "✓" if m["constraint_satisfied"] else "✗",
            "Detected": "⚠️ YES" if det.get("is_adversarial") else "✗ NO",
            "Confidence": round(det.get("confidence", 0), 3),
        })
    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)

    # Delta distribution
    st.markdown("#### Perturbation Distribution (δ per dimension)")
    pair_sel = st.selectbox("Select attack pair", list(attacks.keys()))
    if pair_sel in attacks:
        delta = np.array(attacks[pair_sel]["delta"])
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=list(range(len(delta))),
            y=delta,
            marker_color=[ATTACK_RED if d > 0 else CLEAN_BLUE for d in delta],
            name="δ",
        ))
        fig.add_hline(y=cfg.get("epsilon", 0.05), line_dash="dash",
                      line_color="white", annotation_text="+ε")
        fig.add_hline(y=-cfg.get("epsilon", 0.05), line_dash="dash",
                      line_color="white", annotation_text="-ε")
        # Region dividers
        for region, (s, e) in DIM_LAYOUT.items():
            fig.add_vrect(x0=s, x1=e, fillcolor="white", opacity=0.03,
                          annotation_text=region[:4], annotation_position="top left",
                          annotation_font_size=8)
        fig.update_layout(
            xaxis_title="Fingerprint Dimension",
            yaxis_title="Perturbation δ",
            title=f"Perturbation Vector: {pair_sel}",
            template="plotly_dark",
            paper_bgcolor=CARD_BG,
        )
        st.plotly_chart(fig, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: DETECTION RESULTS
# ─────────────────────────────────────────────────────────────────────────────

elif page == "Detection Results":
    st.markdown("# Statistical Anomaly Detection")
    st.markdown(
        "Z-score deviation from clean embedding distribution. "
        f"**Threshold = {detection.get('threshold', 2.0)}**. "
        "Embedding flagged as adversarial if `mean |z| > threshold`."
    )

    det_metrics = detection.get("metrics", {})
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Precision", det_metrics.get("precision", 0))
    c2.metric("Recall", det_metrics.get("recall", 0))
    c3.metric("F1", det_metrics.get("f1", 0))
    c4.metric("AUROC", det_metrics.get("auroc", 0))

    st.markdown("---")
    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown("#### Clean Embeddings")
        clean_data = detection.get("clean_embeddings", {})
        clean_rows = [
            {"Artist": a, "Mean |z|": round(v["mean_z_score"], 3),
             "Max |z|": round(v["max_z_score"], 3),
             "Status": "✓ Clean" if not v["is_adversarial"] else "⚠️ Flagged"}
            for a, v in clean_data.items()
        ]
        df_clean = pd.DataFrame(clean_rows)
        st.dataframe(df_clean, use_container_width=True, hide_index=True)

    with col_r:
        st.markdown("#### Adversarial Embeddings")
        adv_rows = [
            {"Pair": k,
             "Mean |z|": round(v["detection"]["mean_z_score"], 3),
             "Confidence": round(v["detection"].get("confidence", 0), 3),
             "Status": "⚠️ DETECTED" if v["detection"]["is_adversarial"] else "✗ Missed"}
            for k, v in attacks.items() if "detection" in v
        ]
        df_adv = pd.DataFrame(adv_rows)
        st.dataframe(df_adv, use_container_width=True, hide_index=True)

    # Z-score distribution comparison
    clean_zs = [v["mean_z_score"] for v in clean_data.values()]
    adv_zs = [v["detection"]["mean_z_score"] for v in attacks.values() if "detection" in v]
    threshold = detection.get("threshold", 2.0)

    fig = go.Figure()
    fig.add_trace(go.Histogram(x=clean_zs, name="Clean", nbinsx=10,
                               marker_color=CLEAN_BLUE, opacity=0.7))
    fig.add_trace(go.Histogram(x=adv_zs, name="Adversarial", nbinsx=10,
                               marker_color=ATTACK_RED, opacity=0.7))
    fig.add_vline(x=threshold, line_dash="dash", line_color="white",
                  annotation_text=f"Threshold = {threshold}", annotation_position="top right")
    fig.update_layout(
        barmode="overlay",
        xaxis_title="Mean |z-score|",
        yaxis_title="Count",
        title="Z-Score Distribution: Clean vs Adversarial",
        template="plotly_dark",
        paper_bgcolor=CARD_BG,
    )
    st.plotly_chart(fig, use_container_width=True)

    # Attribution analysis
    attribution = detection.get("attribution", {})
    if attribution:
        st.markdown("#### Region Attribution (Most Exploited Dimensions)")
        fig = px.bar(
            x=list(attribution.keys()),
            y=list(attribution.values()),
            color=list(attribution.values()),
            color_continuous_scale=[[0, CLEAN_BLUE], [0.5, "#F9A825"], [1, ATTACK_RED]],
            title="Mean |z-score| per fingerprint region across all attacks",
            template="plotly_dark",
        )
        fig.update_layout(paper_bgcolor=CARD_BG, coloraxis_showscale=False,
                          xaxis_title="Fingerprint Region", yaxis_title="Mean |z|")
        st.plotly_chart(fig, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: PRODUCT METRICS
# ─────────────────────────────────────────────────────────────────────────────

elif page == "Product Metrics":
    st.markdown("# Product Metrics")
    st.markdown(
        "Five metrics a Data Scientist would **own end-to-end** for Spotify's "
        "Artist-First AI Music Lab — spanning safety, fairness, and quality."
    )

    # Gauge row
    ari = pm.get("ari", {}).get("ari", 0)
    fps = pm.get("fps", {}).get("fps", 0)
    stq = pm.get("mean_stq", 0)
    afhs = pm.get("afhs", 0)
    f1 = detection.get("metrics", {}).get("f1", 0)

    def make_gauge(value, title, color):
        return go.Figure(go.Indicator(
            mode="gauge+number",
            value=value,
            title={"text": title, "font": {"size": 13}},
            number={"font": {"size": 28, "color": color}},
            gauge={
                "axis": {"range": [0, 1]},
                "bar": {"color": color},
                "bgcolor": "#282828",
                "steps": [
                    {"range": [0, 0.5], "color": "#333"},
                    {"range": [0.5, 0.8], "color": "#2a2a2a"},
                    {"range": [0.8, 1], "color": "#222"},
                ],
                "threshold": {"line": {"color": "white", "width": 2}, "value": 0.8},
            }
        )).update_layout(
            paper_bgcolor=CARD_BG, font_color="white",
            height=200, margin=dict(l=20, r=20, t=40, b=10)
        )

    g1, g2, g3, g4, g5 = st.columns(5)
    g1.plotly_chart(make_gauge(afhs, "AFHS", SPOTIFY_GREEN), use_container_width=True)
    g2.plotly_chart(make_gauge(ari, "ARI", CLEAN_BLUE), use_container_width=True)
    g3.plotly_chart(make_gauge(fps, "FPS", "#F9A825"), use_container_width=True)
    g4.plotly_chart(make_gauge(stq, "STQ", "#AB47BC"), use_container_width=True)
    g5.plotly_chart(make_gauge(f1, "Detection F1", SPOTIFY_GREEN), use_container_width=True)

    # Metric definitions
    st.markdown("#### Metric Definitions")
    st.markdown("""
| Metric | Full Name | Formula | What it measures |
|--------|-----------|---------|-----------------|
| **ADPS** | Artist DNA Preservation Score | `cos_sim(original, generated)` | Does generation stay true to the source artist? |
| **STQ** | Style Transfer Quality | `H-mean(fidelity, imperceptibility)` | How faithfully is style transferred while staying imperceptible? |
| **ARI** | Adversarial Robustness Index | `H-mean(detection_rate, confidence)` | How resilient is the system to identity-spoofing? |
| **FPS** | Fairness Parity Score | `1 - Gini(per-artist protection)` | Are all artists protected equally? |
| **AFHS** | Artist-First Health Score | `0.4×ARI + 0.3×FPS + 0.3×STQ` | Composite dashboard KPI |
    """)

    # Per-pair ADPS comparison
    st.markdown("#### Per-Pair Style Transfer Quality")
    stq_pairs = pm.get("per_pair_stq", {})
    if stq_pairs:
        fig = px.bar(
            x=list(stq_pairs.keys()),
            y=list(stq_pairs.values()),
            color=list(stq_pairs.values()),
            color_continuous_scale=[[0, ATTACK_RED], [0.5, "#F9A825"], [1, SPOTIFY_GREEN]],
            title="Style Transfer Quality (STQ) per Attack Pair",
            template="plotly_dark",
        )
        fig.update_layout(paper_bgcolor=CARD_BG, coloraxis_showscale=False,
                          xaxis_title="Attack Pair", yaxis_title="STQ")
        st.plotly_chart(fig, use_container_width=True)

    # Fairness detail
    fps_detail = pm.get("fps", {})
    if fps_detail:
        col_l, col_r = st.columns(2)
        with col_l:
            st.metric("Gini Coefficient", fps_detail.get("gini_coefficient", 0),
                      help="0 = perfect equality in protection across artists")
            st.metric("Worst Protected Artist", fps_detail.get("min_protection", 0))
        with col_r:
            st.metric("Best Protected Artist", fps_detail.get("max_protection", 0))
            st.metric("Std Protection Spread", fps_detail.get("std_protection", 0))


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: EVAL FRAMEWORK
# ─────────────────────────────────────────────────────────────────────────────

elif page == "Eval Framework":
    st.markdown("# Evaluation Framework")
    st.markdown(
        "Infrastructure a Spotify DS would use to design experiments, "
        "validate causal claims, audit fairness, and quantify ecosystem impact."
    )

    tab1, tab2, tab3, tab4 = st.tabs(
        ["A/B Tests", "Causal Inference", "Fairness Audit", "Ecosystem Impact"]
    )

    with tab1:
        st.markdown("#### A/B Test Power Analysis at Spotify Scale")
        st.markdown("Assumes ~25M daily eligible users for generative music features.")
        ab_rows = ef.get("ab_tests", [])
        if ab_rows:
            df_ab = pd.DataFrame(ab_rows)
            df_ab = df_ab.rename(columns={
                "metric": "Metric",
                "n_per_variant": "n per Variant",
                "runtime_days_at_spotify_scale": "Runtime (days)",
                "mde_relative": "MDE",
                "baseline": "Baseline",
                "alpha": "α",
                "power": "Power",
            })
            st.dataframe(df_ab[["Metric", "Baseline", "MDE", "n per Variant",
                                  "Runtime (days)", "α", "Power"]],
                         use_container_width=True, hide_index=True)
            st.info(
                "At Spotify scale, most experiments reach significance within **1 day** "
                "of exposure — but novelty effects and carryover bias still require "
                "a minimum **1–2 week holdout** for reliable estimates."
            )

    with tab2:
        st.markdown("#### Difference-in-Differences (Causal Inference)")
        st.markdown(
            "Simulated rollout analysis using a 2×2 DiD estimator with "
            "bootstrap confidence intervals — for observational settings where "
            "randomization isn't possible."
        )
        did = ef.get("did_example", {})
        if did:
            c1, c2, c3 = st.columns(3)
            c1.metric("ATT Estimate", f"{did.get('att_estimate', 0):+.4f}")
            c2.metric("95% CI",
                      f"[{did.get('ci_95_lo', 0):+.3f}, {did.get('ci_95_hi', 0):+.3f}]")
            c3.metric("p-value", did.get("p_value", 0),
                      delta="Significant ✓" if did.get("significant") else "Not sig")

            # Visualize treatment vs control trend
            rng = np.random.default_rng(42)
            periods = ["Pre", "Post"]
            fig = go.Figure()
            fig.add_trace(go.Bar(name="Control", x=periods,
                                 y=[round(did.get("att_estimate", 0) * 0 + 0.62, 3),
                                    round(0.62 + did.get("control_trend", 0), 3)],
                                 marker_color=CLEAN_BLUE))
            fig.add_trace(go.Bar(name="Treatment", x=periods,
                                 y=[0.62,
                                    round(0.62 + did.get("treatment_trend", 0), 3)],
                                 marker_color=SPOTIFY_GREEN))
            fig.update_layout(
                barmode="group",
                title="DiD: Pre/Post Metric by Group",
                template="plotly_dark",
                paper_bgcolor=CARD_BG,
                yaxis=dict(range=[0.6, 0.72]),
            )
            st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.markdown("#### Fairness Audit — Disparate Impact Analysis")
        st.markdown(
            "4/5ths rule: a group is disadvantaged if its mean protection score "
            "is < 80% of the best-performing group's mean."
        )
        audit = ef.get("fairness_audit", {})
        if audit:
            rows = [
                {"Group": g,
                 "Mean Protection": v["mean"],
                 "Impact Ratio": v["impact_ratio"],
                 "n": v["n"],
                 "Disparate Impact": "⚠️ YES" if v["disparate_impact"] else "✓ No"}
                for g, v in audit.items()
            ]
            df_audit = pd.DataFrame(rows).sort_values("Impact Ratio")
            st.dataframe(df_audit, use_container_width=True, hide_index=True)

            fig = px.bar(
                df_audit, x="Group", y="Impact Ratio",
                color="Impact Ratio",
                color_continuous_scale=[[0, ATTACK_RED], [0.8, "#F9A825"], [1, SPOTIFY_GREEN]],
                title="Fairness Impact Ratio by Group (4/5ths rule: flag if < 0.80)",
                template="plotly_dark",
            )
            fig.add_hline(y=0.80, line_dash="dash", line_color="white",
                          annotation_text="4/5ths threshold")
            fig.update_layout(paper_bgcolor=CARD_BG, coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)

    with tab4:
        st.markdown("#### Ecosystem Impact Model")
        st.markdown(
            "Estimated financial impact of adversarial identity spoofing on "
            "artist royalty flows at Spotify scale."
        )
        eco = ef.get("ecosystem_impact", {})
        if eco:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Daily generations", f"{eco.get('daily_generations', 0):,}")
            c2.metric("Est. daily attacks", f"{eco.get('estimated_daily_attacks', 0):,}")
            c3.metric("Royalty protected/day",
                      f"${eco.get('royalty_protected_daily_usd', 0):.2f}")
            c4.metric("Annual protected value",
                      f"${eco.get('annual_protected_usd', 0):,.0f}")

            # Sankey: generations → attacks → outcomes
            fig = go.Figure(go.Sankey(
                node=dict(
                    label=["Daily Generations", "Attack Attempts", "Detected",
                           "Missed", "No Attack"],
                    color=[SPOTIFY_GREEN, ATTACK_RED, CLEAN_BLUE, "#E53935", SPOTIFY_GREEN],
                ),
                link=dict(
                    source=[0, 1, 1, 0],
                    target=[1, 2, 3, 4],
                    value=[
                        eco.get("estimated_daily_attacks", 0),
                        eco.get("detected_attacks", 0),
                        eco.get("missed_attacks", 0),
                        eco.get("daily_generations", 0) - eco.get("estimated_daily_attacks", 0),
                    ],
                    color=[ATTACK_RED, CLEAN_BLUE, ATTACK_RED, SPOTIFY_GREEN],
                ),
            ))
            fig.update_layout(
                title="Daily Traffic Flow: Generations → Attacks → Outcomes",
                template="plotly_dark",
                paper_bgcolor=CARD_BG,
                font_color="white",
            )
            st.plotly_chart(fig, use_container_width=True)


# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    '<div style="text-align:center; color:#B3B3B3; font-size:0.8rem;">'
    "Built for Spotify's Artist-First AI Music Lab · "
    "SD-MIAE methodology · Last.fm data"
    "</div>",
    unsafe_allow_html=True,
)
