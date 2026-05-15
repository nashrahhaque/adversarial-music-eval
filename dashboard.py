"""
Streamlit dashboard for adversarial-music-eval.

Presents the full SD-MIAE pipeline results in an interactive UI.

Deploy to Streamlit Community Cloud:
    1. Push repo to GitHub
    2. Go to share.streamlit.io -> New app -> select this file
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

st.set_page_config(
    page_title="Adversarial Music Eval",
    page_icon="",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Design tokens ─────────────────────────────────────────────────────────────
T = {
    "bg":       "#F7F6F3",
    "ink":      "#141414",
    "sub":      "#5A5A5A",
    "rule":     "#D4D0C8",
    "red":      "#A81C1C",
    "cobalt":   "#1B3D6B",
    "amber":    "#7A5C00",
    "violet":   "#4B2C6E",
    "green":    "#1A5C2A",
    "font":     "'Helvetica Neue', Helvetica, Arial, sans-serif",
    "mono":     "'JetBrains Mono', 'Fira Mono', 'Courier New', monospace",
}

ATTACK_COLOR = T["red"]
CLEAN_COLOR  = T["cobalt"]
PALETTE = [T["cobalt"], T["red"], T["amber"], T["violet"], T["green"],
           "#8B4513", "#006363", "#5C3A1E", "#2C4A6E", "#6B1A1A"]

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

# ── Global styles ─────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
  @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&display=swap');

  html, body, [class*="css"], [data-testid="stAppViewContainer"],
  [data-testid="stMain"], [data-testid="block-container"] {{
      font-family: {T["font"]} !important;
      background-color: {T["bg"]} !important;
      color: {T["ink"]} !important;
  }}
  .stApp, .stApp > div, [data-testid="stAppViewContainer"] > div {{
      background-color: {T["bg"]} !important;
  }}
  .block-container {{ padding-top: 2rem !important; padding-bottom: 3rem; max-width: 1280px; }}

  /* force all text dark */
  p, li, span, label, div, h1, h2, h3, h4, h5, h6,
  .stMarkdown, .stText, .stCaption,
  [data-testid="stMarkdownContainer"] * {{
      color: {T["ink"]} !important;
  }}
  /* override Streamlit metric/widget backgrounds */
  [data-testid="stMetric"],
  [data-testid="metric-container"],
  [data-testid="stVerticalBlock"],
  [data-testid="stHorizontalBlock"] {{
      background-color: {T["bg"]} !important;
  }}
  /* tabs */
  .stTabs [data-baseweb="tab-list"] {{
      background-color: {T["bg"]} !important;
      border-bottom: 2px solid {T["rule"]} !important;
      gap: 0 !important;
  }}
  .stTabs [data-baseweb="tab"] {{
      background-color: {T["bg"]} !important;
      color: {T["sub"]} !important;
      font-family: {T["mono"]} !important;
      font-size: 0.75rem !important;
      letter-spacing: 0.08em !important;
      border: none !important;
      padding: 0.6rem 1.2rem !important;
  }}
  .stTabs [aria-selected="true"] {{
      color: {T["ink"]} !important;
      border-bottom: 2px solid {T["ink"]} !important;
  }}
  /* selectbox, multiselect */
  [data-testid="stSelectbox"] *, [data-testid="stMultiSelect"] * {{
      color: {T["ink"]} !important;
      background-color: {T["bg"]} !important;
  }}
  /* dataframe */
  .stDataFrame, [data-testid="stDataFrame"] {{
      background-color: {T["bg"]} !important;
  }}
  [data-testid="stDataFrame"] td, [data-testid="stDataFrame"] th {{
      color: {T["ink"]} !important;
      background-color: {T["bg"]} !important;
  }}
  /* info/warning boxes */
  [data-testid="stAlert"] {{
      background-color: #EEECEB !important;
      border: 1px solid {T["rule"]} !important;
      color: {T["ink"]} !important;
  }}
  [data-testid="stAlert"] * {{ color: {T["ink"]} !important; }}

  /* masthead */
  .masthead {{
      border-bottom: 2px solid {T["ink"]};
      padding-bottom: 1.5rem;
      margin-bottom: 2.5rem;
  }}
  .masthead-title {{
      font-size: clamp(2.4rem, 4vw, 3.8rem) !important;
      font-weight: 900 !important;
      letter-spacing: -0.03em;
      line-height: 1.05;
      color: {T["ink"]} !important;
      margin: 0;
  }}
  .masthead-sub {{
      font-size: 0.82rem;
      letter-spacing: 0.14em;
      text-transform: uppercase;
      color: {T["sub"]} !important;
      margin-top: 0.5rem;
  }}

  /* section header */
  .sec-hd {{
      display: flex;
      align-items: baseline;
      gap: 1rem;
      border-top: 1px solid {T["rule"]};
      padding-top: 0.6rem;
      margin: 2rem 0 1rem 0;
  }}
  .sec-num {{
      font-family: {T["mono"]};
      font-size: 0.72rem;
      color: {T["sub"]} !important;
      letter-spacing: 0.1em;
  }}
  .sec-title {{
      font-size: 1.05rem;
      font-weight: 700;
      letter-spacing: -0.01em;
      color: {T["ink"]} !important;
  }}

  /* stat bar */
  .stat-bar {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
      gap: 0;
      border: 1px solid {T["rule"]};
      margin-bottom: 2rem;
  }}
  .stat-cell {{
      padding: 1.2rem 1.4rem;
      border-right: 1px solid {T["rule"]};
      background-color: {T["bg"]};
  }}
  .stat-cell:last-child {{ border-right: none; }}
  .stat-val {{
      font-family: {T["mono"]};
      font-size: 2rem;
      font-weight: 700;
      color: {T["ink"]} !important;
      line-height: 1;
  }}
  .stat-lbl {{
      font-size: 0.72rem;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      color: {T["sub"]} !important;
      margin-top: 0.4rem;
  }}
  .stat-note {{
      font-family: {T["mono"]};
      font-size: 0.72rem;
      color: {T["sub"]} !important;
      margin-top: 0.2rem;
  }}

  /* tag pill */
  .pill {{
      display: inline-block;
      font-family: {T["mono"]};
      font-size: 0.7rem;
      letter-spacing: 0.05em;
      padding: 0.15rem 0.5rem;
      border: 1px solid {T["rule"]};
      color: {T["sub"]} !important;
      margin: 0.15rem;
  }}

  /* attack card */
  .atk-card {{
      border-left: 3px solid {T["red"]};
      padding: 0.75rem 1rem;
      margin-bottom: 0.5rem;
      background-color: {T["bg"]};
      border-top: 1px solid {T["rule"]};
      border-right: 1px solid {T["rule"]};
      border-bottom: 1px solid {T["rule"]};
  }}
  .atk-card b {{ color: {T["ink"]} !important; }}

  /* alert */
  .alert-flag {{
      display: inline-block;
      font-family: {T["mono"]};
      font-size: 0.7rem;
      letter-spacing: 0.06em;
      text-transform: uppercase;
      padding: 0.15rem 0.5rem;
      background: {T["red"]};
      color: white;
  }}
  .ok-flag {{
      display: inline-block;
      font-family: {T["mono"]};
      font-size: 0.7rem;
      letter-spacing: 0.06em;
      text-transform: uppercase;
      padding: 0.15rem 0.5rem;
      border: 1px solid {T["rule"]};
      color: {T["sub"]};
  }}

  /* dataframe override */
  .stDataFrame {{ border: 1px solid {T["rule"]}; }}

  /* hide streamlit chrome */
  #MainMenu, footer, header {{ visibility: hidden; }}
  .stDeployButton {{ display: none; }}
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


fp_data, results = load_data()

if fp_data is None:
    st.error("No data found. Run `python run_pipeline.py` to generate data/, then restart.")
    st.stop()

fingerprints = fp_data["fingerprints"]
vocab = fp_data["vocabulary"]
attacks = results.get("attacks", {})
detection = results.get("detection", {})
pm = results.get("product_metrics", {})
ef = results.get("eval_framework", {})

artist_vecs = {a: np.array(fingerprints[a]["vector"]) for a in ARTISTS if a in fingerprints}


# ── Chart theme helper ────────────────────────────────────────────────────────

def theme(fig, title="", height=380):
    fig.update_layout(
        title=dict(text=title, font=dict(size=13, family=T["font"], color=T["ink"]),
                   x=0, xanchor="left"),
        paper_bgcolor=T["bg"],
        plot_bgcolor=T["bg"],
        font=dict(family=T["font"], color=T["ink"], size=12),
        xaxis=dict(gridcolor=T["rule"], linecolor=T["rule"], zeroline=False),
        yaxis=dict(gridcolor=T["rule"], linecolor=T["rule"], zeroline=False),
        legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor=T["rule"], borderwidth=1),
        margin=dict(l=40, r=20, t=50, b=40),
        height=height,
    )
    return fig


def sec(num, title):
    st.markdown(
        f'<div class="sec-hd">'
        f'<span class="sec-num">{num}</span>'
        f'<span class="sec-title">{title}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )


def stat_bar(cells):
    html = '<div class="stat-bar">'
    for val, lbl, note in cells:
        html += (
            f'<div class="stat-cell">'
            f'<div class="stat-val">{val}</div>'
            f'<div class="stat-lbl">{lbl}</div>'
            f'<div class="stat-note">{note}</div>'
            f'</div>'
        )
    html += '</div>'
    st.markdown(html, unsafe_allow_html=True)


# ── Masthead ──────────────────────────────────────────────────────────────────

st.markdown(
    '<div class="masthead">'
    '<h1 class="masthead-title">Adversarial Music Eval</h1>'
    '<p class="masthead-sub">SD-MIAE applied to generative music identity protection'
    ' &nbsp;/&nbsp; Last.fm data &nbsp;/&nbsp; IEEE methodology</p>'
    '</div>',
    unsafe_allow_html=True,
)

# ── Tab navigation ────────────────────────────────────────────────────────────

tab_overview, tab_fp, tab_attack, tab_detect, tab_metrics, tab_eval = st.tabs([
    "01  Overview",
    "02  Artist Fingerprints",
    "03  Attack Analysis",
    "04  Detection",
    "05  Product Metrics",
    "06  Eval Framework",
])


# ─────────────────────────────────────────────────────────────────────────────
# TAB 01: OVERVIEW
# ─────────────────────────────────────────────────────────────────────────────

with tab_overview:
    det_metrics = detection.get("metrics", {})
    ahs = pm.get("afhs", 0)
    ari = pm.get("ari", {}).get("ari", 0)
    fps = pm.get("fps", {}).get("fps", 0)
    f1  = det_metrics.get("f1", 0)
    stq = pm.get("mean_stq", 0)

    stat_bar([
        (f"{ahs:.3f}",  "AHS",          "artist health score"),
        (f"{f1:.3f}",   "Detection F1", "adversarial recall"),
        (f"{ari:.3f}",  "ARI",          "robustness index"),
        (f"{fps:.3f}",  "FPS",          "fairness parity"),
        (f"{stq:.3f}",  "STQ",          "style quality"),
    ])

    col_l, col_r = st.columns([3, 2])

    with col_l:
        sec("01.1", "Pipeline")
        st.markdown("""
| Step | Module | Output |
|------|--------|--------|
| 1 | `fetch_artists.py` | Last.fm tags, tracks, listeners for 10 artists |
| 2 | `fingerprint.py` | 54-dim artist DNA vector |
| 3 | `attack.py` | SD-MIAE adversarial attack on 5 pairs |
| 4 | `detect.py` | Z-score anomaly detection (threshold = 2.0) |
| 5 | `metrics.py` | ADPS, STQ, ARI, FPS, AHS |
| 6 | `eval_framework.py` | A/B tests, causal inference, fairness, ecosystem impact |
        """)

        sec("01.2", "Artist Dataset")
        try:
            with open(pathlib.Path(__file__).parent / "data" / "artists.json") as f:
                artists_raw = json.load(f)
            df_artists = pd.DataFrame([
                {
                    "Artist": a,
                    "Listeners (M)": round(artists_raw[a].get("listeners", 0) / 1e6, 2),
                    "Plays (B)": round(artists_raw[a].get("playcount", 0) / 1e9, 2),
                    "Top Tags": ", ".join(t["name"] for t in artists_raw[a].get("tags", [])[:3]),
                }
                for a in ARTISTS if a in artists_raw and "error" not in artists_raw[a]
            ])
            fig = go.Figure(go.Bar(
                x=df_artists["Artist"],
                y=df_artists["Listeners (M)"],
                marker_color=T["cobalt"],
                marker_line_width=0,
            ))
            theme(fig, "Monthly Listeners (Last.fm, M)")
            fig.update_xaxes(tickangle=-30)
            st.plotly_chart(fig, use_container_width=True)
        except Exception:
            st.info("Run the pipeline to load artist data.")

    with col_r:
        sec("01.3", "Attack Pairs")
        for pair_key, pd_ in attacks.items():
            sim   = pd_["metrics"]["attack_similarity"]
            delta = pd_["metrics"]["absolute_improvement"]
            det   = pd_.get("detection", {})
            flag  = '<span class="alert-flag">detected</span>' if det.get("is_adversarial") \
                    else '<span class="ok-flag">missed</span>'
            st.markdown(
                f'<div class="atk-card">'
                f'<b>{pd_["source"].split()[0]} &rarr; {pd_["target"].split()[0]}</b><br>'
                f'<span style="font-family:{T["mono"]};font-size:0.8rem;color:{T["sub"]}">'
                f'sim={sim:.3f} ({delta:+.3f})</span>&nbsp;&nbsp;{flag}'
                f'</div>',
                unsafe_allow_html=True,
            )


# ─────────────────────────────────────────────────────────────────────────────
# TAB 02: ARTIST FINGERPRINTS
# ─────────────────────────────────────────────────────────────────────────────

with tab_fp:
    st.markdown(
        "Each artist encoded as a **54-dimensional vector** from Last.fm data. "
        "The fingerprint acts as the conditioning vector a generative model uses "
        "to steer musical style."
    )

    sub1, sub2, sub3 = st.tabs(["Radar Charts", "PCA Space", "Similarity Heatmap"])

    with sub1:
        sec("02.1", "Artist DNA Radar")
        selected = st.multiselect("Select artists", ARTISTS, default=ARTISTS[:4])
        if selected:
            regions = list(DIM_LAYOUT.keys())
            fig = go.Figure()
            for i, artist in enumerate(selected):
                if artist not in artist_vecs:
                    continue
                vec  = artist_vecs[artist]
                vals = [float(vec[s:e].mean()) for s, e in DIM_LAYOUT.values()]
                vals += vals[:1]
                angles = regions + [regions[0]]
                fig.add_trace(go.Scatterpolar(
                    r=vals, theta=angles, fill="toself", name=artist,
                    opacity=0.65,
                    line=dict(color=PALETTE[i % len(PALETTE)], width=2),
                ))
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(visible=True, range=[0, 1],
                                    gridcolor=T["rule"], linecolor=T["rule"]),
                    angularaxis=dict(gridcolor=T["rule"], linecolor=T["rule"]),
                    bgcolor=T["bg"],
                ),
                paper_bgcolor=T["bg"],
                font=dict(family=T["font"], color=T["ink"]),
                legend=dict(bgcolor="rgba(0,0,0,0)"),
                title=dict(text="Per-region mean (normalized)", font=dict(size=12), x=0),
                height=420,
                margin=dict(l=20, r=20, t=50, b=20),
            )
            st.plotly_chart(fig, use_container_width=True)

        sec("02.2", "54-Dimension Layout")
        dim_df = pd.DataFrame([
            {"Dims": f"{s}-{e-1}", "Region": region, "Description": desc}
            for (region, (s, e)), desc in zip(
                DIM_LAYOUT.items(),
                [
                    "Multi-hot tag weights over top-30 global tags",
                    "Log-normalized monthly listener count",
                    "Log-normalized total play count",
                    "Plays-per-listener ratio, normalized",
                    "Unique tag breadth (0-1)",
                    "Similarity scores to each of the 10 canonical artists",
                    "Track play/listener stats, log-normalized",
                    "Tag weight distribution statistics",
                ],
            )
        ])
        st.dataframe(dim_df, use_container_width=True, hide_index=True)

    with sub2:
        sec("02.3", "Fingerprint PCA Space")
        names  = list(artist_vecs.keys())
        vecs   = np.array([artist_vecs[a] for a in names])
        pca    = PCA(n_components=2, random_state=42)
        coords = pca.fit_transform(vecs)
        var_exp = pca.explained_variance_ratio_

        fig = go.Figure()
        for i, (name, (x, y)) in enumerate(zip(names, coords)):
            fig.add_trace(go.Scatter(
                x=[x], y=[y], mode="markers+text",
                marker=dict(size=14, color=PALETTE[i % len(PALETTE)],
                            line=dict(width=1, color=T["ink"])),
                text=[name.split()[0]], textposition="top center",
                textfont=dict(size=11, color=T["ink"]),
                name=name, showlegend=False,
            ))

        for _, pd_ in attacks.items():
            src, tgt = pd_["source"], pd_["target"]
            if src in names and tgt in names:
                si, ti = names.index(src), names.index(tgt)
                fig.add_annotation(
                    x=coords[ti][0], y=coords[ti][1],
                    ax=coords[si][0], ay=coords[si][1],
                    xref="x", yref="y", axref="x", ayref="y",
                    arrowhead=3, arrowcolor=T["red"], arrowwidth=1.5, arrowsize=1.1,
                )

        theme(fig, f"PC1 ({var_exp[0]:.1%} var) x PC2 ({var_exp[1]:.1%} var) -- red arrows = attack direction")
        st.plotly_chart(fig, use_container_width=True)

    with sub3:
        sec("02.4", "Pairwise Cosine Similarity")
        names   = list(artist_vecs.keys())
        vecs    = np.array([artist_vecs[a] for a in names])
        sim_mat = np.array([
            [1.0 - cosine_dist(vecs[i], vecs[j]) for j in range(len(names))]
            for i in range(len(names))
        ])
        short_names = [n.split()[0] for n in names]
        fig = px.imshow(
            sim_mat, x=short_names, y=short_names,
            color_continuous_scale=[[0, "#F7F6F3"], [0.5, T["cobalt"]], [1, T["ink"]]],
            zmin=0, zmax=1, text_auto=".2f",
        )
        fig.update_layout(
            paper_bgcolor=T["bg"], plot_bgcolor=T["bg"],
            font=dict(family=T["font"], color=T["ink"]),
            coloraxis_colorbar=dict(tickfont=dict(color=T["ink"])),
            height=420, margin=dict(l=10, r=10, t=20, b=10),
        )
        st.plotly_chart(fig, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 03: ATTACK ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

with tab_attack:
    st.markdown(
        "Momentum-integrated sign-gradient attack adapted from the IEEE SD-MIAE paper. "
        "Finds delta such that `model(source + delta) approx model(target)` "
        "with `||delta||_inf <= 0.05`."
    )

    cfg = results.get("config", {})
    stat_bar([
        (str(cfg.get("epsilon", "-")), "epsilon (budget)",   "L-inf constraint"),
        (str(cfg.get("mu", "-")),      "mu (momentum)",      "gradient decay"),
        (str(cfg.get("alpha", "-")),   "alpha (step size)",  "per-iteration"),
        (str(cfg.get("steps", "-")),   "T (steps)",          "iterations"),
    ])

    sec("03.1", "Convergence Trajectories")
    fig = go.Figure()
    for i, (pair_key, pd_) in enumerate(attacks.items()):
        history  = pd_.get("history", [])
        if not history:
            continue
        steps    = [h["step"] for h in history]
        sims     = [h["cosine_similarity"] for h in history]
        baseline = history[0]["src_tgt_baseline"]
        label    = f"{pd_['source'].split()[0]} -> {pd_['target'].split()[0]}"
        fig.add_trace(go.Scatter(
            x=steps, y=sims, mode="lines+markers", name=label,
            line=dict(color=PALETTE[i % len(PALETTE)], width=2),
            marker=dict(size=4),
        ))
        fig.add_hline(
            y=baseline, line_dash="dot",
            line_color=PALETTE[i % len(PALETTE)], opacity=0.35,
            annotation_text=f"baseline {label[:5]}",
            annotation_font_color=T["sub"], annotation_font_size=10,
        )
    fig.update_yaxes(range=[0, 1.05])
    theme(fig, "Cosine similarity (adversarial -> target) over attack iterations")
    fig.update_xaxes(title_text="Attack iteration")
    fig.update_yaxes(title_text="Cosine similarity")
    st.plotly_chart(fig, use_container_width=True)

    sec("03.2", "Per-Pair Results")
    rows = []
    for pair_key, pd_ in attacks.items():
        m   = pd_["metrics"]
        det = pd_.get("detection", {})
        rows.append({
            "Pair":          pair_key,
            "Baseline sim":  round(m["baseline_similarity"], 4),
            "Attack sim":    round(m["attack_similarity"], 4),
            "Improvement":   f"{m['absolute_improvement']:+.4f}",
            "||delta||_inf": round(m["delta_linf"], 4),
            "||delta||_2":   round(m["delta_l2"], 4),
            "Constraint":    "pass" if m["constraint_satisfied"] else "FAIL",
            "Detected":      "YES" if det.get("is_adversarial") else "no",
            "Confidence":    round(det.get("confidence", 0), 3),
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    sec("03.3", "Perturbation Distribution")
    pair_sel = st.selectbox("Attack pair", list(attacks.keys()), key="atk_sel")
    if pair_sel in attacks:
        delta_vec = np.array(attacks[pair_sel]["delta"])
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=list(range(len(delta_vec))),
            y=delta_vec,
            marker_color=[T["red"] if d > 0 else T["cobalt"] for d in delta_vec],
            marker_line_width=0,
            name="delta",
        ))
        fig.add_hline(y= cfg.get("epsilon", 0.05), line_dash="dash",
                      line_color=T["sub"], annotation_text="+eps",
                      annotation_font_color=T["sub"])
        fig.add_hline(y=-cfg.get("epsilon", 0.05), line_dash="dash",
                      line_color=T["sub"], annotation_text="-eps",
                      annotation_font_color=T["sub"])
        for region, (s, e) in DIM_LAYOUT.items():
            fig.add_vrect(x0=s, x1=e, fillcolor=T["rule"], opacity=0.3,
                          annotation_text=region[:4], annotation_position="top left",
                          annotation_font_size=8, annotation_font_color=T["sub"])
        theme(fig, f"Perturbation vector: {pair_sel}")
        fig.update_xaxes(title_text="Fingerprint dimension")
        fig.update_yaxes(title_text="Perturbation delta")
        st.plotly_chart(fig, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 04: DETECTION
# ─────────────────────────────────────────────────────────────────────────────

with tab_detect:
    det_metrics = detection.get("metrics", {})
    threshold   = detection.get("threshold", 2.0)

    st.markdown(
        f"Z-score deviation from clean embedding distribution. "
        f"**Threshold = {threshold}.** "
        "Embedding flagged as adversarial if `mean |z| > threshold`."
    )

    stat_bar([
        (str(det_metrics.get("precision", "-")), "Precision", ""),
        (str(det_metrics.get("recall", "-")),    "Recall",    ""),
        (str(det_metrics.get("f1", "-")),        "F1",        ""),
        (str(det_metrics.get("auroc", "-")),     "AUROC",     ""),
    ])

    col_l, col_r = st.columns(2)

    clean_data = detection.get("clean_embeddings", {})

    with col_l:
        sec("04.1", "Clean Embeddings")
        clean_rows = [
            {"Artist": a, "Mean |z|": round(v["mean_z_score"], 3),
             "Max |z|": round(v["max_z_score"], 3),
             "Status": "clean" if not v["is_adversarial"] else "FLAGGED"}
            for a, v in clean_data.items()
        ]
        st.dataframe(pd.DataFrame(clean_rows), use_container_width=True, hide_index=True)

    with col_r:
        sec("04.2", "Adversarial Embeddings")
        adv_rows = [
            {"Pair": k,
             "Mean |z|": round(v["detection"]["mean_z_score"], 3),
             "Confidence": round(v["detection"].get("confidence", 0), 3),
             "Status": "DETECTED" if v["detection"]["is_adversarial"] else "missed"}
            for k, v in attacks.items() if "detection" in v
        ]
        st.dataframe(pd.DataFrame(adv_rows), use_container_width=True, hide_index=True)

    sec("04.3", "Z-Score Distribution")
    clean_zs = [v["mean_z_score"] for v in clean_data.values()]
    adv_zs   = [v["detection"]["mean_z_score"] for v in attacks.values() if "detection" in v]

    fig = go.Figure()
    fig.add_trace(go.Histogram(x=clean_zs, name="Clean",       nbinsx=10,
                               marker_color=T["cobalt"], opacity=0.75))
    fig.add_trace(go.Histogram(x=adv_zs,   name="Adversarial", nbinsx=10,
                               marker_color=T["red"],    opacity=0.75))
    fig.add_vline(x=threshold, line_dash="dash", line_color=T["ink"],
                  annotation_text=f"Threshold = {threshold}",
                  annotation_font_color=T["ink"])
    fig.update_layout(barmode="overlay")
    theme(fig, "Clean vs adversarial mean |z-score|")
    fig.update_xaxes(title_text="Mean |z-score|")
    fig.update_yaxes(title_text="Count")
    st.plotly_chart(fig, use_container_width=True)

    attribution = detection.get("attribution", {})
    if attribution:
        sec("04.4", "Region Attribution")
        st.caption("Which fingerprint regions are most exploited by the attack.")
        vals = list(attribution.values())
        colors = [T["cobalt"] if v < 1.5 else T["amber"] if v < 2.5 else T["red"]
                  for v in vals]
        fig = go.Figure(go.Bar(
            x=list(attribution.keys()), y=vals,
            marker_color=colors, marker_line_width=0,
        ))
        theme(fig, "Mean |z-score| per fingerprint region across all attacks")
        fig.update_xaxes(title_text="Region")
        fig.update_yaxes(title_text="Mean |z|")
        st.plotly_chart(fig, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 05: PRODUCT METRICS
# ─────────────────────────────────────────────────────────────────────────────

with tab_metrics:
    ahs  = pm.get("afhs", 0)
    ari  = pm.get("ari", {}).get("ari", 0)
    fps  = pm.get("fps", {}).get("fps", 0)
    stq  = pm.get("mean_stq", 0)
    f1   = detection.get("metrics", {}).get("f1", 0)

    stat_bar([
        (f"{ahs:.3f}",  "AHS",          "artist health score"),
        (f"{ari:.3f}",  "ARI",          "robustness index"),
        (f"{fps:.3f}",  "FPS",          "fairness parity"),
        (f"{stq:.3f}",  "STQ",          "style quality"),
        (f"{f1:.3f}",   "Detection F1", "adversarial recall"),
    ])

    sec("05.1", "Metric Definitions")
    st.markdown("""
| Metric | Abbr | Formula | Measures |
|--------|------|---------|---------|
| Artist DNA Preservation Score | ADPS | cos\_sim(original, generated) | Identity fidelity |
| Style Transfer Quality | STQ | H-mean(fidelity, imperceptibility) | Attack quality |
| Adversarial Robustness Index | ARI | H-mean(detection\_rate, confidence) | System safety |
| Fairness Parity Score | FPS | 1 - Gini(per-artist protection) | Equitable protection |
| Artist Health Score | AHS | 0.4 x ARI + 0.3 x FPS + 0.3 x STQ | Composite dashboard KPI |
    """)

    sec("05.2", "Style Transfer Quality per Pair")
    stq_pairs = pm.get("per_pair_stq", {})
    if stq_pairs:
        vals   = list(stq_pairs.values())
        colors = [T["red"] if v < 0.5 else T["amber"] if v < 0.75 else T["cobalt"]
                  for v in vals]
        fig = go.Figure(go.Bar(
            x=list(stq_pairs.keys()), y=vals,
            marker_color=colors, marker_line_width=0,
        ))
        theme(fig, "STQ per attack pair")
        fig.update_xaxes(title_text="Attack pair", tickangle=-20)
        fig.update_yaxes(title_text="STQ", range=[0, 1])
        st.plotly_chart(fig, use_container_width=True)

    fps_detail = pm.get("fps", {})
    if fps_detail:
        sec("05.3", "Fairness Detail")
        col_l, col_r = st.columns(2)
        with col_l:
            stat_bar([
                (f"{fps_detail.get('gini_coefficient', 0):.3f}", "Gini coefficient",
                 "0 = perfect equality"),
                (f"{fps_detail.get('min_protection', 0):.3f}",   "Min protection", "worst artist"),
            ])
        with col_r:
            stat_bar([
                (f"{fps_detail.get('max_protection', 0):.3f}",   "Max protection",  "best artist"),
                (f"{fps_detail.get('std_protection', 0):.3f}",   "Std protection",  "spread"),
            ])


# ─────────────────────────────────────────────────────────────────────────────
# TAB 06: EVAL FRAMEWORK
# ─────────────────────────────────────────────────────────────────────────────

with tab_eval:
    st.markdown(
        "Experimental infrastructure for designing A/B tests, validating causal claims, "
        "auditing fairness, and quantifying ecosystem impact."
    )

    ef_tab1, ef_tab2, ef_tab3, ef_tab4 = st.tabs(
        ["A/B Tests", "Causal Inference", "Fairness Audit", "Ecosystem Impact"]
    )

    with ef_tab1:
        sec("06.1", "A/B Test Power Analysis")
        st.caption("Assumes ~25M daily eligible users for generative music features.")
        ab_rows = ef.get("ab_tests", [])
        if ab_rows:
            df_ab = pd.DataFrame(ab_rows).rename(columns={
                "metric":                "Metric",
                "n_per_variant":         "n / Variant",
                "runtime_days_at_scale": "Runtime (days)",
                "mde_relative":          "MDE",
                "baseline":              "Baseline",
                "alpha":                 "alpha",
                "power":                 "Power",
            })
            cols = [c for c in ["Metric", "Baseline", "MDE", "n / Variant",
                                 "Runtime (days)", "alpha", "Power"]
                    if c in df_ab.columns]
            st.dataframe(df_ab[cols], use_container_width=True, hide_index=True)
            st.info(
                "At this scale, most experiments reach significance within 1 day of exposure "
                "-- but novelty effects and carryover bias still require a minimum "
                "1-2 week holdout for reliable estimates."
            )

    with ef_tab2:
        sec("06.2", "Difference-in-Differences")
        st.markdown(
            "Simulated rollout analysis using a 2x2 DiD estimator with "
            "bootstrap confidence intervals -- for observational settings "
            "where randomization is not possible."
        )
        did = ef.get("did_example", {})
        if did:
            stat_bar([
                (f"{did.get('att_estimate', 0):+.4f}", "ATT Estimate", "avg treatment effect"),
                (f"[{did.get('ci_95_lo', 0):+.3f}, {did.get('ci_95_hi', 0):+.3f}]",
                 "95% CI", "bootstrap"),
                (str(did.get("p_value", "-")), "p-value",
                 "significant" if did.get("significant") else "not significant"),
            ])

            periods = ["Pre", "Post"]
            fig = go.Figure()
            fig.add_trace(go.Bar(
                name="Control", x=periods,
                y=[round(0.62, 3), round(0.62 + did.get("control_trend", 0), 3)],
                marker_color=T["cobalt"], marker_line_width=0,
            ))
            fig.add_trace(go.Bar(
                name="Treatment", x=periods,
                y=[0.62, round(0.62 + did.get("treatment_trend", 0), 3)],
                marker_color=T["amber"], marker_line_width=0,
            ))
            fig.update_layout(barmode="group")
            theme(fig, "DiD: Pre/Post metric by group")
            fig.update_yaxes(range=[0.6, 0.72])
            st.plotly_chart(fig, use_container_width=True)

    with ef_tab3:
        sec("06.3", "Fairness Audit -- Disparate Impact Analysis")
        st.markdown(
            "4/5ths rule: a group is disadvantaged if its mean protection score "
            "is < 80% of the best-performing group."
        )
        audit = ef.get("fairness_audit", {})
        if audit:
            audit_rows = [
                {
                    "Group":            g,
                    "Mean Protection":  v["mean"],
                    "Impact Ratio":     v["impact_ratio"],
                    "n":                v["n"],
                    "Disparate Impact": "YES" if v["disparate_impact"] else "no",
                }
                for g, v in audit.items()
            ]
            df_audit = pd.DataFrame(audit_rows).sort_values("Impact Ratio")
            st.dataframe(df_audit, use_container_width=True, hide_index=True)

            vals   = df_audit["Impact Ratio"].tolist()
            colors = [T["red"] if v < 0.8 else T["amber"] if v < 0.9 else T["cobalt"]
                      for v in vals]
            fig = go.Figure(go.Bar(
                x=df_audit["Group"].tolist(), y=vals,
                marker_color=colors, marker_line_width=0,
            ))
            fig.add_hline(y=0.80, line_dash="dash", line_color=T["sub"],
                          annotation_text="4/5ths threshold (0.80)",
                          annotation_font_color=T["sub"])
            theme(fig, "Fairness impact ratio by group")
            fig.update_xaxes(title_text="Group")
            fig.update_yaxes(title_text="Impact ratio")
            st.plotly_chart(fig, use_container_width=True)

    with ef_tab4:
        sec("06.4", "Ecosystem Impact Model")
        st.markdown(
            "Estimated financial impact of adversarial identity spoofing on "
            "artist royalty flows at scale."
        )
        eco = ef.get("ecosystem_impact", {})
        if eco:
            stat_bar([
                (f"{eco.get('daily_generations', 0):,}",
                 "Daily generations", ""),
                (f"{eco.get('estimated_daily_attacks', 0):,}",
                 "Est. daily attacks", "1% attack rate"),
                (f"${eco.get('royalty_protected_daily_usd', 0):.2f}",
                 "Royalty protected/day", "USD"),
                (f"${eco.get('annual_protected_usd', 0):,.0f}",
                 "Annual protected value", "USD"),
            ])

            fig = go.Figure(go.Sankey(
                node=dict(
                    label=["Daily Generations", "Attack Attempts",
                           "Detected", "Missed", "No Attack"],
                    color=[T["cobalt"], T["red"], T["cobalt"], T["red"], T["cobalt"]],
                    pad=20, thickness=20,
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
                    color=[
                        "rgba(168,28,28,0.3)", "rgba(27,61,107,0.3)",
                        "rgba(168,28,28,0.3)", "rgba(27,61,107,0.3)",
                    ],
                ),
            ))
            fig.update_layout(
                paper_bgcolor=T["bg"],
                font=dict(family=T["font"], color=T["ink"], size=12),
                height=380,
                margin=dict(l=20, r=20, t=30, b=20),
            )
            st.plotly_chart(fig, use_container_width=True)


# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown(
    f'<div style="border-top:1px solid {T["rule"]};margin-top:3rem;padding-top:1rem;'
    f'font-family:{T["mono"]};font-size:0.72rem;color:{T["sub"]};letter-spacing:0.06em;">'
    "SD-MIAE methodology &nbsp;/&nbsp; Last.fm data &nbsp;/&nbsp; "
    "adversarial-music-eval"
    "</div>",
    unsafe_allow_html=True,
)
