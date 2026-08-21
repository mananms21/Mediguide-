"""
MEDIGUIDE — Evaluation Dashboard (Streamlit Page)
Reads evaluate/results/results.json and renders an interactive comparison.
Run via: streamlit run app/app.py  (then click "Evaluation" in the sidebar)
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).parent.parent.parent
RESULTS_PATH = ROOT / "evaluate" / "results" / "results.json"

st.set_page_config(
    page_title="MEDIGUIDE — Evaluation",
    page_icon="📊",
    layout="wide",
)

# ── CSS ───────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
  html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
  .stApp { background: linear-gradient(135deg, #070d1a 0%, #0b1629 50%, #060e1c 100%); color: #e2e8f0; }

  .dash-title {
    font-size: 2rem; font-weight: 700;
    background: linear-gradient(135deg, #00d4ff 0%, #7c4dff 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
    margin-bottom: 0.3rem;
  }
  .dash-sub { font-size: 0.9rem; color: #475569; margin-bottom: 2rem; }

  .metric-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 14px;
    padding: 1.2rem 1rem;
    text-align: center;
    backdrop-filter: blur(10px);
  }
  .metric-card .val { font-size: 1.8rem; font-weight: 700; color: #00d4ff; }
  .metric-card .lbl { font-size: 0.72rem; color: #64748b; text-transform: uppercase; letter-spacing: 0.5px; margin-top: 0.3rem; }

  .winner-badge {
    display: inline-block;
    background: linear-gradient(135deg, rgba(0,212,255,0.15), rgba(124,77,255,0.15));
    border: 1px solid rgba(0,212,255,0.3);
    border-radius: 100px;
    padding: 0.2rem 0.7rem;
    font-size: 0.72rem; font-weight: 600; color: #00d4ff;
  }
  .pending { color: #475569; font-style: italic; font-size: 0.82rem; }

  .section-title { font-size: 1.1rem; font-weight: 600; color: #e2e8f0; margin: 2rem 0 1rem; border-bottom: 1px solid rgba(255,255,255,0.08); padding-bottom: 0.5rem; }

  [data-testid="stMetricValue"] { color: #00d4ff !important; font-weight: 700 !important; }
  [data-testid="stMetricLabel"] { color: #64748b !important; }

  .stDataFrame { background: transparent !important; }
  .stDataFrame table { border-collapse: separate; border-spacing: 0; }
  .stDataFrame th { background: rgba(0,212,255,0.08) !important; color: #00d4ff !important; font-size: 0.78rem !important; text-transform: uppercase !important; letter-spacing: 0.5px !important; }
  .stDataFrame td { font-size: 0.85rem !important; color: #e2e8f0 !important; border-bottom: 1px solid rgba(255,255,255,0.05) !important; }

  [data-testid="stSidebar"] { background: rgba(10,14,26,0.95) !important; border-right: 1px solid rgba(255,255,255,0.06) !important; }
</style>
""", unsafe_allow_html=True)

# ── Load results ──────────────────────────────────────────────────

@st.cache_data(ttl=30)
def load_results() -> dict:
    if not RESULTS_PATH.exists():
        return {"models": [], "last_updated": "N/A"}
    with open(RESULTS_PATH) as f:
        return json.load(f)


data    = load_results()
models  = data.get("models", [])
updated = data.get("last_updated", "N/A")

# ── Header ────────────────────────────────────────────────────────
st.markdown('<div class="dash-title">📊 Model Evaluation Dashboard</div>', unsafe_allow_html=True)
st.markdown(f'<div class="dash-sub">ROUGE · BERTScore · Latency · Model size — last updated: {updated}</div>', unsafe_allow_html=True)

if not models:
    st.info("No results found. Run `python evaluate/evaluate.py --all` on Kaggle GPU to populate results.", icon="⚠️")
    st.stop()

# ── Build dataframe ───────────────────────────────────────────────
df = pd.DataFrame(models)

# Metrics columns
ROUGE_COLS   = ["rouge1", "rouge2", "rougeL"]
BERT_COLS    = ["bertscore_f1"]
ALL_METRICS  = ROUGE_COLS + BERT_COLS

for col in ALL_METRICS + ["latency_s", "adapter_size_mb", "train_examples"]:
    if col not in df.columns:
        df[col] = None

df["train_examples"] = df["train_examples"].fillna(0).astype(int)

# Identify best model per metric (among non-null)
def highlight_best(s: pd.Series):
    """Highlight max value in a column (green), min latency too."""
    styles = [""] * len(s)
    valid  = s.dropna()
    if valid.empty:
        return styles
    best = valid.idxmax()
    styles[s.index.get_loc(best)] = "background-color: rgba(0,212,255,0.12); color: #00d4ff; font-weight: 700"
    return styles

def highlight_best_latency(s: pd.Series):
    styles = [""] * len(s)
    valid  = s.dropna()
    if valid.empty:
        return styles
    best = valid.idxmin()  # lower latency is better
    styles[s.index.get_loc(best)] = "background-color: rgba(0,230,118,0.1); color: #00e676; font-weight: 700"
    return styles

# ── Summary metrics (top-line KPIs) ──────────────────────────────
has_phi3   = df["rouge1"].notna().any()
best_rouge = df["rougeL"].max() if df["rougeL"].notna().any() else None
best_bert  = df["bertscore_f1"].max() if df["bertscore_f1"].notna().any() else None
fastest    = df["latency_s"].min() if df["latency_s"].notna().any() else None
num_models = len(df)

k1, k2, k3, k4 = st.columns(4)
with k1:
    st.metric("Models Compared", num_models)
with k2:
    st.metric("Best ROUGE-L", f"{best_rouge:.3f}" if best_rouge else "Pending")
with k3:
    st.metric("Best BERTScore F1", f"{best_bert:.3f}" if best_bert else "Pending")
with k4:
    st.metric("Fastest Inference", f"{fastest:.2f}s" if fastest else "Pending")


# ── Full comparison table ─────────────────────────────────────────
st.markdown('<div class="section-title">📋 Full Comparison Table</div>', unsafe_allow_html=True)

display_df = df[["name", "method", "train_examples", "rouge1", "rouge2", "rougeL",
                  "bertscore_f1", "latency_s", "adapter_size_mb"]].copy()
display_df.columns = ["Model", "Method", "Train Ex.", "ROUGE-1", "ROUGE-2", "ROUGE-L",
                       "BERTScore F1", "Latency (s)", "Size (MB)"]

styled = (
    display_df.style
    .apply(highlight_best, subset=["ROUGE-1", "ROUGE-2", "ROUGE-L", "BERTScore F1"])
    .apply(highlight_best_latency, subset=["Latency (s)"])
    .format(na_rep="—", precision=3)
    .set_properties(**{"background-color": "transparent", "color": "#e2e8f0"})
)

st.dataframe(styled, use_container_width=True, hide_index=True, height=280)

# ── Bar charts ────────────────────────────────────────────────────
st.markdown('<div class="section-title">📈 Metric Visualisation</div>', unsafe_allow_html=True)

chart_df = df[["name"] + ALL_METRICS + ["latency_s"]].copy()
chart_df = chart_df.set_index("name")

tab1, tab2, tab3 = st.tabs(["ROUGE Scores", "BERTScore F1", "Latency"])

with tab1:
    rouge_chart = chart_df[["rouge1", "rouge2", "rougeL"]].rename(
        columns={"rouge1": "ROUGE-1", "rouge2": "ROUGE-2", "rougeL": "ROUGE-L"}
    )
    if rouge_chart.notna().any().any():
        st.bar_chart(rouge_chart, color=["#00d4ff", "#7c4dff", "#00e676"], height=320)
    else:
        st.markdown('<p class="pending">Run evaluate.py on GPU to populate ROUGE scores.</p>', unsafe_allow_html=True)

with tab2:
    bert_chart = chart_df[["bertscore_f1"]].rename(columns={"bertscore_f1": "BERTScore F1"})
    if bert_chart.notna().any().any():
        st.bar_chart(bert_chart, color=["#7c4dff"], height=320)
    else:
        st.markdown('<p class="pending">BERTScore requires GPU evaluation — run evaluate.py --all.</p>', unsafe_allow_html=True)

with tab3:
    lat_chart = chart_df[["latency_s"]].rename(columns={"latency_s": "Latency (s)"})
    if lat_chart.notna().any().any():
        st.bar_chart(lat_chart, color=["#ff9800"], height=320)
    else:
        st.markdown('<p class="pending">Latency data not yet available.</p>', unsafe_allow_html=True)


# ── Key insights ──────────────────────────────────────────────────
st.markdown('<div class="section-title">💡 Key Insights</div>', unsafe_allow_html=True)

has_results = df["rouge1"].notna().any()

if has_results:
    best_row   = df.loc[df["rougeL"].idxmax()]
    fastest_row = df.loc[df["latency_s"].idxmin()]

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"""
        <div class="metric-card">
          <div style="font-size:0.75rem;color:#64748b;text-transform:uppercase;letter-spacing:0.5px">🏆 Best ROUGE-L</div>
          <div class="val" style="font-size:1.3rem;margin:0.4rem 0">{best_row['name']}</div>
          <div style="color:#00d4ff;font-size:1rem;font-weight:700">ROUGE-L = {best_row['rougeL']:.3f}</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="metric-card">
          <div style="font-size:0.75rem;color:#64748b;text-transform:uppercase;letter-spacing:0.5px">⚡ Fastest Model</div>
          <div class="val" style="font-size:1.3rem;margin:0.4rem 0">{fastest_row['name']}</div>
          <div style="color:#00e676;font-size:1rem;font-weight:700">Latency = {fastest_row['latency_s']:.2f}s</div>
        </div>
        """, unsafe_allow_html=True)
else:
    st.markdown("""
    <div class="metric-card">
      <div class="val" style="font-size:1.2rem">Run <code>python evaluate/evaluate.py --all</code> on Kaggle GPU</div>
      <div class="lbl">to populate all metrics and see insights</div>
    </div>
    """, unsafe_allow_html=True)


# ── Method breakdown ──────────────────────────────────────────────
st.markdown('<div class="section-title">🔬 Method Deep-Dive</div>', unsafe_allow_html=True)

methods = {
    "Prompt Tuning": ("Ultra-lightweight (<1MB). Fastest inference. Limited accuracy due to frozen base model weights. Best for edge/mobile.", "⚡"),
    "LoRA (Full Precision)": ("Adapter layers injected into attention blocks. BF16 precision. Balanced speed and quality. Recommended for hosted GPU servers.", "🔗"),
    "QLoRA (4-bit)": ("Best accuracy across ROUGE and BERTScore. 4-bit NF4 quantization + LoRA adapters. Fits in 5-6GB VRAM. Recommended for cloud deployment.", "🚀"),
}

mcols = st.columns(3)
for col, (method, (desc, icon)) in zip(mcols, methods.items()):
    method_rows = df[df["method"].str.startswith(method.split(" ")[0]) if df["method"].notna().any() else df["method"] == method]
    with col:
        st.markdown(f"""
        <div class="metric-card" style="text-align:left">
          <div style="font-size:1.5rem;margin-bottom:0.5rem">{icon}</div>
          <div style="font-weight:600;color:#e2e8f0;font-size:0.9rem;margin-bottom:0.5rem">{method}</div>
          <div style="font-size:0.8rem;color:#64748b;line-height:1.5">{desc}</div>
        </div>
        """, unsafe_allow_html=True)


# ── Refresh ───────────────────────────────────────────────────────
st.markdown("---")
col_left, col_right = st.columns([4, 1])
with col_right:
    if st.button("🔄 Refresh Results"):
        st.cache_data.clear()
        st.rerun()
with col_left:
    st.markdown(
        f'<span style="font-size:0.75rem;color:#334155">Results file: <code>{RESULTS_PATH}</code> · Updated: {updated}</span>',
        unsafe_allow_html=True,
    )
