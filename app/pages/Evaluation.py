"""
MEDIGUIDE — Evaluation Dashboard
Shows classical, semantic, and clinical safety metrics side-by-side.
"""

import json
import sys
from pathlib import Path

import streamlit as st
import pandas as pd

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

st.set_page_config(
    page_title="MEDIGUIDE — Evaluation Dashboard",
    page_icon="📊",
    layout="wide",
)

# ── CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
  html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
  .stApp { background: linear-gradient(135deg,#070d1a,#0b1629,#060e1c); color:#e2e8f0; }

  .dash-header { text-align:center; padding:2rem 0 1rem; }
  .dash-title  { font-size:2rem; font-weight:700;
                 background:linear-gradient(135deg,#00d4ff,#7c4dff);
                 -webkit-background-clip:text; -webkit-text-fill-color:transparent; }
  .dash-sub    { color:#64748b; font-size:0.9rem; margin-top:0.3rem; }

  .section-head { font-size:1rem; font-weight:600; color:#00d4ff;
                  border-bottom:1px solid rgba(0,212,255,0.2);
                  padding-bottom:0.4rem; margin:1.5rem 0 0.8rem; }

  .metric-card  { background:rgba(255,255,255,0.04);
                  border:1px solid rgba(0,212,255,0.12);
                  border-radius:12px; padding:1rem 1.2rem; text-align:center; }
  .metric-label { font-size:0.72rem; color:#64748b; text-transform:uppercase;
                  letter-spacing:0.6px; margin-bottom:0.3rem; }
  .metric-val   { font-size:1.6rem; font-weight:700; color:#e2e8f0; }
  .metric-good  { color:#22d3ee; }
  .metric-warn  { color:#fbbf24; }
  .metric-bad   { color:#f87171; }

  .insight-box  { background:rgba(0,212,255,0.06);
                  border:1px solid rgba(0,212,255,0.2);
                  border-radius:10px; padding:1rem 1.2rem;
                  font-size:0.85rem; color:#94a3b8; margin-top:0.5rem; }

  .danger-box   { background:rgba(248,113,113,0.08);
                  border:1px solid rgba(248,113,113,0.25);
                  border-radius:10px; padding:0.8rem 1rem;
                  font-size:0.82rem; color:#fca5a5; }
  .safe-box     { background:rgba(34,211,238,0.07);
                  border:1px solid rgba(34,211,238,0.25);
                  border-radius:10px; padding:0.8rem 1rem;
                  font-size:0.82rem; color:#67e8f9; }
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────
st.markdown("""
<div class="dash-header">
  <div class="dash-title">📊 Evaluation Dashboard</div>
  <div class="dash-sub">Classical · Semantic · Clinical Safety — four evaluation levels</div>
</div>
""", unsafe_allow_html=True)

# ── Load data ─────────────────────────────────────────────────────────
results_path = ROOT / "evaluate" / "results" / "results.json"
try:
    with open(results_path) as f:
        data = json.load(f)
    models = data["models"]
    last_updated = data.get("last_updated", "unknown")
except Exception as e:
    st.error(f"Could not load results.json: {e}")
    st.stop()

phi3 = next((m for m in models if "phi3" in m["model_id"].lower()
             or "Phi-3" in m.get("base_model", "")), None)

# ── Model selector ────────────────────────────────────────────────────
model_names = [m["name"] for m in models]
selected_name = st.sidebar.selectbox("Select model", model_names,
                                     index=0 if phi3 else 0)
sel = next(m for m in models if m["name"] == selected_name)

st.sidebar.markdown("---")
st.sidebar.markdown(f"**Last updated:** {last_updated}")
st.sidebar.markdown(f"**Eval samples:** {sel.get('eval_examples','—')}")
st.sidebar.markdown(f"**Train examples:** {sel.get('train_examples','—')}")
st.sidebar.markdown(f"**Method:** {sel.get('method','—')}")
st.sidebar.markdown(f"**Adapter size:** {sel.get('adapter_size_mb','—')} MB")

def fmt(v, decimals=4):
    return f"{v:.{decimals}f}" if v is not None else "—"

def color_class(v, good_thresh, bad_thresh, invert=False):
    """Returns CSS class based on value vs thresholds."""
    if v is None:
        return ""
    if invert:
        if v <= good_thresh: return "metric-good"
        if v >= bad_thresh:  return "metric-bad"
        return "metric-warn"
    else:
        if v >= good_thresh: return "metric-good"
        if v <= bad_thresh:  return "metric-bad"
        return "metric-warn"

# ──────────────────────────────────────────────────────────────────────
# SECTION 1: Classical Metrics
# ──────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-head">📏 Classical Metrics</div>', unsafe_allow_html=True)

c1, c2, c3, c4, c5 = st.columns(5)

def metric_card(col, label, value, css_class="", decimals=4):
    col.markdown(f"""
    <div class="metric-card">
      <div class="metric-label">{label}</div>
      <div class="metric-val {css_class}">{fmt(value, decimals)}</div>
    </div>""", unsafe_allow_html=True)

metric_card(c1, "ROUGE-1", sel.get("rouge1"), color_class(sel.get("rouge1"), 0.25, 0.10))
metric_card(c2, "ROUGE-2", sel.get("rouge2"), color_class(sel.get("rouge2"), 0.08, 0.03))
metric_card(c3, "ROUGE-L", sel.get("rougeL"), color_class(sel.get("rougeL"), 0.15, 0.06))
metric_card(c4, "Perplexity", sel.get("perplexity"),
            color_class(sel.get("perplexity"), 0, 30, invert=True), decimals=1)
metric_card(c5, "Latency (s)", sel.get("latency_s"),
            color_class(sel.get("latency_s"), 0, 15, invert=True), decimals=2)

st.markdown("""
<div class="insight-box">
  ⚠️ <b>ROUGE limitation:</b> ROUGE measures n-gram overlap. Medical models often score lower
  because they generate longer, more detailed answers than the reference — not because they're wrong.
  A low ROUGE score alone does <em>not</em> indicate clinical inaccuracy.
</div>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────
# SECTION 2: Semantic Similarity
# ──────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-head">🧠 Semantic Similarity</div>', unsafe_allow_html=True)

c1, c2, c3, c4, c5 = st.columns(5)
metric_card(c1, "BERTScore P", sel.get("bertscore_p"),
            color_class(sel.get("bertscore_p"), 0.80, 0.70))
metric_card(c2, "BERTScore R", sel.get("bertscore_r"),
            color_class(sel.get("bertscore_r"), 0.80, 0.70))
metric_card(c3, "BERTScore F1 (Generic)", sel.get("bertscore_f1"),
            color_class(sel.get("bertscore_f1"), 0.80, 0.70))
metric_card(c4, "Clinical BERTScore F1", sel.get("clinical_bertscore_f1"),
            color_class(sel.get("clinical_bertscore_f1"), 0.75, 0.65))

# Delta metric
generic = sel.get("bertscore_f1")
clinical = sel.get("clinical_bertscore_f1")
if generic is not None and clinical is not None:
    delta = round(clinical - generic, 4)
    delta_class = "metric-warn" if delta < -0.05 else "metric-good"
    c5.markdown(f"""
    <div class="metric-card">
      <div class="metric-label">Δ Clinical − Generic</div>
      <div class="metric-val {delta_class}">{delta:+.4f}</div>
    </div>""", unsafe_allow_html=True)
else:
    metric_card(c5, "Δ Clinical − Generic", None)

st.markdown("""
<div class="insight-box">
  🔬 <b>Why Clinical BERTScore matters:</b> Generic BERTScore uses <code>roberta-large</code>
  trained on Wikipedia — where "heart" and "lung" are close neighbours. BiomedBERT is trained
  on 29M PubMed abstracts, so clinically distinct terms are further apart in embedding space.
  A large negative Δ (Clinical &lt; Generic) means the model uses imprecise clinical language.
</div>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────
# SECTION 3: Clinical Accuracy (NER Entity F1)
# ──────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-head">🏥 Clinical Accuracy — Medical Entity F1</div>',
            unsafe_allow_html=True)

c1, c2, c3, c4 = st.columns(4)
metric_card(c1, "Entity Precision", sel.get("entity_precision"),
            color_class(sel.get("entity_precision"), 0.60, 0.35))
metric_card(c2, "Entity Recall", sel.get("entity_recall"),
            color_class(sel.get("entity_recall"), 0.55, 0.30))
metric_card(c3, "Entity F1", sel.get("entity_f1"),
            color_class(sel.get("entity_f1"), 0.55, 0.30))

with c4:
    ep = sel.get("entity_precision")
    er = sel.get("entity_recall")
    if ep and er:
        if ep > er + 0.1:
            insight = "📌 High precision, lower recall — model uses correct terms but misses some key ones."
        elif er > ep + 0.1:
            insight = "📌 High recall, lower precision — model mentions correct terms but adds extras."
        else:
            insight = "📌 Precision ≈ Recall — well-balanced entity coverage."
        st.markdown(f'<div class="insight-box" style="height:100%">{insight}</div>',
                    unsafe_allow_html=True)
    else:
        st.markdown('<div class="insight-box">Run clinical_kaggle.py to populate.</div>',
                    unsafe_allow_html=True)

st.markdown("""
<div class="insight-box">
  🧬 <b>How it works:</b> scispacy (<code>en_core_sci_md</code>) extracts medical/scientific named
  entities from both prediction and reference. Precision = correct entities predicted, Recall = 
  reference entities the model captured. This catches "left ventricle" → "right ventricle" errors
  that generic BERTScore would miss.
</div>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────
# SECTION 4: Factual Safety — NLI + Hallucination
# ──────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-head">🛡️ Factual Safety — Contradiction & Hallucination</div>',
            unsafe_allow_html=True)

c1, c2, c3, c4 = st.columns(4)

# Entailment — higher is better
metric_card(c1, "Entailment Rate ↑", sel.get("entailment_rate"),
            color_class(sel.get("entailment_rate"), 0.50, 0.30))
metric_card(c2, "Neutral Rate", sel.get("neutral_rate"))
# Contradiction — lower is better (invert scale)
metric_card(c3, "Contradiction Rate ↓", sel.get("contradiction_rate"),
            color_class(sel.get("contradiction_rate"), 0, 0.15, invert=True))
# Hallucination — lower is better
metric_card(c4, "Hallucination Rate ↓", sel.get("hallucination_rate"),
            color_class(sel.get("hallucination_rate"), 0, 0.30, invert=True))

# Safety verdict
cont = sel.get("contradiction_rate")
hall = sel.get("hallucination_rate")

if cont is not None and hall is not None:
    if cont < 0.10 and hall < 0.25:
        st.markdown(f"""
        <div class="safe-box">
          ✅ <b>Clinically safe:</b> Contradiction rate {cont:.3f} &lt; 0.10 threshold.
          Hallucination rate {hall:.3f} &lt; 0.25 threshold.
          The model is unlikely to generate information that directly contradicts established medical facts.
        </div>""", unsafe_allow_html=True)
    elif cont >= 0.15:
        st.markdown(f"""
        <div class="danger-box">
          ⚠️ <b>Caution — high contradiction rate ({cont:.3f}):</b> The model may generate responses
          that directly conflict with the reference medical information. Review generated outputs
          before deployment. Consider additional RLHF or DPO fine-tuning on contradiction pairs.
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="insight-box">
          🔶 Contradiction rate {cont:.3f} is moderate. Monitor carefully before clinical deployment.
        </div>""", unsafe_allow_html=True)
else:
    st.markdown("""
    <div class="insight-box">
      Run <code>evaluate/clinical_kaggle.py</code> on Kaggle T4 to compute NLI and hallucination scores.
    </div>""", unsafe_allow_html=True)

st.markdown("""
<div class="insight-box">
  ⚖️ <b>NLI method:</b> <code>roberta-large-mnli</code> checks if the prediction
  <em>contradicts</em> the reference. Premise = reference (ground truth), Hypothesis = prediction.
  CONTRADICTION means the model said something clinically opposite to the correct answer —
  the most dangerous failure mode.
  <br><br>
  🔍 <b>Hallucination:</b> Medical entities in the prediction that do not appear in either 
  the question or reference are flagged as potentially hallucinated.
</div>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────
# SECTION 5: Model Comparison Table
# ──────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-head">📋 Model Comparison</div>', unsafe_allow_html=True)

rows = []
for m in models:
    rows.append({
        "Model":          m["name"],
        "Method":         m.get("method", "—"),
        "ROUGE-1":        m.get("rouge1"),
        "ROUGE-L":        m.get("rougeL"),
        "BERTScore F1":   m.get("bertscore_f1"),
        "Clin. BERT F1":  m.get("clinical_bertscore_f1"),
        "Entity F1":      m.get("entity_f1"),
        "Contradiction↓": m.get("contradiction_rate"),
        "Hallucination↓": m.get("hallucination_rate"),
        "Latency (s)":    m.get("latency_s"),
    })

df = pd.DataFrame(rows)

def style_cell(val):
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return "color: #475569"
    return ""

st.dataframe(
    df.style.format({
        "ROUGE-1": lambda x: f"{x:.4f}" if x else "—",
        "ROUGE-L": lambda x: f"{x:.4f}" if x else "—",
        "BERTScore F1":  lambda x: f"{x:.4f}" if x else "—",
        "Clin. BERT F1": lambda x: f"{x:.4f}" if x else "—",
        "Entity F1":     lambda x: f"{x:.4f}" if x else "—",
        "Contradiction↓":lambda x: f"{x:.4f}" if x else "—",
        "Hallucination↓":lambda x: f"{x:.4f}" if x else "—",
        "Latency (s)":   lambda x: f"{x:.2f}s" if x else "—",
    }).applymap(style_cell),
    use_container_width=True,
    hide_index=True,
)

# ──────────────────────────────────────────────────────────────────────
# SECTION 6: Metric Explanation Reference
# ──────────────────────────────────────────────────────────────────────
with st.expander("📖 Metric Definitions & Thresholds"):
    st.markdown("""
| Metric | What it measures | Good | Caution | Bad |
|--------|-----------------|------|---------|-----|
| **ROUGE-1/2/L** | N-gram overlap with reference | >0.25 | 0.10–0.25 | <0.10 |
| **Perplexity** | Model probability of reference (lower=better) | <10 | 10–30 | >30 |
| **BERTScore F1** | Semantic similarity (general) | >0.80 | 0.70–0.80 | <0.70 |
| **Clinical BERTScore F1** | Semantic similarity (clinical domain) | >0.75 | 0.65–0.75 | <0.65 |
| **Entity F1** | Medical term precision/recall | >0.55 | 0.30–0.55 | <0.30 |
| **Entailment Rate** | Fraction of responses consistent with truth ↑ | >0.50 | 0.30–0.50 | <0.30 |
| **Contradiction Rate** | Fraction of responses clinically wrong ↓ | <0.10 | 0.10–0.15 | >0.15 |
| **Hallucination Rate** | Fraction of novel ungrounded entities ↓ | <0.20 | 0.20–0.35 | >0.35 |

> **Key insight:** A model can score 0.80 on generic BERTScore while having a 0.15 contradiction rate —
> meaning 15% of its responses directly conflict with correct medical information.
> Clinical BERTScore + NLI together give a much more honest picture.
    """)
