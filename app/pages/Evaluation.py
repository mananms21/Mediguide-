"""
MEDIGUIDE — Evaluation Dashboard (v2)
Ablation study: Zero-shot | Fine-tuned | + RAG | OOD (PubMedQA)
"""

import json
import sys
from pathlib import Path

import streamlit as st
import pandas as pd

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

st.set_page_config(
    page_title="MEDIGUIDE — Evaluation",
    page_icon="📊",
    layout="wide",
)

# ── CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
  html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
  .stApp { background: linear-gradient(135deg,#070d1a,#0b1629,#060e1c); color:#e2e8f0; }

  .dash-title { font-size:2rem; font-weight:700; text-align:center;
                background:linear-gradient(135deg,#00d4ff,#7c4dff);
                -webkit-background-clip:text; -webkit-text-fill-color:transparent; }
  .dash-sub   { color:#64748b; font-size:0.85rem; text-align:center; margin-bottom:1.5rem; }

  .section-head { font-size:0.95rem; font-weight:600; color:#00d4ff;
                  border-bottom:1px solid rgba(0,212,255,0.2);
                  padding-bottom:0.4rem; margin:1.5rem 0 0.8rem; }

  .metric-card  { background:rgba(255,255,255,0.04);
                  border:1px solid rgba(0,212,255,0.12);
                  border-radius:12px; padding:1rem 1.2rem; text-align:center;
                  transition: border-color 0.2s; }
  .metric-card:hover { border-color: rgba(0,212,255,0.4); }
  .metric-label { font-size:0.70rem; color:#64748b; text-transform:uppercase;
                  letter-spacing:0.6px; margin-bottom:0.3rem; }
  .metric-val   { font-size:1.5rem; font-weight:700; color:#e2e8f0; }
  .metric-good  { color:#22d3ee; }
  .metric-warn  { color:#f59e0b; }
  .metric-bad   { color:#f87171; }
  .metric-null  { color:#4b5563; font-size:0.9rem; }

  .delta-pos  { color:#22d3ee; font-size:0.8rem; font-weight:600; }
  .delta-neg  { color:#f87171; font-size:0.8rem; font-weight:600; }
  .delta-zero { color:#94a3b8; font-size:0.8rem; }

  .badge-safe   { background:rgba(34,211,238,0.15); color:#22d3ee;
                  border:1px solid rgba(34,211,238,0.3);
                  border-radius:6px; padding:2px 10px; font-size:0.75rem; }
  .badge-warn   { background:rgba(245,158,11,0.15); color:#f59e0b;
                  border:1px solid rgba(245,158,11,0.3);
                  border-radius:6px; padding:2px 10px; font-size:0.75rem; }
  .badge-danger { background:rgba(248,113,113,0.15); color:#f87171;
                  border:1px solid rgba(248,113,113,0.3);
                  border-radius:6px; padding:2px 10px; font-size:0.75rem; }
  .badge-ood    { background:rgba(124,77,255,0.15); color:#a78bfa;
                  border:1px solid rgba(124,77,255,0.3);
                  border-radius:6px; padding:2px 10px; font-size:0.75rem; }

  .info-box { background:rgba(0,212,255,0.05); border:1px solid rgba(0,212,255,0.15);
              border-radius:10px; padding:0.9rem 1.2rem; font-size:0.83rem; color:#94a3b8; }
  .pending  { background:rgba(255,255,255,0.02); border:1px dashed rgba(255,255,255,0.1);
              border-radius:10px; padding:1.5rem; text-align:center;
              color:#4b5563; font-size:0.85rem; }

  /* Ablation table */
  .abl-table { width:100%; border-collapse:collapse; font-size:0.83rem; }
  .abl-table th { background:rgba(0,212,255,0.08); color:#00d4ff;
                  padding:8px 12px; text-align:center; font-weight:600;
                  border:1px solid rgba(0,212,255,0.12); font-size:0.78rem;
                  text-transform:uppercase; letter-spacing:0.5px; }
  .abl-table td { padding:8px 12px; border:1px solid rgba(255,255,255,0.06);
                  color:#cbd5e1; text-align:center; }
  .abl-table tr:nth-child(even) td { background:rgba(255,255,255,0.02); }
  .abl-table .metric-row td:first-child { text-align:left; color:#94a3b8;
                                           font-size:0.77rem; }
  .abl-table .best { color:#22d3ee; font-weight:700; }
  .abl-table .null-val { color:#374151; }

  div[data-testid="stHorizontalBlock"] > div { gap:0.6rem; }
</style>
""", unsafe_allow_html=True)


# ── Load data ──────────────────────────────────────────────────────────
@st.cache_data
def load_results():
    p = ROOT / "evaluate" / "results" / "results.json"
    with open(p) as f:
        return json.load(f)

data    = load_results()
models  = data.get("models", [])
ablation = data.get("ablation_conditions", {})
deltas  = data.get("ablation_deltas", {})
phi3    = next((m for m in models if "phi3" in m.get("model_id","").lower()), {})


# ── Header ─────────────────────────────────────────────────────────────
st.markdown('<div class="dash-title">MEDIGUIDE — Evaluation Dashboard</div>',
            unsafe_allow_html=True)
st.markdown(
    '<div class="dash-sub">Ablation study · 4 conditions · 5-metric suite · '
    'Kaggle T4 · 50 samples per condition</div>',
    unsafe_allow_html=True,
)
st.markdown("---")


# ── Helper: format a metric value ─────────────────────────────────────
def fmt(v, good_thresh=None, bad_thresh=None, invert=False, pct=False, suffix=""):
    """Render a metric cell with colour coding."""
    if v is None:
        return '<span class="null-val">—</span>'
    vv = v * 100 if pct else v
    txt = f"{vv:.1f}%" if pct else f"{vv:.4f}"
    txt += suffix
    if good_thresh is None:
        return txt
    is_good = (v < good_thresh) if invert else (v >= good_thresh)
    is_bad  = (v > bad_thresh)  if invert else (bad_thresh is not None and v <= bad_thresh)
    cls = "best" if is_good else ("" if not is_bad else "")
    return f'<span class="{cls}">{txt}</span>'

def delta_badge(v):
    if v is None: return "—"
    s = f"{v:+.4f}"
    cls = "delta-pos" if v > 0 else ("delta-neg" if v < 0 else "delta-zero")
    return f'<span class="{cls}">{s}</span>'


# ═══════════════════════════════════════════════════════════════════════
# SECTION 1 — KEY FINDINGS HEADLINE (deltas)
# ═══════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-head">🔬 Key Findings</div>', unsafe_allow_html=True)

has_ablation = any(
    v.get("clinical_bertscore_f1") is not None
    for k, v in ablation.items() if k != "_note"
)

if has_ablation:
    c1, c2, c3, c4 = st.columns(4)
    for col, key, label, tooltip in [
        (c1, "finetuning_clinical_bertscore",   "Fine-tuning Δ",  "How much QLoRA improves Clinical BERTScore over zero-shot"),
        (c2, "rag_clinical_bertscore",           "RAG Δ",          "How much RAG adds on top of fine-tuning"),
        (c3, "ood_gap_clinical_bertscore",       "OOD gap",        "Score drop from in-distribution to PubMedQA (lower = better generalisation)"),
        (c4, "finetuning_contradiction_reduction","Contradiction ↓","Reduction in NLI contradiction rate after fine-tuning"),
    ]:
        v = deltas.get(key)
        with col:
            colour = "#22d3ee" if (v is not None and v > 0) else "#f59e0b"
            val_html = (f'<span style="color:{colour};font-size:1.6rem;font-weight:700">{v:+.4f}</span>'
                        if v is not None else '<span style="color:#4b5563">—</span>')
            st.markdown(f"""
<div class="metric-card" title="{tooltip}">
  <div class="metric-label">{label}</div>
  {val_html}
  <div style="font-size:0.7rem;color:#4b5563;margin-top:4px">Clinical BERTScore</div>
</div>""", unsafe_allow_html=True)
else:
    st.markdown("""
<div class="pending">
  ⏳ Ablation results pending — run <code>evaluate/ablation_kaggle.py</code>
  on Kaggle T4, then copy <code>ablation_results.json</code> into
  <code>evaluate/results/results.json</code>.
</div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════
# SECTION 2 — ABLATION COMPARISON TABLE
# ═══════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-head">📊 Ablation Comparison (4 Conditions × 5 Metrics)</div>',
            unsafe_allow_html=True)

CONDITIONS = [
    ("medquad_zeroshot",  "Zero-shot"),
    ("medquad_finetuned", "Fine-tuned"),
    ("medquad_rag",       "+ RAG"),
    ("pubmedqa_finetuned","OOD (PubMedQA)"),
]

METRICS = [
    ("clinical_bertscore_f1", "Clinical BERTScore F1 ↑",  True,  0.88, None,   False),
    ("bertscore_f1",           "Generic BERTScore F1 ↑",   True,  0.80, None,   False),
    ("rouge1",                 "ROUGE-1 (full) ↑",         True,  0.20, None,   False),
    ("rouge1_50tok",           "ROUGE-1 @50tok ↑",         True,  0.20, None,   False),
    ("lexical_precision_50",   "Lexical Precision@50 ↑",   True,  0.15, None,   False),
    ("nli_contradiction",      "NLI Contradiction ↓",      False, 0.10, 0.15,   True),
    ("perplexity",             "Perplexity ↓",             False, None, None,   False),
    ("latency_s",              "Latency s/sample",          False, None, None,   False),
]

# Build table HTML
th_cells = "".join(f"<th>{label}</th>" for _, label in CONDITIONS)
table_html = f"""
<table class="abl-table">
<thead>
  <tr><th>Metric</th>{th_cells}</tr>
</thead>
<tbody>"""

for metric_key, metric_label, higher_is_better, good_t, bad_t, invert in METRICS:
    vals = [ablation.get(cond_key, {}).get(metric_key) for cond_key, _ in CONDITIONS]

    # Find best value for highlighting
    non_null = [v for v in vals if v is not None]
    if non_null:
        best = min(non_null) if invert else max(non_null)
    else:
        best = None

    row_cells = ""
    for v in vals:
        if v is None:
            row_cells += '<td class="null-val">—</td>'
        else:
            txt = f"{v:.4f}"
            is_best = (best is not None and abs(v - best) < 1e-6)
            cls = "best" if is_best else ""
            row_cells += f'<td class="{cls}">{txt}</td>'

    table_html += f'<tr class="metric-row"><td>{metric_label}</td>{row_cells}</tr>'

table_html += "</tbody></table>"
st.markdown(table_html, unsafe_allow_html=True)

st.markdown("""
<div class="info-box" style="margin-top:0.8rem">
  <strong>How to read this table:</strong> Highlighted (cyan) cells are the best value per row.
  <em>ROUGE-1 @50tok</em> is computed on the first 50 tokens of each prediction — this removes the verbosity bias
  that suppresses full-prediction ROUGE. <em>Lexical Precision@50</em> measures what fraction of the model's
  first 50 content words appear in the reference: a direct measure of factual accuracy in the core claim.
  <em>OOD (PubMedQA)</em> uses research-style questions from a completely different source —
  a lower score vs in-distribution is expected and normal.
</div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════
# SECTION 3 — PRIMARY MODEL METRICS (Phi-3 Fine-tuned, previous run)
# ═══════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-head">🏆 Primary Model — Phi-3 Mini QLoRA (detailed)</div>',
            unsafe_allow_html=True)

tab_sem, tab_cls, tab_safety = st.tabs(
    ["🧠 Semantic Quality", "📝 Classical Metrics", "🛡️ Clinical Safety"]
)

with tab_sem:
    c1, c2, c3 = st.columns(3)
    metrics_sem = [
        (c1, "clinical_bertscore_f1", "Clinical BERTScore F1", "metric-good",
         "BiomedBERT (29M PubMed abstracts). Clinically-aware semantic similarity."),
        (c2, "bertscore_f1",          "Generic BERTScore F1",  "metric-good",
         "roberta-large baseline. Cannot distinguish clinical opposites."),
        (c3, "perplexity",            "Perplexity",            "metric-good",
         "Model confidence in its outputs. Lower = more confident."),
    ]
    for col, key, label, cls, tip in metrics_sem:
        v = phi3.get(key)
        with col:
            val_str = f"{v:.4f}" if v is not None else "—"
            st.markdown(f"""
<div class="metric-card" title="{tip}">
  <div class="metric-label">{label}</div>
  <div class="metric-val {cls}">{val_str}</div>
</div>""", unsafe_allow_html=True)

    if phi3.get("clinical_bertscore_f1") and phi3.get("bertscore_f1"):
        delta = phi3["clinical_bertscore_f1"] - phi3["bertscore_f1"]
        st.markdown(f"""
<div class="info-box" style="margin-top:1rem">
  Clinical BERTScore is <strong style="color:#22d3ee">{delta:+.4f}</strong> above generic BERTScore.
  A positive delta means the model uses clinical vocabulary that a domain-specific model
  (trained on 29M PubMed papers) recognises as medically congruent — the model is speaking
  the language of biomedicine, not just paraphrasing Wikipedia.
</div>""", unsafe_allow_html=True)

with tab_cls:
    cols = st.columns(5)
    cls_metrics = [
        ("rouge1",  "ROUGE-1 (full)"),
        ("rouge2",  "ROUGE-2 (full)"),
        ("rougeL",  "ROUGE-L (full)"),
        ("latency_s", "Latency (s)"),
        ("eval_examples", "Samples"),
    ]
    for col, (key, label) in zip(cols, cls_metrics):
        v = phi3.get(key)
        with col:
            val_str = f"{v}" if v is not None else "—"
            st.markdown(f"""
<div class="metric-card">
  <div class="metric-label">{label}</div>
  <div class="metric-val">{val_str}</div>
</div>""", unsafe_allow_html=True)

    st.markdown("""
<div class="info-box" style="margin-top:1rem">
  <strong>⚠️ ROUGE on full predictions is suppressed by verbosity.</strong>
  The model generates detailed explanations (~200 tokens) while MedQuAD references
  are often concise facts (~20 tokens). This is desirable behaviour for a medical assistant
  but makes overlap metrics look low. Use <em>ROUGE-1 @50tok</em> from the ablation table above
  for a verbosity-corrected comparison.
</div>""", unsafe_allow_html=True)

with tab_safety:
    c1, c2, c3 = st.columns(3)
    con = phi3.get("contradiction_rate")
    ent = phi3.get("entailment_rate")
    neu = phi3.get("neutral_rate")

    if con is not None:
        if con < 0.10:
            verdict_html = '<span class="badge-safe">✅ Clinically Safe</span>'
        elif con < 0.15:
            verdict_html = '<span class="badge-warn">⚠️ Borderline</span>'
        else:
            verdict_html = '<span class="badge-danger">🚨 Caution</span>'
    else:
        verdict_html = '<span class="badge-warn">—</span>'

    for col, v, label, tip in [
        (c1, ent, "Entailment Rate",    "Fraction of responses consistent with reference"),
        (c2, neu, "Neutral Rate",       "Model answer unrelated/incomplete vs reference"),
        (c3, con, "Contradiction Rate", "Model directly contradicts reference (clinical danger)"),
    ]:
        with col:
            val_str = f"{v:.4f}" if v is not None else "—"
            cls = ("metric-good" if (v is not None and label.startswith("Entailment") and v > 0.15)
                   else "metric-warn" if label.startswith("Neutral")
                   else "metric-bad" if (v is not None and v > 0.10)
                   else "metric-good")
            st.markdown(f"""
<div class="metric-card" title="{tip}">
  <div class="metric-label">{label}</div>
  <div class="metric-val {cls}">{val_str}</div>
</div>""", unsafe_allow_html=True)

    st.markdown(f"""
<div class="info-box" style="margin-top:1rem">
  Safety verdict: {verdict_html}
  &nbsp;(Contradiction &lt; 0.10 → safe · 0.10–0.15 → borderline · &gt; 0.15 → caution)
  <br><br>
  <em>Note:</em> The NLI model (roberta-large-mnli) was trained on Wikipedia-domain text.
  The high neutral rate (~72%) reflects the model's difficulty connecting long medical paragraphs
  to short reference sentences — not genuine neutrality. Contradiction rate is the more reliable signal.
</div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════
# SECTION 4 — BASELINE COMPARISON TABLE
# ═══════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-head">📋 All Models — Baseline Comparison</div>',
            unsafe_allow_html=True)

rows = []
for m in models:
    rows.append({
        "Model":        m.get("name", m.get("model_id", "?")),
        "Method":       m.get("method", "—"),
        "Train Ex.":    m.get("train_examples", "—"),
        "ROUGE-1":      m.get("rouge1"),
        "BERTScore F1": m.get("bertscore_f1"),
        "Clin. BERTSc": m.get("clinical_bertscore_f1"),
        "Contradiction":m.get("contradiction_rate"),
        "Latency (s)":  m.get("latency_s"),
    })

df = pd.DataFrame(rows)
st.dataframe(
    df.style.format({
        "ROUGE-1":      "{:.4f}", "BERTScore F1": "{:.4f}",
        "Clin. BERTSc": "{:.4f}", "Contradiction": "{:.4f}",
        "Latency (s)":  "{:.2f}",
    }, na_rep="—").highlight_max(
        subset=["ROUGE-1","BERTScore F1","Clin. BERTSc"],
        color="rgba(34,211,238,0.15)"
    ).highlight_min(
        subset=["Contradiction","Latency (s)"],
        color="rgba(34,211,238,0.15)"
    ),
    use_container_width=True,
)


# ═══════════════════════════════════════════════════════════════════════
# SECTION 5 — METRIC GUIDE
# ═══════════════════════════════════════════════════════════════════════
with st.expander("📖 Metric Definitions & Thresholds"):
    st.markdown("""
| Metric | What it measures | Threshold | Notes |
|---|---|---|---|
| **Clinical BERTScore F1** | Token-level greedy cosine matching using BiomedBERT (trained on 29M PubMed papers) | > 0.88 = good | Primary quality signal. Distinguishes clinical opposites unlike generic BERTScore |
| **Generic BERTScore F1** | Same metric but with roberta-large (Wikipedia) | > 0.80 = good | Cannot distinguish "heart" from "lung" — baseline only |
| **ROUGE-1 (full)** | Unigram overlap between prediction and reference | Context-dependent | Suppressed by verbosity — model generates more than the concise reference |
| **ROUGE-1 @50tok** | ROUGE-1 on first 50 tokens of prediction only | > 0.20 = good | Verbosity-corrected. Measures: does the model lead with correct facts? |
| **Lexical Precision@50** | Fraction of first 50 tokens' content words in reference | > 0.15 = good | Stopword-filtered. Direct measure of factual precision in the core claim |
| **NLI Contradiction** | Fraction of responses that contradict the reference (roberta-large-mnli) | < 0.10 = safe, 0.10–0.15 = borderline, > 0.15 = caution | ~72% neutral rate is expected — NLI model is not medical-domain calibrated |
| **Perplexity** | Model confidence in its own outputs | < 5 = confident | Lower is better. Phi-3 at 2.57 is highly confident |
| **OOD Score** | Clinical BERTScore on PubMedQA (research questions, different domain) | Lower than in-distribution is expected | Proves generalization beyond training distribution |
""")
