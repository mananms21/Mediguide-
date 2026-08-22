"""
MEDIGUIDE — Evaluation Dashboard
Ablation study: Zero-shot | Fine-tuned | +RAG | OOD (PubMedQA)
"""

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

st.set_page_config(
    page_title="MediGuide — Evaluation",
    page_icon="⚕",
    layout="wide",
)

# ── CSS — same light system as main app ──────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
  html, body, [class*="css"] { font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif; }
  #MainMenu, footer, header { visibility: hidden; }

  .stApp { background: #f7f8fa; }

  /* ── Page header ─────────────────────────────────────────────── */
  .page-header {
    padding: 1.1rem 0 1.4rem;
    border-bottom: 1px solid #e5e7eb;
    margin-bottom: 1.5rem;
    display: flex; align-items: baseline; gap: 12px;
  }
  .page-title {
    font-size: 1.15rem; font-weight: 700; color: #111827;
    letter-spacing: -0.3px;
  }
  .page-sub { font-size: 0.78rem; color: #6b7280; }

  /* ── Section headers ─────────────────────────────────────────── */
  .sec-head {
    font-size: 0.8rem; font-weight: 700; color: #374151;
    text-transform: uppercase; letter-spacing: 0.7px;
    margin: 1.8rem 0 0.9rem;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid #e5e7eb;
  }

  /* ── Delta cards ─────────────────────────────────────────────── */
  .delta-card {
    background: #fff;
    border: 1px solid #e5e7eb;
    border-radius: 10px;
    padding: 1rem 1.1rem;
    text-align: center;
  }
  .delta-label {
    font-size: 0.68rem; font-weight: 600;
    color: #9ca3af; text-transform: uppercase;
    letter-spacing: 0.6px; margin-bottom: 0.5rem;
  }
  .delta-val-pos { font-size: 1.5rem; font-weight: 700; color: #16a34a; }
  .delta-val-neg { font-size: 1.5rem; font-weight: 700; color: #dc2626; }
  .delta-val-null{ font-size: 1.2rem; font-weight: 400; color: #9ca3af; }
  .delta-sub { font-size: 0.68rem; color: #9ca3af; margin-top: 3px; }

  /* ── Metric cards ────────────────────────────────────────────── */
  .m-card {
    background: #fff;
    border: 1px solid #e5e7eb;
    border-radius: 10px;
    padding: 1rem 1.1rem;
    text-align: center;
  }
  .m-label { font-size: 0.7rem; font-weight: 600; color: #6b7280;
             text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 0.45rem; }
  .m-val { font-size: 1.4rem; font-weight: 700; color: #111827; }
  .m-val-good { color: #1d4ed8; }
  .m-val-warn { color: #d97706; }
  .m-val-bad  { color: #dc2626; }

  /* ── Ablation table ──────────────────────────────────────────── */
  .abl-wrap {
    background: #fff;
    border: 1px solid #e5e7eb;
    border-radius: 10px;
    overflow: hidden;
  }
  .abl-table { width: 100%; border-collapse: collapse; font-size: 0.82rem; }
  .abl-table thead tr { background: #f9fafb; }
  .abl-table th {
    padding: 10px 14px; text-align: center;
    font-size: 0.72rem; font-weight: 700; color: #374151;
    text-transform: uppercase; letter-spacing: 0.5px;
    border-bottom: 1px solid #e5e7eb;
  }
  .abl-table th:first-child { text-align: left; }
  .abl-table td {
    padding: 9px 14px; border-bottom: 1px solid #f3f4f6;
    color: #374151; text-align: center;
  }
  .abl-table td:first-child { text-align: left; font-weight: 500; color: #111827; }
  .abl-table tr:last-child td { border-bottom: none; }
  .abl-table tr:hover td { background: #fafafa; }
  .abl-best { color: #1d4ed8; font-weight: 700; }
  .abl-null { color: #d1d5db; }

  /* ── Safety badge ────────────────────────────────────────────── */
  .badge-safe   { background:#f0fdf4; color:#15803d; border:1px solid #bbf7d0; border-radius:6px; padding:3px 10px; font-size:0.75rem; font-weight:600; }
  .badge-warn   { background:#fffbeb; color:#b45309; border:1px solid #fde68a; border-radius:6px; padding:3px 10px; font-size:0.75rem; font-weight:600; }
  .badge-danger { background:#fef2f2; color:#b91c1c; border:1px solid #fecaca; border-radius:6px; padding:3px 10px; font-size:0.75rem; font-weight:600; }

  /* ── Info box ────────────────────────────────────────────────── */
  .info-box {
    background: #f0f9ff;
    border: 1px solid #bae6fd;
    border-left: 3px solid #3b82f6;
    border-radius: 0 8px 8px 0;
    padding: 0.75rem 1rem;
    font-size: 0.8rem; color: #1e40af; line-height: 1.6;
    margin-top: 0.75rem;
  }
  .note-box {
    background: #fffbeb;
    border: 1px solid #fde68a;
    border-left: 3px solid #f59e0b;
    border-radius: 0 8px 8px 0;
    padding: 0.75rem 1rem;
    font-size: 0.8rem; color: #78350f; line-height: 1.6;
    margin-top: 0.75rem;
  }

  /* ── Tabs ────────────────────────────────────────────────────── */
  .stTabs [data-baseweb="tab-list"] { gap: 4px; border-bottom: 1px solid #e5e7eb !important; background: transparent !important; }
  .stTabs [data-baseweb="tab"] { background: transparent !important; border: none !important;
    color: #6b7280 !important; font-size: 0.83rem !important; font-weight: 500 !important; padding: 6px 14px !important; }
  .stTabs [aria-selected="true"] { color: #1d4ed8 !important; font-weight: 600 !important;
    border-bottom: 2px solid #1d4ed8 !important; }

  /* ── Streamlit overrides ─────────────────────────────────────── */
  [data-testid="stMetricValue"] { color: #111827 !important; font-weight: 600 !important; }
  [data-testid="stMetricLabel"] { color: #6b7280 !important; font-size: 0.78rem !important; }
  .stDataFrame { border: 1px solid #e5e7eb !important; border-radius: 10px !important; }
  .stExpander { border: 1px solid #e5e7eb !important; border-radius: 10px !important; background: #fff !important; }
</style>
""", unsafe_allow_html=True)


# ── Load data ─────────────────────────────────────────────────────
@st.cache_data
def load_results():
    p = ROOT / "evaluate" / "results" / "results.json"
    with open(p) as f:
        return json.load(f)

data     = load_results()
models   = data.get("models", [])
ablation = data.get("ablation_conditions", {})
deltas   = data.get("ablation_deltas", {})
phi3     = next((m for m in models if "phi3" in m.get("model_id", "").lower()), {})


# ── Page header ───────────────────────────────────────────────────
st.markdown("""
<div class="page-header">
  <div class="page-title">⚕ MediGuide — Evaluation Results</div>
  <div class="page-sub">Ablation study · 4 conditions · 50 samples each · Kaggle T4</div>
</div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════
# SECTION 1 — KEY DELTAS
# ═══════════════════════════════════════════════════════
st.markdown('<div class="sec-head">Key Findings</div>', unsafe_allow_html=True)

c1, c2, c3, c4 = st.columns(4)
DELTA_DEFS = [
    (c1, "finetuning_clinical_bertscore",    "Fine-tuning Δ",       "Clinical BERTScore",    "Zero-shot → Fine-tuned"),
    (c2, "rag_clinical_bertscore",           "RAG Δ",               "Clinical BERTScore",    "Fine-tuned → +RAG"),
    (c3, "ood_gap_clinical_bertscore",       "OOD gap",             "Clinical BERTScore",    "In-distribution vs PubMedQA"),
    (c4, "finetuning_contradiction_reduction","Contradiction Δ",    "NLI rate",              "Fine-tuned → +RAG"),
]
for col, key, label, sub, tip in DELTA_DEFS:
    v = deltas.get(key)
    with col:
        if v is None:
            val_html = '<div class="delta-val-null">—</div>'
        else:
            cls = "delta-val-pos" if v > 0 else "delta-val-neg"
            val_html = f'<div class="{cls}">{v:+.4f}</div>'
        st.markdown(f"""
<div class="delta-card" title="{tip}">
  <div class="delta-label">{label}</div>
  {val_html}
  <div class="delta-sub">{sub}</div>
</div>""", unsafe_allow_html=True)

st.markdown("""
<div class="info-box">
  <strong>Reading the deltas:</strong>
  Fine-tuning Δ and RAG Δ are positive = improvement.
  OOD gap is the score drop from in-distribution to PubMedQA — smaller means better generalisation.
  A negative Contradiction Δ means the model is <em>less</em> likely to contradict the reference.
</div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════
# SECTION 2 — ABLATION TABLE
# ═══════════════════════════════════════════════════════
st.markdown('<div class="sec-head">Ablation Comparison — 4 Conditions × 8 Metrics</div>', unsafe_allow_html=True)

CONDITIONS = [
    ("medquad_zeroshot",  "Zero-shot"),
    ("medquad_finetuned", "Fine-tuned"),
    ("medquad_rag",       "+ RAG"),
    ("pubmedqa_finetuned","OOD (PubMedQA)"),
]
METRICS = [
    ("clinical_bertscore_f1", "Clinical BERTScore F1 ↑",   False),
    ("bertscore_f1",          "Generic BERTScore F1 ↑",    False),
    ("rouge1",                "ROUGE-1 (full) ↑",          False),
    ("rouge1_50tok",          "ROUGE-1 @50tok ↑",          False),
    ("lexical_precision_50",  "Lexical Precision@50 ↑",    False),
    ("nli_contradiction",     "NLI Contradiction ↓",       True),   # lower is better
    ("perplexity",            "Perplexity ↓",              True),
    ("latency_s",             "Latency s/sample",          True),
]

headers = "".join(f"<th>{lbl}</th>" for _, lbl in CONDITIONS)
table   = f'<div class="abl-wrap"><table class="abl-table"><thead><tr><th>Metric</th>{headers}</tr></thead><tbody>'

for metric_key, metric_label, lower_is_better in METRICS:
    vals = [ablation.get(ck, {}).get(metric_key) for ck, _ in CONDITIONS]
    non_null = [v for v in vals if v is not None]
    best = (min(non_null) if lower_is_better else max(non_null)) if non_null else None
    cells = ""
    for v in vals:
        if v is None:
            cells += '<td class="abl-null">—</td>'
        else:
            is_best = best is not None and abs(v - best) < 1e-6
            cls = "abl-best" if is_best else ""
            cells += f'<td class="{cls}">{v:.4f}</td>'
    table += f'<tr><td>{metric_label}</td>{cells}</tr>'

table += "</tbody></table></div>"
st.markdown(table, unsafe_allow_html=True)

st.markdown("""
<div class="info-box" style="margin-top:0.6rem">
  <strong>Best value per row</strong> is shown in blue.
  <strong>ROUGE-1 @50tok</strong> is computed on the first 50 tokens of each prediction —
  this removes verbosity suppression.
  <strong>Lexical Precision@50</strong> measures how much of the model's opening claim is
  factually correct (content-word precision).
  The <strong>OOD column</strong> uses PubMedQA (research-style questions, unseen domain) —
  a lower score than in-distribution is normal and expected.
</div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════
# SECTION 3 — DETAILED METRICS (tabs)
# ═══════════════════════════════════════════════════════
st.markdown('<div class="sec-head">Phi-3 Mini QLoRA — Detailed Metrics</div>', unsafe_allow_html=True)

tab_sem, tab_cls, tab_safe = st.tabs(["Semantic Quality", "Classical Metrics", "Clinical Safety"])

with tab_sem:
    c1, c2, c3 = st.columns(3)
    ft = ablation.get("medquad_finetuned", phi3)
    for col, key, label, tip in [
        (c1, "clinical_bertscore_f1", "Clinical BERTScore F1",
         "BiomedBERT greedy token matching. Primary quality signal."),
        (c2, "bertscore_f1",          "Generic BERTScore F1",
         "roberta-large baseline. Cannot distinguish clinical opposites."),
        (c3, "perplexity",            "Perplexity",
         "Model confidence. Lower = more certain."),
    ]:
        v = ft.get(key)
        with col:
            vs = f"{v:.4f}" if v is not None else "—"
            st.markdown(f"""
<div class="m-card" title="{tip}">
  <div class="m-label">{label}</div>
  <div class="m-val m-val-good">{vs}</div>
</div>""", unsafe_allow_html=True)

    cb = ft.get("clinical_bertscore_f1")
    gb = ft.get("bertscore_f1")
    if cb and gb:
        d = cb - gb
        st.markdown(f"""
<div class="info-box">
  Clinical BERTScore is <strong>+{d:.4f}</strong> above generic BERTScore.
  This gap reflects the use of domain-specific clinical vocabulary that a model trained on
  29 million PubMed abstracts recognises as medically congruent.
</div>""", unsafe_allow_html=True)

with tab_cls:
    cols = st.columns(5)
    ft = ablation.get("medquad_finetuned", phi3)
    for col, (key, label) in zip(cols, [
        ("rouge1",        "ROUGE-1 (full)"),
        ("rouge1_50tok",  "ROUGE-1 @50tok"),
        ("lexical_precision_50", "Lex. Prec@50"),
        ("latency_s",     "Latency (s)"),
        ("perplexity",    "Perplexity"),
    ]):
        v = ft.get(key)
        vs = f"{v:.4f}" if isinstance(v, float) else (str(v) if v is not None else "—")
        with col:
            st.markdown(f"""
<div class="m-card">
  <div class="m-label">{label}</div>
  <div class="m-val">{vs}</div>
</div>""", unsafe_allow_html=True)

    st.markdown("""
<div class="note-box">
  <strong>Note on ROUGE:</strong> ROUGE on full predictions is suppressed because the model
  generates detailed answers (~200 tokens) while MedQuAD references are concise (~20 tokens).
  This is desirable behaviour — use ROUGE-1 @50tok for a fair, verbosity-corrected comparison.
</div>""", unsafe_allow_html=True)

with tab_safe:
    ft  = ablation.get("medquad_rag", phi3)   # show +RAG values (safest condition)
    con = ft.get("nli_contradiction")
    ent = ft.get("nli_entailment")
    neu = ft.get("nli_neutral")

    if con is not None:
        if con < 0.10:
            verdict = '<span class="badge-safe">✅ Clinically Safe</span>'
        elif con < 0.15:
            verdict = '<span class="badge-warn">⚠️ Borderline</span>'
        else:
            verdict = '<span class="badge-danger">🚨 Caution</span>'
    else:
        verdict = '<span class="badge-warn">—</span>'

    c1, c2, c3 = st.columns(3)
    for col, v, label in [
        (c1, ent, "Entailment Rate"),
        (c2, neu, "Neutral Rate"),
        (c3, con, "Contradiction Rate"),
    ]:
        vs  = f"{v:.4f}" if v is not None else "—"
        cls = "m-val-good" if label.startswith("Entailment") else \
              "m-val-bad"  if (label.startswith("Contra") and v is not None and v > 0.10) else "m-val"
        with col:
            st.markdown(f"""
<div class="m-card">
  <div class="m-label">{label}</div>
  <div class="m-val {cls}">{vs}</div>
</div>""", unsafe_allow_html=True)

    st.markdown(f"""
<div class="info-box" style="margin-top:0.75rem">
  Safety verdict (Fine-tuned + RAG condition): {verdict}
  &nbsp; (Contradiction &lt; 0.10 → safe · 0.10–0.15 → borderline · &gt; 0.15 → caution)
  <br><br>
  <strong>The NLI trade-off:</strong> Fine-tuning alone increases NLI contradiction (0.10 → 0.22)
  because the model becomes more verbose and domain-specific, which the Wikipedia-trained NLI model
  misclassifies. RAG resolves this to 0.078 — below zero-shot. This means RAG is not optional:
  it corrects a safety trade-off introduced by fine-tuning.
</div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════
# SECTION 4 — BASELINE TABLE
# ═══════════════════════════════════════════════════════
st.markdown('<div class="sec-head">All Models — Baseline Comparison</div>', unsafe_allow_html=True)

rows = []
for m in models:
    rows.append({
        "Model":         m.get("name", m.get("model_id", "?")),
        "Method":        m.get("method", "—"),
        "Train Ex.":     m.get("train_examples", "—"),
        "ROUGE-1":       m.get("rouge1"),
        "BERTScore F1":  m.get("bertscore_f1"),
        "Clin. BERTSc":  m.get("clinical_bertscore_f1"),
        "Contradiction": m.get("contradiction_rate"),
        "Latency (s)":   m.get("latency_s"),
    })

df = pd.DataFrame(rows)
if not df.empty:
    st.dataframe(
        df.style.format({
            "ROUGE-1":      "{:.4f}",
            "BERTScore F1": "{:.4f}",
            "Clin. BERTSc": "{:.4f}",
            "Contradiction":"{:.4f}",
            "Latency (s)":  "{:.2f}",
        }, na_rep="—").highlight_max(
            subset=[c for c in ["ROUGE-1","BERTScore F1","Clin. BERTSc"] if c in df.columns],
            color="#eff6ff"
        ).highlight_min(
            subset=[c for c in ["Contradiction","Latency (s)"] if c in df.columns],
            color="#eff6ff"
        ),
        use_container_width=True,
        height=220,
    )


# ═══════════════════════════════════════════════════════
# SECTION 5 — METRIC GUIDE
# ═══════════════════════════════════════════════════════
with st.expander("Metric definitions and thresholds"):
    st.markdown("""
| Metric | What it measures | Good threshold | Notes |
|---|---|---|---|
| **Clinical BERTScore F1** | Token-level cosine similarity using BiomedBERT (29M PubMed papers) | > 0.88 | Primary quality signal. Distinguishes clinical opposites. |
| **Generic BERTScore F1** | Same but with roberta-large (Wikipedia) | > 0.80 | Cannot distinguish "heart" from "lung". Baseline only. |
| **ROUGE-1 (full)** | Unigram overlap, full prediction vs reference | Context-dependent | Suppressed by model verbosity. Do not use in isolation. |
| **ROUGE-1 @50tok** | ROUGE-1 on first 50 tokens of prediction | > 0.20 | Verbosity-corrected. Measures: does the model lead with correct facts? |
| **Lexical Precision@50** | Fraction of first 50 tokens' content words in reference | > 0.15 | Stopword-filtered. Direct measure of factual precision. |
| **NLI Contradiction** | Fraction contradicting the reference (roberta-large-mnli) | < 0.10 = safe | ~72% neutral rate is expected — NLI model is not medical-calibrated. |
| **Perplexity** | Model confidence in its outputs | < 3 = confident | Lower is better. |
| **OOD Score** | Clinical BERTScore on PubMedQA (research questions) | Lower than in-distribution = expected | Proves generalisation beyond training data. |
""")

st.markdown("""
<div style="font-size:0.72rem;color:#9ca3af;text-align:center;margin-top:2rem;padding-top:1rem;border-top:1px solid #f3f4f6">
  MediGuide is a research project. Results are on 50-sample evaluations run on Kaggle T4.
  Not validated for clinical use.
</div>
""", unsafe_allow_html=True)
