"""
MEDIGUIDE — Chat Application
Run: streamlit run app/app.py
"""

import os
import sys
import time
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

# ── Page config ───────────────────────────────────────────────────
st.set_page_config(
    page_title="MediGuide",
    page_icon="⚕",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS — clean, light, professional ─────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

  html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
  }

  /* ── Background ─────────────────────────────────────────────── */
  .stApp { background: #f7f8fa; }

  /* ── Hide default Streamlit chrome ──────────────────────────── */
  #MainMenu, footer, header { visibility: hidden; }

  /* ── Top bar ─────────────────────────────────────────────────── */
  .topbar {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 1.1rem 0 1.4rem;
    border-bottom: 1px solid #e5e7eb;
    margin-bottom: 1.5rem;
  }
  .topbar-logo {
    width: 38px; height: 38px;
    background: #1d4ed8;
    border-radius: 10px;
    display: flex; align-items: center; justify-content: center;
    font-size: 1.2rem; color: white; flex-shrink: 0;
  }
  .topbar-name {
    font-size: 1.15rem; font-weight: 700;
    color: #111827; letter-spacing: -0.3px;
  }
  .topbar-sub {
    font-size: 0.78rem; color: #6b7280;
    font-weight: 400; margin-top: 1px;
  }
  .topbar-spacer { flex: 1; }
  .topbar-badge {
    font-size: 0.7rem; font-weight: 600;
    background: #eff6ff; color: #1d4ed8;
    border: 1px solid #bfdbfe;
    padding: 3px 10px; border-radius: 20px;
  }

  /* ── Status row ──────────────────────────────────────────────── */
  .status-row {
    display: flex; gap: 10px; flex-wrap: wrap;
    margin-bottom: 1.25rem;
  }
  .status-pill {
    display: flex; align-items: center; gap: 6px;
    background: #fff; border: 1px solid #e5e7eb;
    border-radius: 8px; padding: 5px 12px;
    font-size: 0.75rem; color: #374151; font-weight: 500;
  }
  .status-dot { width:7px;height:7px;border-radius:50%;flex-shrink:0; }
  .dot-green  { background:#22c55e; }
  .dot-blue   { background:#3b82f6; }
  .dot-amber  { background:#f59e0b; }
  .dot-gray   { background:#9ca3af; }

  /* ── Chat container ──────────────────────────────────────────── */
  .chat-area {
    background: #fff;
    border: 1px solid #e5e7eb;
    border-radius: 12px;
    padding: 1.25rem 1.5rem;
    min-height: 320px;
    margin-bottom: 1rem;
  }

  /* ── Empty state ─────────────────────────────────────────────── */
  .empty-state {
    text-align: center;
    padding: 3.5rem 1.5rem;
  }
  .empty-icon { font-size: 2.4rem; margin-bottom: 1rem; }
  .empty-title { font-size: 1rem; font-weight: 600; color: #111827; margin-bottom: 0.4rem; }
  .empty-sub { font-size: 0.83rem; color: #6b7280; line-height: 1.6; }

  /* ── Messages ────────────────────────────────────────────────── */
  .msg + .msg { margin-top: 1.1rem; }

  .msg-user {
    display: flex; justify-content: flex-end;
  }
  .msg-user-bubble {
    max-width: 72%;
    background: #1d4ed8;
    color: #fff;
    padding: 0.7rem 1rem;
    border-radius: 16px 16px 4px 16px;
    font-size: 0.875rem;
    line-height: 1.55;
  }

  .msg-bot { display: flex; gap: 10px; }
  .msg-bot-avatar {
    width: 30px; height: 30px; flex-shrink: 0;
    background: #eff6ff;
    border: 1px solid #bfdbfe;
    border-radius: 8px;
    display: flex; align-items: center; justify-content: center;
    font-size: 0.9rem; margin-top: 2px;
  }
  .msg-bot-bubble {
    flex: 1;
    background: #f9fafb;
    border: 1px solid #f3f4f6;
    border-radius: 4px 16px 16px 16px;
    padding: 0.75rem 1rem;
    font-size: 0.875rem;
    line-height: 1.65;
    color: #111827;
  }
  .msg-bot-name {
    font-size: 0.7rem; font-weight: 600;
    color: #1d4ed8; text-transform: uppercase;
    letter-spacing: 0.4px; margin-bottom: 0.35rem;
  }
  .msg-meta {
    font-size: 0.68rem; color: #9ca3af;
    margin-top: 0.45rem;
  }

  /* ── RAG reference box ───────────────────────────────────────── */
  .rag-box {
    background: #fafafa;
    border: 1px solid #e5e7eb;
    border-left: 3px solid #3b82f6;
    border-radius: 0 6px 6px 0;
    padding: 0.6rem 0.8rem;
    margin-top: 0.6rem;
    font-size: 0.75rem;
    color: #4b5563;
    line-height: 1.5;
  }
  .rag-box-title {
    font-size: 0.68rem; font-weight: 600;
    color: #3b82f6; text-transform: uppercase;
    letter-spacing: 0.4px; margin-bottom: 0.3rem;
  }

  /* ── Input area ──────────────────────────────────────────────── */
  .input-area {
    background: #fff;
    border: 1px solid #d1d5db;
    border-radius: 12px;
    padding: 0.1rem;
    margin-bottom: 0.8rem;
    transition: border-color 0.15s;
  }
  .input-area:focus-within { border-color: #3b82f6; }

  .stTextArea textarea {
    border: none !important;
    box-shadow: none !important;
    background: transparent !important;
    color: #111827 !important;
    font-size: 0.9rem !important;
    font-family: 'Inter', sans-serif !important;
    resize: none !important;
  }

  /* ── Buttons ─────────────────────────────────────────────────── */
  div[data-testid="stFormSubmitButton"] > button {
    background: #1d4ed8 !important;
    color: #fff !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-size: 0.875rem !important;
    padding: 0.55rem 1.5rem !important;
    transition: background 0.15s !important;
  }
  div[data-testid="stFormSubmitButton"] > button:hover {
    background: #1e40af !important;
  }

  .stButton > button {
    background: #fff !important;
    color: #374151 !important;
    border: 1px solid #e5e7eb !important;
    border-radius: 8px !important;
    font-size: 0.82rem !important;
    font-weight: 500 !important;
    transition: all 0.15s !important;
  }
  .stButton > button:hover {
    border-color: #9ca3af !important;
    background: #f9fafb !important;
  }

  /* ── Example chips ───────────────────────────────────────────── */
  .chip-label {
    font-size: 0.72rem; font-weight: 600;
    color: #9ca3af; text-transform: uppercase;
    letter-spacing: 0.5px; margin-bottom: 0.5rem;
  }

  /* ── Disclaimer ──────────────────────────────────────────────── */
  .disclaimer-bar {
    background: #fffbeb;
    border: 1px solid #fde68a;
    border-radius: 8px;
    padding: 0.6rem 0.9rem;
    font-size: 0.75rem;
    color: #92400e;
    line-height: 1.5;
    margin-top: 0.8rem;
  }

  /* ── Sidebar ─────────────────────────────────────────────────── */
  [data-testid="stSidebar"] {
    background: #fff !important;
    border-right: 1px solid #e5e7eb !important;
  }
  [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
    color: #374151;
  }
  .sb-section { margin-bottom: 1.5rem; }
  .sb-label {
    font-size: 0.68rem; font-weight: 600;
    color: #9ca3af; text-transform: uppercase;
    letter-spacing: 0.8px; margin-bottom: 0.6rem;
    display: block;
  }
  .sb-divider { border: none; border-top: 1px solid #f3f4f6; margin: 1.1rem 0; }

  /* ── Streamlit widget overrides ──────────────────────────────── */
  .stSelectbox [data-baseweb="select"] div {
    background: #f9fafb !important;
    border-color: #e5e7eb !important;
    color: #111827 !important;
  }
  [data-testid="stSlider"] label { color: #374151 !important; font-size: 0.82rem !important; }
  [data-testid="stMetricValue"] { color: #111827 !important; font-weight: 600 !important; }
  [data-testid="stMetricLabel"] { color: #6b7280 !important; font-size: 0.8rem !important; }
  .stToggle label { color: #374151 !important; font-size: 0.85rem !important; }
  .stSpinner { color: #1d4ed8 !important; }
</style>
""", unsafe_allow_html=True)


# ── Model registry ────────────────────────────────────────────────
MODELS = {
    "Phi-3 Mini QLoRA (Recommended)": {
        "model_id":  "Shriyanshml/phi3-mini-qlora-mediguide",
        "base":      "microsoft/Phi-3-mini-4k-instruct",
        "type":      "phi3",
        "desc":      "3.8B · QLoRA · 2,000 NIH MedQuAD samples · Clinical BERTScore 0.94",
        "quantized": True,
    },
    "Falcon-7B QLoRA": {
        "model_id":  "TestCase1/falcon-7b-qlora-chat-medical-bot",
        "base":      "tiiuae/falcon-7b",
        "type":      "falcon",
        "desc":      "7B · QLoRA · earlier baseline",
        "quantized": True,
    },
    "Falcon-7B LoRA": {
        "model_id":  "TestCase1/falcon-7b-lora-chat-medical-bot",
        "base":      "tiiuae/falcon-7b",
        "type":      "falcon",
        "desc":      "7B · LoRA BF16 · fastest inference",
        "quantized": False,
    },
}

SYSTEM_PROMPT = (
    "You are MediGuide, a medical information assistant trained on "
    "authoritative NIH sources. Provide accurate, evidence-based answers "
    "to medical questions in clear, plain language. Always end with a brief "
    "note that this information is educational and the user should consult "
    "a qualified healthcare professional for personal medical decisions."
)


# ── Cached loaders ────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model(model_key: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftConfig, PeftModel

    cfg = MODELS[model_key]

    if torch.cuda.is_available():
        dev, dtype = "cuda", torch.float16
    elif torch.backends.mps.is_available():
        dev, dtype = "mps", torch.float16
    else:
        dev, dtype = "cpu", torch.float32

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=dtype,
    ) if cfg["quantized"] and dev == "cuda" else None

    try:
        peft_cfg = PeftConfig.from_pretrained(cfg["model_id"])
        base_id  = peft_cfg.base_model_name_or_path
        kwargs   = dict(device_map={"": dev}, return_dict=True)
        if bnb:
            kwargs["quantization_config"] = bnb
        else:
            kwargs["torch_dtype"] = dtype
        if cfg["type"] == "phi3":
            kwargs["attn_implementation"] = "eager"
        model = AutoModelForCausalLM.from_pretrained(base_id, **kwargs)
        model = PeftModel.from_pretrained(model, cfg["model_id"], device_map={"": dev})
        tok   = AutoTokenizer.from_pretrained(base_id)
        tok.pad_token = tok.unk_token if cfg["type"] == "phi3" else tok.eos_token
        return model, tok, None
    except Exception as e:
        return None, None, str(e)


@st.cache_resource(show_spinner=False)
def load_retriever():
    try:
        from rag.retriever import MedRAGRetriever
        r = MedRAGRetriever(index_dir=str(ROOT / "rag" / "index"))
        return r, None
    except Exception as e:
        return None, str(e)


# ── Inference ─────────────────────────────────────────────────────
def build_prompt(question: str, model_type: str, context: str = "") -> str:
    if model_type == "phi3":
        user_msg = f"[Context from NIH MedQuAD]\n{context}\n\nQuestion: {question}" if context else question
        return (
            f"<|system|>\n{SYSTEM_PROMPT}<|end|>\n"
            f"<|user|>\n{user_msg}<|end|>\n"
            f"<|assistant|>\n"
        )
    q = f"{context}\n\nQuestion: {question}" if context else question
    return f": {q}?\n: "


def generate(model, tokenizer, prompt, max_tokens, temperature, top_p) -> str:
    import torch
    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.1,
        )
    return tokenizer.decode(
        out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True
    ).strip()


# ── Sidebar ───────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div style="font-size:1.05rem;font-weight:700;color:#111827;padding:0.5rem 0 1rem">⚕ MediGuide</div>', unsafe_allow_html=True)
    st.markdown('<hr class="sb-divider">', unsafe_allow_html=True)

    st.markdown('<span class="sb-label">Model</span>', unsafe_allow_html=True)
    selected_model = st.selectbox("Model", list(MODELS.keys()), index=0, label_visibility="collapsed")
    meta = MODELS[selected_model]
    st.caption(meta["desc"])

    st.markdown('<hr class="sb-divider">', unsafe_allow_html=True)
    st.markdown('<span class="sb-label">Retrieval</span>', unsafe_allow_html=True)
    use_rag = st.toggle("Enable RAG", value=True, help="Retrieve relevant NIH passages before generating")
    top_k   = st.slider("References retrieved", 1, 5, 3, disabled=not use_rag)

    st.markdown('<hr class="sb-divider">', unsafe_allow_html=True)
    st.markdown('<span class="sb-label">Generation</span>', unsafe_allow_html=True)
    max_tokens  = st.slider("Max tokens", 50, 400, 200)
    temperature = st.slider("Temperature", 0.1, 1.0, 0.7, step=0.05)
    top_p       = st.slider("Top-p", 0.5, 1.0, 0.9, step=0.05)

    st.markdown('<hr class="sb-divider">', unsafe_allow_html=True)
    import torch
    if torch.cuda.is_available():
        gname = torch.cuda.get_device_name(0)
        gmem  = torch.cuda.get_device_properties(0).total_memory / 1e9
        st.caption(f"🟢 GPU · {gname} ({gmem:.0f} GB)")
    else:
        st.caption("🟡 Running on CPU — responses will be slow")

    st.markdown('<hr class="sb-divider">', unsafe_allow_html=True)
    st.markdown(
        '<p style="font-size:0.72rem;color:#6b7280;line-height:1.6">'
        '⚕ MediGuide is a research project trained on NIH MedQuAD. '
        'It is not a medical device and must not replace professional clinical advice.'
        '</p>',
        unsafe_allow_html=True
    )


# ── Session state ─────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "current_model" not in st.session_state:
    st.session_state.current_model = selected_model
if st.session_state.current_model != selected_model:
    st.session_state.messages = []
    st.session_state.current_model = selected_model


# ── Load model ────────────────────────────────────────────────────
with st.spinner("Loading model…"):
    model, tokenizer, load_err = load_model(selected_model)
retriever, rag_err = load_retriever()

model_ready = load_err is None
rag_ready   = retriever is not None and getattr(retriever, "is_available", False)


# ── Top bar ───────────────────────────────────────────────────────
st.markdown("""
<div class="topbar">
  <div class="topbar-logo">⚕</div>
  <div>
    <div class="topbar-name">MediGuide</div>
    <div class="topbar-sub">NIH MedQuAD · Phi-3 Mini QLoRA · RAG</div>
  </div>
  <div class="topbar-spacer"></div>
  <div class="topbar-badge">Research Preview</div>
</div>
""", unsafe_allow_html=True)


# ── Status row ────────────────────────────────────────────────────
model_dot  = "dot-green" if model_ready else "dot-amber"
model_txt  = "Model loaded" if model_ready else "Model unavailable"
rag_dot    = "dot-blue" if rag_ready else "dot-gray"
rag_txt    = f"RAG · {retriever.num_documents:,} passages" if rag_ready else "RAG index not found"
mode_txt   = "RAG enabled" if (use_rag and rag_ready) else "Direct generation"
mode_dot   = "dot-blue" if (use_rag and rag_ready) else "dot-gray"

st.markdown(f"""
<div class="status-row">
  <div class="status-pill"><div class="status-dot {model_dot}"></div>{model_txt}</div>
  <div class="status-pill"><div class="status-dot {rag_dot}"></div>{rag_txt}</div>
  <div class="status-pill"><div class="status-dot {mode_dot}"></div>{mode_txt}</div>
</div>
""", unsafe_allow_html=True)


# ── Chat area ─────────────────────────────────────────────────────
st.markdown('<div class="chat-area">', unsafe_allow_html=True)

if not st.session_state.messages:
    st.markdown("""
<div class="empty-state">
  <div class="empty-icon">💬</div>
  <div class="empty-title">Ask a medical question</div>
  <div class="empty-sub">
    MediGuide retrieves relevant passages from 14,782 NIH MedQuAD Q&amp;A pairs<br>
    and generates evidence-based answers using a fine-tuned Phi-3 Mini model.
  </div>
</div>
""", unsafe_allow_html=True)

for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(
            f'<div class="msg msg-user"><div class="msg-user-bubble">{msg["content"]}</div></div>',
            unsafe_allow_html=True,
        )
    else:
        context_html = ""
        if msg.get("context"):
            snippet = msg["context"][:400].replace("<", "&lt;").replace(">", "&gt;")
            context_html = (
                f'<div class="rag-box">'
                f'<div class="rag-box-title">📚 Retrieved context</div>'
                f'{snippet}…</div>'
            )
        st.markdown(
            f'<div class="msg msg-bot">'
            f'<div class="msg-bot-avatar">⚕</div>'
            f'<div class="msg-bot-bubble">'
            f'<div class="msg-bot-name">MediGuide</div>'
            f'{msg["content"]}'
            f'{context_html}'
            f'<div class="msg-meta">⏱ {msg.get("latency","")} &nbsp;·&nbsp; {msg.get("model","")}</div>'
            f'</div></div>',
            unsafe_allow_html=True,
        )

st.markdown('</div>', unsafe_allow_html=True)


# ── Input ─────────────────────────────────────────────────────────
with st.form("chat_form", clear_on_submit=True):
    st.markdown('<div class="input-area">', unsafe_allow_html=True)
    user_input = st.text_area(
        "Question",
        placeholder="e.g. What are the early warning signs of Type 2 diabetes?",
        height=80,
        label_visibility="collapsed",
    )
    st.markdown('</div>', unsafe_allow_html=True)
    col1, col2 = st.columns([5, 1])
    with col1:
        submitted = st.form_submit_button("Send message", use_container_width=True)
    with col2:
        st.form_submit_button("Clear", use_container_width=True)

if submitted and user_input.strip():
    if not model_ready:
        st.error("Model is not available. Check your HF credentials and network connection.")
    else:
        question = user_input.strip()
        st.session_state.messages.append({"role": "user", "content": question})

        context = ""
        if use_rag and rag_ready:
            context = retriever.format_context(question, top_k)

        prompt = build_prompt(question, meta["type"], context)

        with st.spinner("Generating response…"):
            t0       = time.time()
            response = generate(model, tokenizer, prompt, max_tokens, temperature, top_p)
            latency  = time.time() - t0

        st.session_state.messages.append({
            "role":    "bot",
            "content": response,
            "context": context,
            "latency": f"{latency:.1f}s",
            "model":   selected_model,
        })
        st.rerun()


# ── Example questions ─────────────────────────────────────────────
if not st.session_state.messages:
    st.markdown('<div class="chip-label">Suggested questions</div>', unsafe_allow_html=True)
    examples = [
        "What is Type 2 diabetes and how is it managed?",
        "What are the symptoms of hypertension?",
        "How can I reduce my risk of heart disease?",
    ]
    cols = st.columns(3)
    for col, ex in zip(cols, examples):
        with col:
            if st.button(ex, use_container_width=True):
                st.session_state.messages.append({"role": "user", "content": ex})
                st.rerun()


# ── Disclaimer ────────────────────────────────────────────────────
st.markdown("""
<div class="disclaimer-bar">
  <strong>Disclaimer:</strong> MediGuide is a research project and is not a medical device.
  Information provided is for educational purposes only. Always consult a qualified
  healthcare professional for medical advice, diagnosis, or treatment.
</div>
""", unsafe_allow_html=True)
