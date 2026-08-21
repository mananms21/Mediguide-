"""
MEDIGUIDE — Main Streamlit Chat Application
Run: streamlit run app/app.py
"""

import json
import os
import sys
import time
from pathlib import Path

import streamlit as st

# ── Path setup ────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

# ── Page configuration ────────────────────────────────────────────
st.set_page_config(
    page_title="MEDIGUIDE — Medical AI",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Premium CSS ───────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

  /* ── Global ─────────────────────────────────── */
  html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

  .stApp {
    background: linear-gradient(135deg, #070d1a 0%, #0b1629 50%, #060e1c 100%);
    color: #e2e8f0;
  }

  /* ── Header ─────────────────────────────────── */
  .mg-header {
    text-align: center;
    padding: 2rem 0 1.5rem;
  }
  .mg-title {
    font-size: 2.6rem;
    font-weight: 700;
    background: linear-gradient(135deg, #00d4ff 0%, #7c4dff 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    letter-spacing: -0.5px;
    margin-bottom: 0.3rem;
  }
  .mg-subtitle {
    font-size: 0.95rem;
    color: #64748b;
    font-weight: 400;
  }

  /* ── Chat messages ───────────────────────────── */
  .chat-wrap { display: flex; flex-direction: column; gap: 1rem; margin-bottom: 1.5rem; }

  .msg-user {
    align-self: flex-end;
    max-width: 78%;
    background: linear-gradient(135deg, #1d4ed8 0%, #2563eb 100%);
    color: #fff;
    padding: 0.85rem 1.1rem;
    border-radius: 18px 18px 4px 18px;
    font-size: 0.92rem;
    line-height: 1.55;
    box-shadow: 0 4px 15px rgba(37,99,235,0.3);
  }
  .msg-user .label { font-size: 0.72rem; opacity: 0.75; margin-bottom: 0.35rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; }

  .msg-bot {
    align-self: flex-start;
    max-width: 82%;
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(0,212,255,0.15);
    color: #e2e8f0;
    padding: 0.9rem 1.1rem;
    border-radius: 18px 18px 18px 4px;
    font-size: 0.92rem;
    line-height: 1.6;
    box-shadow: 0 4px 20px rgba(0,0,0,0.3);
    backdrop-filter: blur(10px);
  }
  .msg-bot .label { font-size: 0.72rem; color: #00d4ff; margin-bottom: 0.35rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; }

  /* ── RAG context box ─────────────────────────── */
  .rag-context {
    background: rgba(124,77,255,0.08);
    border: 1px solid rgba(124,77,255,0.25);
    border-radius: 10px;
    padding: 0.75rem 1rem;
    font-size: 0.8rem;
    color: #a78bfa;
    margin-top: 0.5rem;
    line-height: 1.5;
  }
  .rag-label { font-weight: 600; color: #7c4dff; margin-bottom: 0.4rem; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.5px; }

  /* ── Sidebar ─────────────────────────────────── */
  [data-testid="stSidebar"] {
    background: rgba(10,14,26,0.95) !important;
    border-right: 1px solid rgba(255,255,255,0.06) !important;
  }
  .sidebar-section { margin-bottom: 1.5rem; }
  .sidebar-title { font-size: 0.72rem; font-weight: 600; color: #475569; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 0.6rem; }

  /* ── Badge chips ─────────────────────────────── */
  .badge {
    display: inline-block;
    padding: 0.2rem 0.6rem;
    border-radius: 100px;
    font-size: 0.7rem;
    font-weight: 600;
    margin-right: 0.3rem;
  }
  .badge-new  { background: rgba(0,230,118,0.15); color: #00e676; border: 1px solid rgba(0,230,118,0.3); }
  .badge-rag  { background: rgba(124,77,255,0.15); color: #7c4dff; border: 1px solid rgba(124,77,255,0.3); }
  .badge-gpu  { background: rgba(255,152,0,0.15); color: #ff9800; border: 1px solid rgba(255,152,0,0.3); }

  /* ── Disclaimer ──────────────────────────────── */
  .disclaimer {
    background: rgba(255,152,0,0.08);
    border: 1px solid rgba(255,152,0,0.2);
    border-radius: 10px;
    padding: 0.75rem 1rem;
    font-size: 0.78rem;
    color: #fbbf24;
    line-height: 1.5;
  }

  /* ── Latency badge ───────────────────────────── */
  .latency-badge {
    font-size: 0.72rem;
    color: #475569;
    margin-top: 0.4rem;
  }

  /* ── Empty state ─────────────────────────────── */
  .empty-state {
    text-align: center;
    padding: 4rem 2rem;
    color: #334155;
  }
  .empty-state .icon { font-size: 4rem; margin-bottom: 1rem; }
  .empty-state p { font-size: 1rem; }

  /* ── Example chips ───────────────────────────── */
  .example-label { font-size: 0.75rem; color: #475569; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 0.5rem; }

  /* ── Scrollbar ───────────────────────────────── */
  ::-webkit-scrollbar { width: 5px; }
  ::-webkit-scrollbar-track { background: transparent; }
  ::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.1); border-radius: 10px; }

  /* ── Streamlit overrides ─────────────────────── */
  .stTextArea textarea { background: rgba(255,255,255,0.04) !important; border: 1px solid rgba(0,212,255,0.2) !important; color: #e2e8f0 !important; border-radius: 12px !important; font-family: 'Inter', sans-serif !important; }
  .stTextArea textarea:focus { border-color: #00d4ff !important; box-shadow: 0 0 0 2px rgba(0,212,255,0.15) !important; }
  .stButton > button { border-radius: 10px !important; font-weight: 600 !important; transition: all 0.2s !important; }
  div[data-testid="stFormSubmitButton"] > button { background: linear-gradient(135deg, #1d4ed8, #7c4dff) !important; color: white !important; border: none !important; font-size: 0.9rem !important; padding: 0.6rem 1.5rem !important; }
  div[data-testid="stFormSubmitButton"] > button:hover { transform: translateY(-1px); box-shadow: 0 4px 15px rgba(37,99,235,0.4) !important; }
  .stSelectbox [data-baseweb="select"] { background: rgba(255,255,255,0.04) !important; border: 1px solid rgba(255,255,255,0.1) !important; }
  .stSlider [data-testid="stSlider"] label { color: #94a3b8 !important; }
  [data-testid="stMetricValue"] { color: #00d4ff !important; font-weight: 700 !important; }
</style>
""", unsafe_allow_html=True)


# ── Model registry ────────────────────────────────────────────────
MODELS = {
    "🆕 Phi-3 Mini QLoRA": {
        "model_id":   "Shriyanshml/phi3-mini-qlora-mediguide",
        "base":       "microsoft/Phi-3-mini-4k-instruct",
        "type":       "phi3",
        "desc":       "Phi-3 Mini 3.8B · QLoRA · 2,000 MedQuAD examples",
        "quantized":  True,
    },
    "Falcon-7B QLoRA": {
        "model_id":  "TestCase1/falcon-7b-qlora-chat-medical-bot",
        "base":      "tiiuae/falcon-7b",
        "type":      "falcon",
        "desc":      "Falcon 7B · QLoRA · best original baseline",
        "quantized": True,
    },
    "Falcon-7B LoRA": {
        "model_id":  "TestCase1/falcon-7b-lora-chat-medical-bot",
        "base":      "tiiuae/falcon-7b",
        "type":      "falcon",
        "desc":      "Falcon 7B · LoRA BF16 · fast inference",
        "quantized": False,
    },
}

SYSTEM_PROMPT = (
    "You are MEDIGUIDE, a knowledgeable medical assistant trained on "
    "authoritative NIH sources. Provide accurate, evidence-based answers "
    "to medical questions in a clear, empathetic tone. Always end your "
    "response with a brief disclaimer that this information is educational "
    "and patients should consult a qualified healthcare professional."
)


# ── Cached loaders ────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def load_model(model_key: str):
    """Load model + tokenizer. Cached per model key."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftConfig, PeftModel

    cfg_meta = MODELS[model_key]
    model_id = cfg_meta["model_id"]

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    ) if cfg_meta["quantized"] else None

        # Detect best local device: MPS (Apple Silicon) > CUDA > CPU
        import torch
        if torch.cuda.is_available():
            dev = "cuda"
            dtype_fallback = torch.float16
        elif torch.backends.mps.is_available():
            dev = "mps"
            dtype_fallback = torch.float16
        else:
            dev = "cpu"
            dtype_fallback = torch.float32

    try:
        peft_cfg = PeftConfig.from_pretrained(model_id)
        base_id  = peft_cfg.base_model_name_or_path

        kwargs = dict(
            return_dict=True,
            device_map={"" : dev},  # explicit map avoids needing accelerate on local
            # trust_remote_code removed — Phi-3 is natively supported in transformers 4.40+
        )
        if bnb and dev == "cuda":
            kwargs["quantization_config"] = bnb
        else:
            kwargs["dtype"] = dtype_fallback  # fp16 on MPS/CUDA, fp32 on CPU

        if cfg_meta["type"] == "phi3":
            kwargs["attn_implementation"] = "eager"

        model = AutoModelForCausalLM.from_pretrained(base_id, **kwargs)
        model = PeftModel.from_pretrained(model, model_id, device_map={"" : dev})
        tok   = AutoTokenizer.from_pretrained(base_id)
        tok.pad_token = tok.unk_token if cfg_meta["type"] == "phi3" else tok.eos_token
        return model, tok, None
    except Exception as e:
        return None, None, str(e)


@st.cache_resource(show_spinner=False)
def load_retriever():
    """Load RAG retriever (lazy)."""
    try:
        from rag.retriever import MedRAGRetriever
        r = MedRAGRetriever(index_dir=str(ROOT / "rag" / "index"))
        return r, None
    except Exception as e:
        return None, str(e)


# ── Inference ─────────────────────────────────────────────────────

def build_prompt(question: str, model_type: str, context: str = "") -> str:
    if model_type == "phi3":
        user_msg = f"{context}\n\nQuestion: {question}" if context else question
        return (
            f"<|system|>\n{SYSTEM_PROMPT}<|end|>\n"
            f"<|user|>\n{user_msg}<|end|>\n"
            f"<|assistant|>\n"
        )
    else:
        q = f"{context}\n\nQuestion: {question}" if context else question
        return f": {q}?\n: "


def generate(model, tokenizer, prompt: str, max_tokens: int, temperature: float, top_p: float) -> str:
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.2,
            no_repeat_ngram_size=3,
        )
    return tokenizer.decode(
        out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True
    ).strip()


# ── Sidebar ───────────────────────────────────────────────────────

with st.sidebar:
    st.markdown('<div class="mg-header"><span class="mg-title" style="font-size:1.4rem">🏥 MEDIGUIDE</span></div>', unsafe_allow_html=True)
    st.markdown("---")

    st.markdown('<div class="sidebar-title">Model</div>', unsafe_allow_html=True)
    selected_model = st.selectbox(
        "Select model",
        list(MODELS.keys()),
        index=0,
        label_visibility="collapsed",
    )
    meta = MODELS[selected_model]
    st.markdown(f'<p style="font-size:0.75rem;color:#475569;margin-top:0.3rem">{meta["desc"]}</p>', unsafe_allow_html=True)
    st.markdown("---")

    st.markdown('<div class="sidebar-title">RAG Settings</div>', unsafe_allow_html=True)
    use_rag = st.toggle("Enable RAG", value=True, help="Retrieve relevant MedQuAD references before generating")
    top_k   = st.slider("Retrieved references", 1, 5, 3, disabled=not use_rag)
    st.markdown("---")

    st.markdown('<div class="sidebar-title">Generation</div>', unsafe_allow_html=True)
    max_tokens  = st.slider("Max tokens", 50, 300, 150)
    temperature = st.slider("Temperature", 0.1, 1.0, 0.7, step=0.05)
    top_p       = st.slider("Top-p", 0.5, 1.0, 0.9, step=0.05)
    st.markdown("---")

    import torch
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem  = torch.cuda.get_device_properties(0).total_memory / 1e9
        st.markdown(f'<div class="badge badge-gpu">GPU</div><span style="font-size:0.78rem;color:#94a3b8">{gpu_name} ({gpu_mem:.0f}GB)</span>', unsafe_allow_html=True)
    else:
        st.warning("⚠️ No GPU — inference will be slow", icon="🐢")

    st.markdown("---")
    st.markdown("""
    <div class="disclaimer">
    ⚠️ <strong>Disclaimer</strong><br>
    For educational purposes only. Always consult a qualified healthcare professional for medical advice.
    </div>
    """, unsafe_allow_html=True)


# ── Header ────────────────────────────────────────────────────────

st.markdown("""
<div class="mg-header">
  <div class="mg-title">🏥 MEDIGUIDE</div>
  <div class="mg-subtitle">Medical AI · Fine-tuned Phi-3 Mini · RAG over 16,000 NIH Q&amp;A pairs</div>
</div>
""", unsafe_allow_html=True)


# ── Session state ─────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "current_model" not in st.session_state:
    st.session_state.current_model = selected_model

if st.session_state.current_model != selected_model:
    st.session_state.messages = []
    st.session_state.current_model = selected_model


# ── Load model ────────────────────────────────────────────────────
with st.spinner(f"Loading {selected_model}…"):
    model, tokenizer, load_err = load_model(selected_model)

retriever, rag_err = load_retriever()

if load_err:
    st.error(f"⚠️ Model unavailable: `{load_err}`\n\nMake sure you have run the Kaggle training and the model is pushed to HF Hub.")
    model_ready = False
else:
    model_ready = True

rag_ready = retriever is not None and retriever.is_available


# ── Status bar ───────────────────────────────────────────────────
col_a, col_b, col_c = st.columns(3)
with col_a:
    st.metric("Model", "✅ Loaded" if model_ready else "❌ Unavailable")
with col_b:
    st.metric("RAG Index", f"✅ {retriever.num_documents:,} docs" if rag_ready else "⚠️ Not found")
with col_c:
    rag_status = ("ON · Retrieving" if (use_rag and rag_ready) else "OFF")
    st.metric("RAG Status", rag_status)


# ── Chat display ──────────────────────────────────────────────────
st.markdown('<div class="chat-wrap">', unsafe_allow_html=True)

if not st.session_state.messages:
    st.markdown("""
    <div class="empty-state">
      <div class="icon">💬</div>
      <p>Ask any medical question below.<br><span style="color:#475569;font-size:0.85rem">RAG will retrieve relevant references from 16k+ NIH-sourced Q&amp;A pairs.</span></p>
    </div>
    """, unsafe_allow_html=True)

for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(
            f'<div class="msg-user"><div class="label">You</div>{msg["content"]}</div>',
            unsafe_allow_html=True,
        )
    else:
        context_html = ""
        if msg.get("context"):
            context_html = (
                f'<div class="rag-context">'
                f'<div class="rag-label">📚 RAG Context Retrieved</div>'
                f'{msg["context"][:500]}…</div>'
            )
        st.markdown(
            f'<div class="msg-bot">'
            f'<div class="label">🏥 MEDIGUIDE</div>'
            f'{msg["content"]}'
            f'{context_html}'
            f'<div class="latency-badge">⏱ {msg.get("latency", "")} · {msg.get("model", "")}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

st.markdown('</div>', unsafe_allow_html=True)


# ── Input form ───────────────────────────────────────────────────
with st.form("chat_form", clear_on_submit=True):
    user_input = st.text_area(
        "Your question",
        placeholder="e.g. What are the early signs of Type 2 diabetes?",
        height=90,
        label_visibility="collapsed",
    )
    submitted = st.form_submit_button("Send →", use_container_width=True)

if submitted and user_input.strip() and model_ready:
    question = user_input.strip()
    st.session_state.messages.append({"role": "user", "content": question})

    # RAG context
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
        "context": context if context else "",
        "latency": f"{latency:.1f}s",
        "model":   selected_model,
    })
    st.rerun()


# ── Example questions ─────────────────────────────────────────────
st.markdown('<div class="example-label" style="margin-top:1rem">💡 Try these</div>', unsafe_allow_html=True)
ex_cols = st.columns(3)
examples = [
    "What is Type 2 diabetes and how is it managed?",
    "What are the symptoms of hypertension?",
    "How can I reduce my risk of heart disease?",
]
for col, ex in zip(ex_cols, examples):
    with col:
        if st.button(ex, use_container_width=True):
            st.session_state.messages.append({"role": "user", "content": ex})
            st.rerun()

# Clear
if st.session_state.messages:
    if st.button("🗑️ Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
