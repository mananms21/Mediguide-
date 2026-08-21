"""
MEDIGUIDE — HuggingFace Spaces Gradio App
Deploy this to https://huggingface.co/spaces/Shriyanshml/mediguide

Features:
  • Phi-3 Mini QLoRA fine-tuned model loaded via PEFT
  • FAISS RAG over 16,000+ MedQuAD entries
  • ZeroGPU support for free GPU inference
  • Medical disclaimer on every response
"""

import os
import pickle
import sys
import time
from pathlib import Path

import gradio as gr
import numpy as np
import torch

# ── HF Spaces ZeroGPU ─────────────────────────────────────────────
try:
    import spaces
    HAS_ZEROGPU = True
except ImportError:
    HAS_ZEROGPU = False
    # Create a no-op decorator if not on HF Spaces
    class spaces:
        @staticmethod
        def GPU(fn):
            return fn

MODEL_ID     = "Shriyanshml/phi3-mini-qlora-mediguide"
RAG_DATASET  = "Shriyanshml/mediguide-rag-index"
INDEX_DIR    = Path("rag_cache")
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"

SYSTEM_PROMPT = (
    "You are MEDIGUIDE, a knowledgeable medical assistant trained on "
    "authoritative NIH sources. Provide accurate, evidence-based answers "
    "to medical questions in a clear, empathetic tone. Always conclude with "
    "a brief disclaimer that this information is educational and that patients "
    "should consult a qualified healthcare professional for personal medical advice."
)

DISCLAIMER = (
    "\n\n---\n⚠️ *This response is for educational purposes only. "
    "Please consult a qualified healthcare professional for personal medical advice.*"
)

# ── Globals (loaded once) ─────────────────────────────────────────
_model     = None
_tokenizer = None
_retriever = None


def _load_model():
    global _model, _tokenizer
    if _model is not None:
        return

    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftConfig, PeftModel

    print(f"Loading {MODEL_ID}…")
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    cfg   = PeftConfig.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        cfg.base_model_name_or_path,
        quantization_config=bnb,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="eager",
    )
    model       = PeftModel.from_pretrained(model, MODEL_ID)
    model.eval()
    tok         = AutoTokenizer.from_pretrained(cfg.base_model_name_or_path, trust_remote_code=True)
    tok.pad_token = tok.unk_token
    _model, _tokenizer = model, tok
    print("✅ Model loaded")


def _load_retriever():
    global _retriever
    if _retriever is not None:
        return

    import faiss
    from huggingface_hub import hf_hub_download
    from sentence_transformers import SentenceTransformer

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    index_path = INDEX_DIR / "faiss_index.bin"
    docs_path  = INDEX_DIR / "medquad_docs.pkl"

    if not index_path.exists():
        print(f"Downloading RAG index from {RAG_DATASET}…")
        for fname in ["faiss_index.bin", "medquad_docs.pkl"]:
            hf_hub_download(
                repo_id=RAG_DATASET,
                filename=fname,
                repo_type="dataset",
                local_dir=str(INDEX_DIR),
                local_dir_use_symlinks=False,
            )
        print("✅ RAG index downloaded")

    index    = faiss.read_index(str(index_path))
    with open(docs_path, "rb") as f:
        docs = pickle.load(f)
    encoder  = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    _retriever = {"index": index, "docs": docs, "encoder": encoder}
    print(f"✅ RAG index loaded: {index.ntotal:,} vectors")


def retrieve(query: str, top_k: int = 3) -> str:
    if _retriever is None:
        return ""
    import faiss
    embedding = _retriever["encoder"].encode([query], normalize_embeddings=True).astype(np.float32)
    faiss.normalize_L2(embedding)
    scores, indices = _retriever["index"].search(embedding, top_k)
    parts = ["**Relevant medical references:**\n"]
    for i, (score, idx) in enumerate(zip(scores[0], indices[0]), 1):
        if idx >= 0:
            doc     = _retriever["docs"][idx]
            preview = doc["answer"][:300] + ("…" if len(doc["answer"]) > 300 else "")
            parts.append(f"**[Ref {i}]** *{doc.get('focus_area', 'General')}*\n**Q:** {doc['question']}\n**A:** {preview}")
    return "\n\n".join(parts)


# ── Inference ─────────────────────────────────────────────────────

@spaces.GPU
def generate_answer(
    question: str,
    history: list,
    use_rag: bool,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
):
    if not question.strip():
        yield history, "", ""

    _load_model()

    # RAG context
    context_md = ""
    if use_rag:
        try:
            _load_retriever()
            context_md = retrieve(question.strip())
        except Exception as e:
            context_md = f"⚠️ RAG unavailable: {e}"

    # Build prompt
    user_msg = question.strip()
    if use_rag and context_md and "unavailable" not in context_md:
        # Strip markdown from context for model input
        plain_ctx = context_md.replace("**", "").replace("*", "")
        user_msg  = f"{plain_ctx}\n\nBased on the above references, answer:\n{question.strip()}"

    prompt = (
        f"<|system|>\n{SYSTEM_PROMPT}<|end|>\n"
        f"<|user|>\n{user_msg}<|end|>\n"
        f"<|assistant|>\n"
    )

    inputs  = _tokenizer(prompt, return_tensors="pt").to(DEVICE)

    t0 = time.time()
    with torch.no_grad():
        output = _model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=_tokenizer.eos_token_id,
            repetition_penalty=1.2,
            no_repeat_ngram_size=3,
        )
    latency = time.time() - t0

    response = _tokenizer.decode(
        output[0][inputs.input_ids.shape[1]:], skip_special_tokens=True
    ).strip()

    # Ensure disclaimer
    if "consult" not in response.lower():
        response += DISCLAIMER

    latency_str = f"⏱ Generated in {latency:.1f}s | Model: {MODEL_ID}"
    history = history + [[question, response]]
    yield history, context_md or "*(RAG disabled or no relevant references found)*", latency_str


def clear_chat():
    return [], "", ""


# ── Gradio UI ─────────────────────────────────────────────────────
theme = gr.themes.Soft(
    primary_hue=gr.themes.colors.cyan,
    secondary_hue=gr.themes.colors.violet,
    neutral_hue=gr.themes.colors.slate,
    font=[gr.themes.GoogleFont("Inter"), "sans-serif"],
).set(
    body_background_fill="linear-gradient(135deg, #070d1a 0%, #0b1629 50%, #060e1c 100%)",
    body_text_color="#e2e8f0",
    block_background_fill="rgba(255,255,255,0.04)",
    block_border_color="rgba(0,212,255,0.15)",
    block_title_text_color="#00d4ff",
    button_primary_background_fill="linear-gradient(135deg, #1d4ed8, #7c4dff)",
    button_primary_text_color="#ffffff",
    button_secondary_background_fill="rgba(255,255,255,0.06)",
    button_secondary_text_color="#e2e8f0",
    input_background_fill="rgba(255,255,255,0.04)",
    input_border_color="rgba(0,212,255,0.2)",
    slider_color="#00d4ff",
    checkbox_background_color_selected="#7c4dff",
)

EXAMPLES = [
    "What is Type 2 diabetes and how is it managed?",
    "What are the early warning signs of a heart attack?",
    "What causes hypertension and how can it be controlled?",
    "What is the recommended treatment for seasonal allergies?",
    "How does the COVID-19 vaccine work?",
    "What are the symptoms of anemia and how is it diagnosed?",
]

with gr.Blocks(theme=theme, title="MEDIGUIDE — Medical AI Chatbot") as demo:

    gr.HTML("""
    <div style="text-align:center;padding:1.5rem 0 1rem">
      <h1 style="font-size:2.2rem;font-weight:700;
                 background:linear-gradient(135deg,#00d4ff,#7c4dff);
                 -webkit-background-clip:text;-webkit-text-fill-color:transparent;
                 background-clip:text;margin-bottom:0.3rem">
        🏥 MEDIGUIDE
      </h1>
      <p style="color:#64748b;font-size:0.95rem">
        Fine-tuned Phi-3 Mini · QLoRA · RAG over 16,000+ NIH Q&amp;A pairs
      </p>
      <div style="display:flex;gap:0.5rem;justify-content:center;margin-top:0.7rem;flex-wrap:wrap">
        <span style="background:rgba(0,212,255,0.12);color:#00d4ff;border:1px solid rgba(0,212,255,0.3);padding:0.2rem 0.7rem;border-radius:100px;font-size:0.72rem;font-weight:600">Phi-3 Mini 3.8B</span>
        <span style="background:rgba(124,77,255,0.12);color:#a78bfa;border:1px solid rgba(124,77,255,0.3);padding:0.2rem 0.7rem;border-radius:100px;font-size:0.72rem;font-weight:600">QLoRA 4-bit</span>
        <span style="background:rgba(0,230,118,0.12);color:#00e676;border:1px solid rgba(0,230,118,0.3);padding:0.2rem 0.7rem;border-radius:100px;font-size:0.72rem;font-weight:600">FAISS RAG</span>
        <span style="background:rgba(255,152,0,0.12);color:#ff9800;border:1px solid rgba(255,152,0,0.3);padding:0.2rem 0.7rem;border-radius:100px;font-size:0.72rem;font-weight:600">NIH Sources</span>
      </div>
    </div>
    """)

    with gr.Row():
        # ── Main chat column ──────────────────────────────────────
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(
                label="Conversation",
                height=480,
                bubble_full_width=False,
                avatar_images=(None, "https://em-content.zobj.net/source/apple/354/hospital_1f3e5.png"),
                show_copy_button=True,
            )

            with gr.Row():
                question_box = gr.Textbox(
                    placeholder="Ask a medical question, e.g. What is Type 2 diabetes?",
                    label="Your question",
                    lines=2,
                    show_label=False,
                    scale=5,
                )
                send_btn = gr.Button("Send →", variant="primary", scale=1, min_width=80)

            latency_md = gr.Markdown("", elem_classes=["latency"])

            gr.Examples(
                examples=EXAMPLES,
                inputs=question_box,
                label="💡 Example questions",
            )

            clear_btn = gr.Button("🗑️ Clear Conversation", variant="secondary", size="sm")

        # ── Settings + RAG panel ──────────────────────────────────
        with gr.Column(scale=2):
            with gr.Group():
                gr.Markdown("### ⚙️ Settings")
                use_rag     = gr.Checkbox(value=True, label="Enable RAG (recommended)")
                max_tokens  = gr.Slider(50, 300, value=150, step=10, label="Max new tokens")
                temperature = gr.Slider(0.1, 1.0, value=0.7, step=0.05, label="Temperature")
                top_p       = gr.Slider(0.5, 1.0, value=0.9, step=0.05, label="Top-p")

            with gr.Group():
                gr.Markdown("### 📚 Retrieved Context")
                context_box = gr.Markdown(
                    "*Enable RAG and send a question to see retrieved references.*",
                    label="RAG Context",
                )

            gr.HTML("""
            <div style="background:rgba(255,152,0,0.08);border:1px solid rgba(255,152,0,0.2);
                        border-radius:10px;padding:0.75rem 1rem;font-size:0.78rem;color:#fbbf24;
                        line-height:1.5;margin-top:1rem">
              ⚠️ <strong>Medical Disclaimer</strong><br>
              This chatbot is for informational &amp; educational purposes only.
              Always consult a qualified healthcare professional for diagnosis and treatment.
            </div>
            """)

    # ── Event bindings ────────────────────────────────────────────
    submit_args  = [question_box, chatbot, use_rag, max_tokens, temperature, top_p]
    submit_outs  = [chatbot, context_box, latency_md]

    send_btn.click(
        generate_answer, inputs=submit_args, outputs=submit_outs
    ).then(lambda: "", None, question_box)

    question_box.submit(
        generate_answer, inputs=submit_args, outputs=submit_outs
    ).then(lambda: "", None, question_box)

    clear_btn.click(clear_chat, outputs=[chatbot, context_box, latency_md])

    # ── Footer ────────────────────────────────────────────────────
    gr.HTML("""
    <div style="text-align:center;margin-top:2rem;padding:1rem;
                border-top:1px solid rgba(255,255,255,0.06);
                font-size:0.75rem;color:#334155">
      MEDIGUIDE · Phi-3 Mini QLoRA · Fine-tuned on MedQuAD (NIH) ·
      <a href="https://huggingface.co/Shriyanshml/phi3-mini-qlora-mediguide"
         style="color:#00d4ff;text-decoration:none" target="_blank">Model on HF Hub</a> ·
      <a href="https://github.com/mananms21/Mediguide-"
         style="color:#7c4dff;text-decoration:none" target="_blank">GitHub</a>
    </div>
    """)


if __name__ == "__main__":
    demo.launch()
