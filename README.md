# MEDIGUIDE 🩺

**A domain-adapted medical question-answering system built on Phi-3 Mini, fine-tuned with QLoRA and augmented with Retrieval-Augmented Generation (RAG) over NIH MedQuAD.**

[![Live Demo](https://img.shields.io/badge/🤗%20Space-Shriyanshml%2Fmediguide-orange)](https://huggingface.co/spaces/Shriyanshml/mediguide)
[![Model](https://img.shields.io/badge/🤗%20Model-Shriyanshml%2Fphi3--mini--qlora--mediguide-blue)](https://huggingface.co/Shriyanshml/phi3-mini-qlora-mediguide)
[![RAG Index](https://img.shields.io/badge/🤗%20Dataset-Shriyanshml%2Fmediguide--rag--index-green)](https://huggingface.co/datasets/Shriyanshml/mediguide-rag-index)
[![GitHub](https://img.shields.io/badge/GitHub-mananms21%2FMediguide-black)](https://github.com/mananms21/Mediguide-)

---

## Overview

MEDIGUIDE demonstrates a complete, production-style pipeline for building a medical AI assistant using only free-tier compute (Kaggle T4 GPU, Hugging Face Hub). The project covers:

- **Fine-tuning** — QLoRA adaptation of Microsoft Phi-3 Mini (3.8B) on 2,000 MedQuAD examples
- **RAG** — FAISS retrieval over 14,782 NIH MedQuAD question-answer pairs with `all-MiniLM-L6-v2` embeddings
- **Clinical evaluation** — a 5-level framework with a full ablation study (zero-shot / fine-tuned / +RAG / OOD) addressing the limitation that generic BERTScore cannot distinguish clinically different but semantically similar terms
- **Deployment** — Streamlit chat application with RAG toggle, and HF Spaces configuration

---

## Repository Structure

```
Mediguide/
├── app/                        # Streamlit chat application
│   ├── app.py                  # Main chat UI with RAG toggle and model loading
│   └── pages/
│       └── Evaluation.py       # Ablation dashboard (4 conditions × 5 metrics)
│
├── training/                   # Training scripts (all GPU experiments)
│   ├── train_qlora_phi3.py     # ★ Primary: Phi-3 Mini QLoRA (Kaggle T4)
│   ├── train_lora_falcon.py    # Baseline: Falcon-7B LoRA (BF16)
│   └── train_prompt_tuning_falcon.py  # Baseline: Falcon-7B Prompt Tuning
│
├── rag/                        # Retrieval-Augmented Generation module
│   ├── __init__.py
│   ├── build_index.py          # Build FAISS index from MedQuAD dataset
│   └── retriever.py            # MedRAGRetriever class (lazy-loaded)
│
├── evaluate/                   # Evaluation framework
│   ├── clinical_eval.py        # Reusable 5-level clinical evaluation module
│   ├── ablation_kaggle.py      # ★ Comprehensive ablation (4 conditions, paste-and-run)
│   ├── clinical_kaggle.py      # Legacy single-model eval script
│   ├── evaluate.py             # General evaluation utilities
│   └── results/
│       ├── results.json        # All model + ablation results
│       └── ablation_results.json  # Raw ablation output from Kaggle
│
├── spaces/                     # Hugging Face Spaces deployment
│   ├── app.py                  # Gradio interface
│   ├── requirements.txt
│   └── README.md               # HF Spaces metadata card
│
├── docs/                       # Supporting documents
│   ├── TECHNICAL_DOCUMENTATION.md  # Full technical reference (11 sections)
│   ├── bertscore_kaggle.py     # Archived: first BERTScore-only eval script
│   └── bertscore_local.py      # Archived: local BERTScore attempt
│
├── requirements.txt            # Full dependency list
├── .gitignore
└── README.md
```

---

## Quickstart

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the chat application locally

```bash
streamlit run app/app.py
```

The app auto-detects your device (CUDA → MPS → CPU) and loads the model accordingly. On Apple Silicon Macs it uses MPS in `float16`; on CPU it uses `float32`.

### 3. Build the RAG index (first run only)

The index is auto-downloaded from HF Hub on first use. To rebuild locally:

```bash
python rag/build_index.py
```

---

## Model Details

| | |
|---|---|
| **Base model** | `microsoft/Phi-3-mini-4k-instruct` (3.8B parameters) |
| **Adapter** | `Shriyanshml/phi3-mini-qlora-mediguide` |
| **Method** | QLoRA — 4-bit NF4 quantisation + LoRA rank 8 on attention layers |
| **Training data** | 2,000 samples from `keivalya/MedQuad-MedicalQnADataset` (NIH source) |
| **Training hardware** | Kaggle T4 (15 GB VRAM) |
| **Training time** | ~58.5 minutes, 3 epochs |
| **Adapter size** | 12.6 MB |
| **Trainable parameters** | 3,145,728 (0.08% of total) |

### System prompt

```
You are MEDIGUIDE, a knowledgeable medical assistant trained on authoritative
NIH sources. Provide accurate, evidence-based answers to medical questions in
a clear, empathetic tone. Always end your response with a brief disclaimer
that this is not a substitute for professional medical advice.
```

---

## RAG Pipeline

The retriever indexes the full 14,782-sample MedQuAD corpus using `sentence-transformers/all-MiniLM-L6-v2` embeddings stored in a FAISS flat-L2 index.

At inference time:
1. The user's question is embedded
2. Top-3 most similar MedQuAD passages are retrieved
3. Retrieved context is prepended to the prompt before generation

The RAG index is hosted at [`Shriyanshml/mediguide-rag-index`](https://huggingface.co/datasets/Shriyanshml/mediguide-rag-index) and downloaded automatically on first use.

---

## Evaluation

### Ablation Study Results

Four conditions evaluated on the same metric suite (Kaggle T4, 50 samples each):

| Metric | Zero-shot | Fine-tuned | **+ RAG** | OOD (PubMedQA) |
|---|---|---|---|---|
| **Clinical BERTScore F1** | 0.9203 | 0.9401 | **0.9740** | 0.9186 |
| Generic BERTScore F1 | 0.8376 | 0.8708 | **0.8965** | 0.8574 |
| ROUGE-1 (full) | 0.3149 | 0.4364 | **0.7528** | 0.1953 |
| **ROUGE-1 @50tok** ← verbosity-corrected | 0.1949 | 0.2903 | **0.4104** | 0.2392 |
| **Lexical Precision@50** | 0.3330 | 0.6252 | **0.8799** | 0.1717 |
| NLI Contradiction ↓ | 0.1007 | 0.2173 | **0.0780** | 0.1374 |
| Perplexity ↓ | 1.50 | 1.57 | **1.09** | 2.20 |

**Key findings:**
- **Fine-tuning Δ:** +0.0198 Clinical BERTScore — fine-tuning improves clinical semantic quality
- **RAG Δ:** +0.0339 — RAG contributes *more* than fine-tuning alone on top of it
- **OOD gap:** only −0.0215 (0.9401 → 0.9186 on PubMedQA) — strong generalisation to unseen biomedical domain
- **Lexical Precision@50:** 0.333 → 0.625 → **0.880** — fine-tuning nearly doubles factual precision; RAG pushes it to 88%

> **Notable finding — The NLI trade-off:** Fine-tuning *increases* NLI contradiction (0.10 → 0.22) because the model becomes more verbose and domain-specific, which the Wikipedia-trained NLI model misclassifies. RAG then *reduces* it to 0.078 — below zero-shot — because the model answers in the same NIH phrasing as the reference. **RAG is not optional; it resolves a safety trade-off introduced by fine-tuning.**

### Why Generic BERTScore Is Not Enough

Generic BERTScore uses `roberta-large` (trained on Wikipedia). In that embedding space, "heart" and "lung" are neighbours because they co-occur in similar contexts. A model that says "lung" when the correct answer is "heart" will still score ≥ 0.93.

**The evaluation framework (5 levels):**

| Level | Metric | What it catches |
|---|---|---|
| 1 | **Clinical BERTScore** (BiomedBERT) | Clinically imprecise vocabulary — distinguishes cardiac from pulmonary terms |
| 2 | **ROUGE-1 @50tok** | Verbosity-corrected overlap — are the first 50 tokens factually aligned? |
| 3 | **Lexical Precision@50** | Content-word precision on core claim — are the first 50 tokens mostly correct? |
| 4 | **NLI Contradiction Rate** | Direct factual contradiction — "increases" vs "decreases" |
| 5 | **OOD Clinical BERTScore** | Generalisation to PubMedQA, a completely different biomedical dataset |

### Running the evaluation

To reproduce the full ablation (requires Kaggle T4 or equivalent GPU):

```bash
# Paste evaluate/ablation_kaggle.py into a Kaggle notebook cell
# Results pushed automatically to Shriyanshml/mediguide-rag-index
```

To use the reusable module locally:

```python
from evaluate.clinical_eval import run_all_clinical_metrics

results = run_all_clinical_metrics(
    preds=generated_answers,
    refs=reference_answers,
    questions=questions,
    device="cpu",

)
```

---

## Comparison with Baselines

| Model | Method | Train Ex. | ROUGE-1 | ROUGE-1@50tok | Clin. BERTScore | Latency |
|---|---|---|---|---|---|---|
| **Phi-3 Mini QLoRA + RAG** ★ | QLoRA + FAISS | 2,000 | **0.753** | **0.410** | **0.974** | 11.75 s |
| **Phi-3 Mini QLoRA** | QLoRA 4-bit | 2,000 | 0.436 | 0.290 | 0.940 | 11.33 s |
| Phi-3 Mini (zero-shot) | Base model | 0 | 0.315 | 0.195 | 0.920 | 13.74 s |
| Falcon-7B QLoRA | QLoRA 4-bit | 200 | 0.250 | — | — | 10.94 s |
| Falcon-7B LoRA | LoRA BF16 | 200 | 0.210 | — | — | 3.53 s |
| Falcon-7B Prompt (4-bit) | Prompt Tuning | 200 | 0.210 | — | — | 8.81 s |
| Falcon-7B Prompt (BF16) | Prompt Tuning | 200 | 0.180 | — | — | 1.89 s |

Falcon baselines have higher full-prediction ROUGE-1 because they generate shorter, more reference-copying answers (only 200 training examples). Phi-3's lower full ROUGE reflects richer, more elaborate answers — which is desirable for a medical assistant. ROUGE-1 @50tok (verbosity-corrected) shows Phi-3's true factual precision advantage.

---

## Training Your Own Adapter

```bash
# Requires Kaggle T4 or equivalent GPU (≥15 GB VRAM)
# Add HF_Token to Kaggle secrets before running

python training/train_qlora_phi3.py
```

Key training decisions:
- **4-bit NF4 quantisation** reduces the base model from ~7.6 GB (fp16) to ~2.5 GB
- **Rank 8 LoRA** on `qkv_proj` — Phi-3 fuses Q, K, V into a single projection
- **Gradient checkpointing** trades compute for memory, enabling batch size 1 + accumulation on a T4
- **`eager` attention** used instead of `flash_attention_2` for T4 compatibility (Turing, compute cap 7.5)

---

## HF Spaces Deployment

The `spaces/` directory contains a Gradio app deployed to [Hugging Face Spaces](https://huggingface.co/spaces/Shriyanshml/mediguide) with **ZeroGPU** (free A10G tier). ZeroGPU provides GPU access on demand — inference runs in full 4-bit mode with ~11 s latency, identical to Kaggle T4 results.

1. Create a new Space at [huggingface.co/new-space](https://huggingface.co/new-space)
   - SDK: **Gradio** · Hardware: **ZeroGPU (free A10G)**
2. Push the three files from `spaces/` via git
3. Add `HF_TOKEN` as a Space secret (Settings → Variables and secrets)

---

## Dependencies

See [`requirements.txt`](requirements.txt) for the full list. Key packages:

| Package | Purpose |
|---|---|
| `transformers >= 4.44` | Model loading, tokenisation |
| `peft >= 0.12` | LoRA / QLoRA adapter management |
| `trl >= 0.11` | SFTTrainer for instruction fine-tuning |
| `bitsandbytes >= 0.43` | 4-bit NF4 quantisation |
| `sentence-transformers` | MedQuAD embedding for RAG |
| `faiss-cpu` | FAISS vector index |
| `bert-score >= 0.3.13` | BERTScore computation |
| `streamlit >= 1.38` | Chat application |
| `gradio >= 4.44` | HF Spaces interface |

---

## Limitations & Disclaimer

- This system is trained on NIH MedQuAD and is intended for **educational and research purposes only**
- It is **not a substitute for professional medical advice**, diagnosis, or treatment
- The model may generate plausible-sounding but incorrect medical information (hallucinations)
- Clinical evaluation shows fine-tuned + RAG achieves NLI contradiction rate of 7.8% (safe threshold: <10%); fine-tuned alone is 21.7% — RAG is required for safe operation
- Always consult a qualified healthcare professional for medical decisions

---

## Citation

If you use MEDIGUIDE in your research, please cite:

```bibtex
@misc{mediguide2026,
  title   = {MEDIGUIDE: Domain-Adapted Medical QA with Phi-3 Mini QLoRA and RAG},
  author  = {Raj, Shriyansh and Sharma, Manan},
  year    = {2026},
  url     = {https://github.com/mananms21/Mediguide-},
  note    = {Model: https://huggingface.co/Shriyanshml/phi3-mini-qlora-mediguide}
}
```

---

## License

This project is released under the **MIT License**. The base model (Phi-3 Mini) is subject to [Microsoft's Research License](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct). The MedQuAD dataset is from the NIH National Library of Medicine.
