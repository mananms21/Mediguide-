# MEDIGUIDE 🩺

**A domain-adapted medical question-answering system built on Phi-3 Mini, fine-tuned with QLoRA and augmented with Retrieval-Augmented Generation (RAG) over NIH MedQuAD.**

[![Model](https://img.shields.io/badge/🤗%20Model-Shriyanshml%2Fphi3--mini--qlora--mediguide-blue)](https://huggingface.co/Shriyanshml/phi3-mini-qlora-mediguide)
[![RAG Index](https://img.shields.io/badge/🤗%20Dataset-Shriyanshml%2Fmediguide--rag--index-green)](https://huggingface.co/datasets/Shriyanshml/mediguide-rag-index)
[![GitHub](https://img.shields.io/badge/GitHub-mananms21%2FMediguide-black)](https://github.com/mananms21/Mediguide-)

---

## Overview

MEDIGUIDE demonstrates a complete, production-style pipeline for building a medical AI assistant using only free-tier compute (Kaggle T4 GPU, Hugging Face Hub). The project covers:

- **Fine-tuning** — QLoRA adaptation of Microsoft Phi-3 Mini (3.8B) on 2,000 MedQuAD examples
- **RAG** — FAISS retrieval over 14,782 NIH MedQuAD question-answer pairs with `all-MiniLM-L6-v2` embeddings
- **Clinical evaluation** — a 4-level framework addressing the limitation that generic BERTScore cannot distinguish clinically different but semantically similar terms
- **Deployment** — Streamlit chat application with RAG toggle, and HF Spaces configuration

---

## Repository Structure

```
Mediguide/
├── app/                        # Streamlit chat application
│   ├── app.py                  # Main chat UI with RAG toggle and model loading
│   └── pages/
│       └── Evaluation.py       # Interactive metrics dashboard
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
│   ├── clinical_eval.py        # Reusable 4-level clinical evaluation module
│   ├── clinical_kaggle.py      # ★ Complete Kaggle eval script (paste-and-run)
│   ├── evaluate.py             # General evaluation utilities
│   └── results/
│       └── results.json        # All model results (updated after each run)
│
├── spaces/                     # Hugging Face Spaces deployment
│   ├── app.py                  # Gradio interface
│   ├── requirements.txt
│   └── README.md               # HF Spaces metadata card
│
├── docs/                       # Supporting documents
│   ├── Mediguide report.pdf    # Original project report
│   ├── bertscore_kaggle.py     # Archived: first BERTScore-only eval script
│   └── bertscore_local.py      # Archived: local BERTScore attempt
│
├── notebooks/                  # Archived Jupyter experiments (for reference)
│   ├── final_qlora_mediguide.ipynb
│   ├── final lora mediguide.ipynb
│   ├── final_prompt_mediguide.ipynb
│   └── final fp prompt mediguide.ipynb
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

### Results Summary

| Metric | Value | Notes |
|---|---|---|
| **Clinical BERTScore F1** | **0.9012** | BiomedBERT (PubMed-trained) greedy token matching |
| Generic BERTScore F1 | 0.8042 | roberta-large baseline |
| Δ Clinical − Generic | +0.0970 | Model uses domain-specific clinical vocabulary |
| Perplexity | **2.57** | Model is confident in its outputs |
| ROUGE-1 | 0.1852 | Suppressed by verbosity (see note below) |
| ROUGE-2 | 0.0255 | |
| ROUGE-L | 0.0952 | |
| Content-Word F1 | 0.127 | Stopword-filtered token overlap |
| NLI Entailment Rate | 0.1595 | Fraction of responses consistent with reference |
| NLI Contradiction Rate | 0.1248 | Clinical danger metric (borderline) |
| Avg latency | 7.24 s | Per response on Kaggle T4 |

> **Note on ROUGE and Content-Word F1:** These scores appear low because the model generates verbose, explanatory answers (e.g., 150 tokens) while MedQuAD references are often concise (e.g., 15 tokens). The low scores reflect this verbosity gap, not factual inaccuracy. Clinical BERTScore and perplexity are the more meaningful signals.

### Why Generic BERTScore Is Not Enough

A core contribution of this project is demonstrating that generic BERTScore (using `roberta-large` trained on Wikipedia) cannot distinguish clinically different but semantically similar terms. In Wikipedia's embedding space, "heart" and "lung" are neighbours because they co-occur in similar contexts.

**The 4-level clinical evaluation framework:**

| Level | Metric | What it catches |
|---|---|---|
| 1 | Clinical BERTScore | Imprecise clinical vocabulary — uses BiomedBERT (29M PubMed abstracts) |
| 2 | Content-Word F1 | Term-level factual overlap after removing generic stopwords |
| 3 | NLI Contradiction Rate | Direct clinical contradiction — "increases" vs "decreases", "left" vs "right" |
| 4 | Content-Word Hallucination | Prediction words not grounded in question or reference |

### Running the evaluation

To reproduce the full clinical evaluation (requires Kaggle T4 or equivalent GPU):

```bash
# Paste evaluate/clinical_kaggle.py into a Kaggle notebook cell
# Results are pushed automatically to Shriyanshml/mediguide-rag-index
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

| Model | Method | Train examples | BERTScore F1 | ROUGE-1 | Latency |
|---|---|---|---|---|---|
| **Phi-3 Mini QLoRA** ★ | QLoRA 4-bit | 2,000 | **0.8042** | 0.1852 | 7.24 s |
| Falcon-7B QLoRA | QLoRA 4-bit | 200 | — | 0.25 | 10.94 s |
| Falcon-7B LoRA | LoRA BF16 | 200 | — | 0.21 | 3.53 s |
| Falcon-7B Prompt (4-bit) | Prompt Tuning | 200 | — | 0.21 | 8.81 s |
| Falcon-7B Prompt (BF16) | Prompt Tuning | 200 | — | 0.18 | 1.89 s |

The Phi-3 Mini QLoRA model was trained on 10× more data and achieves a **Clinical BERTScore F1 of 0.9012** — the most clinically meaningful metric.

---

## Training Your Own Adapter

```bash
# Requires Kaggle T4 or equivalent GPU (≥15 GB VRAM)
# Add HF_Token to Kaggle secrets before running

python training/train_qlora_phi3.py
```

Key training decisions:
- **4-bit NF4 quantisation** reduces the base model from ~14 GB (fp16) to ~4 GB
- **Rank 8 LoRA** on `q_proj` and `v_proj` only — minimal parameters, maximum efficiency
- **Gradient checkpointing** trades compute for memory, enabling batch size 2 on a T4
- **`eager` attention** used instead of `flash_attention_2` for compatibility

---

## HF Spaces Deployment

The `spaces/` directory contains a Gradio app ready for one-click deployment to [Hugging Face Spaces](https://huggingface.co/spaces) (free CPU tier).

1. Create a new Space at [huggingface.co/new-space](https://huggingface.co/new-space)
   - SDK: **Gradio** · Hardware: **CPU Basic (free)**
2. Upload the three files from `spaces/`
3. Add `HF_Token` as a Space secret

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
- Clinical evaluation (NLI contradiction rate 12.5%) shows the model is borderline safe but not validated for clinical deployment
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
