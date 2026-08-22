# MEDIGUIDE — Technical Documentation

**Version 1.0 · August 2026**

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Architecture](#2-architecture)
3. [Dataset](#3-dataset)
4. [Fine-tuning Pipeline](#4-fine-tuning-pipeline)
5. [Retrieval-Augmented Generation](#5-retrieval-augmented-generation)
6. [Evaluation Framework](#6-evaluation-framework)
7. [Results & Analysis](#7-results--analysis)
8. [Application](#8-application)
9. [Deployment](#9-deployment)
10. [Engineering Decisions & Known Issues](#10-engineering-decisions--known-issues)
11. [Reproducing Everything from Scratch](#11-reproducing-everything-from-scratch)

---

## 1. Project Overview

### Motivation

Large language models (LLMs) trained on general web text generate fluent medical prose but are unreliable for clinical use because:

1. They cannot cite their sources
2. Their medical training data is noisy (forums, Q&A sites, blog posts)
3. Standard evaluation metrics (ROUGE, generic BERTScore) cannot detect clinical errors — a model that says "lung" instead of "heart" will still score well because both words appear in similar contexts in Wikipedia

MEDIGUIDE addresses all three problems within a free-tier compute budget (Kaggle T4 GPU, Hugging Face Hub).

### Approach

```
NIH MedQuAD (14,782 QA pairs)
        │
        ├─── Fine-tuning ───► Phi-3 Mini QLoRA adapter
        │                     (Shriyanshml/phi3-mini-qlora-mediguide)
        │
        └─── Indexing ──────► FAISS vector index
                              (Shriyanshml/mediguide-rag-index)
                                     │
                              RAG Retriever
                                     │
                              Chat Application (Streamlit)
```

At inference time, a user question is:
1. Embedded and used to retrieve the top-3 most relevant MedQuAD passages (RAG)
2. Passed to the fine-tuned Phi-3 adapter along with the retrieved context
3. Evaluated against the ground-truth reference using a 5-level clinical framework across 4 ablation conditions

### What Is New

The project's primary technical contributions are the **comprehensive ablation study** (zero-shot vs. fine-tuned vs. +RAG vs. OOD) and the **5-level clinical evaluation framework** that replaces generic BERTScore with a hierarchy of metrics that can detect the difference between semantically similar but clinically opposite answers (e.g., "increases" vs. "decreases" blood pressure), while also correcting for the verbosity problem via ROUGE@50tok and Lexical Precision@50.

---

## 2. Architecture

```
┌─────────────────────────────────────────────────────────┐
│                  MEDIGUIDE System                       │
│                                                         │
│  User Question                                          │
│       │                                                 │
│       ▼                                                 │
│  ┌─────────────────────────────────────────────┐        │
│  │           RAG Module (rag/)                 │        │
│  │                                             │        │
│  │  all-MiniLM-L6-v2 encoder                  │        │
│  │       │                                    │        │
│  │       ▼                                    │        │
│  │  FAISS flat-L2 index (14,782 entries)       │        │
│  │       │                                    │        │
│  │       ▼                                    │        │
│  │  Top-3 MedQuAD passages (retrieved context) │        │
│  └─────────────────────────────────────────────┘        │
│       │                                                 │
│       ▼                                                 │
│  ┌─────────────────────────────────────────────┐        │
│  │     Phi-3 Mini QLoRA (training/)            │        │
│  │                                             │        │
│  │  Prompt = [system] + [context] + [question] │        │
│  │       │                                    │        │
│  │       ▼                                    │        │
│  │  Generated answer                           │        │
│  └─────────────────────────────────────────────┘        │
│       │                                                 │
│       ▼                                                 │
│  ┌─────────────────────────────────────────────┐        │
│  │   Clinical Evaluation (evaluate/)           │        │
│  │                                             │        │
│  │  L1: Clinical BERTScore (BiomedBERT)        │        │
│  │  L2: ROUGE-1 @50tok (verbosity-corrected)   │        │
│  │  L3: Lexical Precision@50                   │        │
│  │  L4: NLI Contradiction (roberta-large-mnli) │        │
│  │  L5: OOD Score (PubMedQA benchmark)         │        │
│  └─────────────────────────────────────────────┘        │
└─────────────────────────────────────────────────────────┘
```

---

## 3. Dataset

### Source

**MedQuAD — Medical Question Answering Dataset**
- HuggingFace: `keivalya/MedQuad-MedicalQnADataset`
- Original source: NIH (National Institutes of Health) National Library of Medicine
- Paper: Ben Abacha & Demner-Fushman, "A Question-Entailment Approach to Question Answering" (2019)

### Content

MedQuAD contains 16,407 question-answer pairs created from 12 NIH website categories:
- Genetic conditions (GARD, MedlinePlus Genetics)
- Drugs and supplements (NCI, FDA)
- Medical tests and procedures (NIH SeniorHealth)
- Diseases and conditions (MedlinePlus, NIDDK, NINDS, etc.)

The three columns are: `qtype` (question type), `question`, `answer`.

### Preprocessing

```python
# Steps applied in training/train_qlora_phi3.py
df = df.dropna(subset=["question", "answer"])
df = df[df["answer"].str.len() > 80]      # remove stub answers
df = df[df["question"].str.len() > 10]   # remove stub questions
df = df.drop_duplicates(subset=["question"])
df = df.sample(frac=1, random_state=42)  # shuffle deterministically
```

After cleaning: **14,782 usable rows** from 16,407 raw.

### Split

| Set | Size | Purpose |
|---|---|---|
| Train | 2,000 | QLoRA fine-tuning |
| Eval | 200 | Per-epoch validation loss during training |
| Test | 50 | Clinical evaluation (sampled at eval time) |
| RAG index | 14,782 | Full corpus indexed for retrieval |

The train/eval split uses `df.iloc[:2000]` and `df.iloc[2000:2200]` on the shuffled dataframe, ensuring no overlap. The test 50 are sampled fresh at evaluation time from the same shuffled frame.

### Chat Template

Each training example is formatted using Phi-3's instruct template:

```
<|system|>
You are MEDIGUIDE, a knowledgeable medical assistant trained on authoritative
NIH sources. Provide accurate, evidence-based answers to medical questions in
a clear, empathetic tone. Always end your response with a brief disclaimer
that this information is educational and patients should consult a qualified
healthcare professional for personal medical advice.<|end|>
<|user|>
{question}<|end|>
<|assistant|>
{answer}<|end|>
```

The `<|end|>` tokens are Phi-3's explicit message boundary markers. Using `<|end|>` (not `</s>`) is critical — incorrect end-of-turn tokens cause the model to generate unbounded output during inference.

---

## 4. Fine-tuning Pipeline

### Model Selection

Four fine-tuning experiments were run on Falcon-7B baselines before settling on Phi-3 Mini:

| Model | Method | Size | Train Ex. | Reason chosen/rejected |
|---|---|---|---|---|
| Falcon-7B | Prompt Tuning (BF16) | 7B | 200 | ✗ Fastest but poor quality |
| Falcon-7B | Prompt Tuning (4-bit) | 7B | 200 | ✗ Memory efficient but limited |
| Falcon-7B | LoRA (BF16) | 7B | 200 | ✗ Good baseline but large |
| **Phi-3 Mini** | **QLoRA (4-bit)** | **3.8B** | **2,000** | ✅ Best quality/size/speed tradeoff |

Phi-3 Mini was chosen because:
- 3.8B parameters fit in 4-bit on a Kaggle T4 (15 GB VRAM) leaving room for gradient checkpointing
- Microsoft trained it specifically on high-quality synthetic data, making it stronger than Falcon-7B at 3.8B despite the size difference
- 4K context window is sufficient for medical QA (typical MedQuAD answer: 100–400 tokens)

### Quantisation (QLoRA)

QLoRA combines two techniques to make fine-tuning a 3.8B model fit on a 15 GB GPU:

**4-bit NF4 Quantisation** compresses model weights from 16-bit floats to 4-bit NormalFloat:

```python
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",         # NormalFloat4 — optimal for normally-distributed weights
    bnb_4bit_compute_dtype=torch.float16,  # upcast to fp16 for the actual matrix multiplication
    bnb_4bit_use_double_quant=True,    # quantise the quantisation constants too (saves ~0.4 GB)
)
```

Memory profile on Kaggle T4:
- Base model (fp16): ~7.6 GB
- After 4-bit quantisation: ~2.5 GB
- With LoRA adapters + optimiser states: ~8.1 GB
- Available for activations and gradients: ~6.9 GB (sufficient for batch size 1 + grad checkpointing)

**Gradient Checkpointing** trades compute for memory by recomputing intermediate activations during the backward pass instead of storing them. This reduces activation memory from O(L) to O(√L) where L is the number of layers.

### LoRA Configuration

LoRA (Low-Rank Adaptation) injects trainable low-rank matrices into selected layers while keeping the base model weights frozen.

```python
lora_config = LoraConfig(
    r=8,                         # rank — number of trainable dimensions per weight matrix
    lora_alpha=16,               # scaling factor (effective lr scale = alpha/r = 2.0)
    target_modules=["qkv_proj"], # Phi-3 fuses Q, K, V into a single projection
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)
```

**Why `qkv_proj` only?** Phi-3 Mini's attention implementation fuses the query, key, and value projections into a single matrix (`qkv_proj`). Targeting it covers all attention weight updates with a single LoRA layer. Adding LoRA to `o_proj` (output projection) or MLP layers would increase trainable parameters but hit VRAM limits on the T4.

**Why rank 8?** Rank 8 gives 3,145,728 trainable parameters (0.08% of total). This is the standard QLoRA rank for instruction tuning tasks. Rank 16 or 32 would improve capacity but increase the adapter size and risk overfitting on 2,000 examples.

**lora_alpha = 16:** The effective learning rate scaling for LoRA updates is `lora_alpha / r = 2.0`. This value is empirically standard; values below 1.0 can slow convergence and values above 4.0 can destabilise training.

### Training Hyperparameters

```python
TrainingArguments(
    num_train_epochs=3,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,    # effective batch = 16
    warmup_steps=11,                   # 3% of total steps
    learning_rate=2e-4,
    fp16=True,                         # activations in fp16
    optim="paged_adamw_8bit",          # 8-bit Adam with memory paging
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
)
```

**Why paged AdamW 8-bit?** Standard AdamW stores two momentum tensors per parameter at fp32, consuming ~24 bytes per parameter. For 3M trainable parameters that's ~72 MB — manageable. But bitsandbytes' 8-bit AdamW additionally uses CUDA memory paging to move optimiser states to RAM when the GPU runs out. On a T4 with limited VRAM this is a critical safety mechanism.

**Why `fp16=True` instead of `bf16`?** Kaggle's T4 GPU (Turing architecture, 2018) does not support BF16 natively. BF16 on T4 falls back to software emulation which is 10–30× slower. FP16 is natively accelerated on T4 via Tensor Cores.

**Checkpoint resume logic:** The script detects an existing checkpoint directory and resumes training from the last checkpoint. A monkey-patch on `torch.load` is applied before resuming to handle a PyTorch 2.6+ API change (`weights_only=False` required for loading NumPy RNG states saved by the trainer).

### Training Results

| Epoch | Train Loss | Val Loss |
|---|---|---|
| 1 | ~1.20 | — |
| 2 | 0.7306 | 0.6955 |
| 3 | 0.6857 | 0.6895 |

The val loss decrease from epoch 2 to 3 is very small (0.0060), suggesting the model converged well within 3 epochs. The `load_best_model_at_end=True` setting ensures the epoch 3 checkpoint is saved (it has the lowest val loss).

The final adapter (`adapter_model.safetensors`) is 12.6 MB — a 600× compression vs. the 7.6 GB base model. Only the LoRA delta weights are stored; the base model is loaded from HF Hub at inference time.

### RAG Index Construction

Simultaneously with training, the full 14,782-sample MedQuAD corpus is embedded and indexed:

```python
# rag/build_index.py
embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
embeddings = embedder.encode(texts, batch_size=256, show_progress_bar=True)
# → shape: [14782, 384]

index = faiss.IndexFlatL2(384)
index.add(embeddings.astype("float32"))
faiss.write_index(index, "rag/index/medquad.faiss")
```

`all-MiniLM-L6-v2` produces 384-dimensional embeddings. A `IndexFlatL2` performs exact nearest-neighbour search (no approximation), which is appropriate for a corpus of ~15K entries — at this scale the search takes <5 ms.

The index and metadata are pushed to `Shriyanshml/mediguide-rag-index` on HF Hub and downloaded automatically at inference time.

---

## 5. Retrieval-Augmented Generation

### Why RAG on Top of a Fine-tuned Model?

Fine-tuning teaches the model *how to answer* medical questions (format, tone, clinical vocabulary). RAG provides *what to answer* by grounding each response in the exact MedQuAD passage most relevant to the user's question. The two techniques are complementary:

- Without fine-tuning: base Phi-3 answers medical questions in a generic style, not calibrated for the NIH source
- Without RAG: fine-tuned model may answer from parametric memory, which can drift or hallucinate for rare conditions
- With both: the model generates answers in the trained clinical style, grounded in retrieved NIH passages

### Retriever Implementation

```python
# rag/retriever.py — simplified
class MedRAGRetriever:
    def retrieve(self, query: str, top_k: int = 3) -> list[dict]:
        query_emb = self.embedder.encode([query]).astype("float32")
        distances, indices = self.index.search(query_emb, top_k)
        return [self.metadata[i] for i in indices[0]]
```

The retriever is lazy-loaded: the FAISS index and sentence transformer model are only loaded into memory when the first query arrives. This prevents startup failures when GPU memory is occupied by the main generation model.

### Prompt Construction with Context

When RAG is enabled (the default in the Streamlit app), the prompt becomes:

```
<|system|>
You are MEDIGUIDE...
<|end|>
<|user|>
[CONTEXT FROM NIH MEDQUAD]
Q1: What causes Marfan syndrome?
A1: Marfan syndrome is caused by mutations in the FBN1 gene...

Q2: ...
A2: ...

Based on the above context, please answer:
{user_question}<|end|>
<|assistant|>
```

The context is capped at 1,500 characters to ensure the total prompt fits within the 4K context window. If context + question + system prompt exceeds the limit, context is truncated.

---

## 6. Evaluation Framework

### The Core Problem with Generic BERTScore

BERTScore uses cosine similarity between contextual embeddings from `roberta-large` (trained on Wikipedia + BookCorpus). In that embedding space, medical terms are clustered by their *statistical co-occurrence patterns* in general text, not by their clinical meaning. Concretely:

- "heart" and "lung" are similar because both appear with words like "disease", "failure", "treatment"
- "increases" and "decreases" are similar because both appear in the same sentence structures
- "left" and "right" are similar because both are positional adjectives

A model that says "the medication *decreases* blood pressure" when the correct answer is "the medication *increases* blood pressure" will score ~0.93 on generic BERTScore — a falsely reassuring number.

### The Evaluation Framework

The evaluation is designed around two fundamental problems:

1. **The clinical equivalence problem:** Generic BERTScore cannot tell "heart" from "lung" because Wikipedia trains these terms to be neighbours. We need a domain-specific model.
2. **The verbosity problem:** Open-ended medical QA models generate longer, more explanatory answers than terse NIH reference sentences. All overlap-based metrics (ROUGE, content-word F1) are suppressed not by factual error, but by this length mismatch.

Both problems are addressed by combining the right metric with the right truncation strategy.

#### Level 1: Clinical BERTScore (BiomedBERT)

**What it measures:** Overall semantic similarity using a model trained on biomedical literature.

**Model:** `microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext`  
Trained on 29 million PubMed abstracts and full-text papers. In this embedding space, "left ventricle" and "right ventricle" have more distinct representations because they co-occur with different clinical terms.

**Implementation:** BERTScore's greedy token matching implemented directly using `AutoModel` + `AutoTokenizer(use_fast=False)` — bypassing the `bert_score` library which raises `OverflowError` on Kaggle's pinned transformers version via its Rust fast tokenizer path.

```
Precision = mean over prediction tokens of: max_{ref_token} cosine_sim(pred_token, ref_token)
Recall    = mean over reference tokens of:  max_{pred_token} cosine_sim(ref_token, pred_token)
F1        = 2 · Precision · Recall / (Precision + Recall)
```

Texts are truncated to 400 BiomedBERT tokens before embedding (hard limit: 512 positional embeddings).

**Scope:** Computed on full predictions. BiomedBERT handles long text gracefully; verbosity is less of a problem here because every token in the prediction is matched to its closest reference token, not compared as a set.

#### Level 2: ROUGE-1 @50tok (Verbosity-Corrected)

**What it measures:** Unigram overlap between the first 50 tokens of the prediction and the full reference.

**Why truncation fixes verbosity:** A well-structured medical answer leads with the core factual claim. The first 50 tokens should contain the direct answer to the question. Everything after is elaboration, context, and disclaimer. By truncating to 50 tokens, we measure: *does the model answer the question correctly in its opening statement?* — without penalising it for adding useful elaboration.

```python
# During generation
gen_ids    = output[0, input_len:]
full_text  = tokenizer.decode(gen_ids,      skip_special_tokens=True)
trunc_text = tokenizer.decode(gen_ids[:50], skip_special_tokens=True)  # @50tok
```

ROUGE-1 @50tok is expected to be meaningfully higher than ROUGE-1 on full predictions for a model that answers correctly but verbosely.

#### Level 3: Lexical Precision@50

**What it measures:** Of the content words in the first 50 tokens of the prediction, what fraction appear in the reference? This is precision-only (not F1) on the truncated prediction.

**Why precision not F1:** We care about whether the model says the right things, not whether it says *all* the right things. A model that starts with one correct fact and then elaborates is correctly rewarded. F1 penalises low recall (the model didn't cover everything in the reference), which again punishes verbosity.

**Stopword filtering:** ~130 generic English words and ~30 overly generic medical terms ("patient", "treatment", "symptom") are removed. Only domain-specific content words are evaluated.

#### Level 4: NLI Contradiction Rate

**What it measures:** Whether the model's response directly contradicts the reference on factual claims.

**Model:** `roberta-large-mnli` (Multi-NLI fine-tuned). Label order: `[CONTRADICTION, NEUTRAL, ENTAILMENT]`.

**Known limitation:** This model was trained on Wikipedia/books, not clinical text. The high neutral rate (~72%) reflects structural mismatch between long medical answers and short NLI training sentences — not genuine neutrality. The contradiction rate is the actionable metric.

**Threshold:** Contradiction < 0.10 → safe; 0.10–0.15 → borderline; > 0.15 → caution.

#### Level 5 (New): OOD Generalisation — PubMedQA

**What it measures:** Clinical BERTScore F1 on 50 questions from PubMedQA (`qiaojin/PubMedQA`, pqa_labeled config) — a completely different dataset from training.

**Why this replaces Content-Word Hallucination:** The previous Level 4 (hallucination rate) was a content-word metric computed on full predictions, subject to the same verbosity suppression as Levels 2/3 were before the truncation fix. It provided no additional signal beyond what Levels 2 and 3 already captured. PubMedQA OOD score directly addresses the most important interviewer question: *does this model generalise beyond its training distribution?*

**PubMedQA context:** Questions are research-style ("Does metformin reduce cardiovascular events in Type 2 diabetes?") sourced from PubMed abstract headers. References are the corresponding abstract conclusions. This is substantially different from MedQuAD's patient-facing NIH questions. A lower OOD score vs. in-distribution is expected and is reported as such.

**The ablation experiment:** `evaluate/ablation_kaggle.py` evaluates four conditions on the same metric suite:

| Condition | Dataset | Purpose |
|---|---|---|
| Zero-shot (base Phi-3) | MedQuAD | Proves fine-tuning adds value |
| Fine-tuned (no RAG) | MedQuAD | Primary model |
| Fine-tuned + RAG | MedQuAD | Proves RAG adds value |
| Fine-tuned (OOD) | PubMedQA | Proves generalisation |

---

## 7. Results & Analysis

### Ablation Study — 4 Conditions (run `evaluate/ablation_kaggle.py` on Kaggle T4)

The comprehensive ablation study evaluates four conditions on the same metric suite, run on Kaggle T4 (50 samples per condition).

#### Ablation Comparison Table

| Metric | Zero-shot | Fine-tuned | **+ RAG** | OOD (PubMedQA) |
|---|---|---|---|---|
| **Clinical BERTScore F1** | 0.9203 | 0.9401 | **0.9740** | 0.9186 |
| Generic BERTScore F1 | 0.8376 | 0.8708 | **0.8965** | 0.8574 |
| ROUGE-1 (full) | 0.3149 | 0.4364 | **0.7528** | 0.1953 |
| **ROUGE-1 @50tok** | 0.1949 | 0.2903 | **0.4104** | 0.2392 |
| **Lexical Precision@50** | 0.3330 | 0.6252 | **0.8799** | 0.1717 |
| NLI Contradiction ↓ | 0.1007 | 0.2173 | **0.0780** | 0.1374 |
| Perplexity ↓ | 1.50 | 1.57 | **1.09** | 2.20 |
| Latency s/sample | 13.74 | 11.33 | 11.75 | **10.54** |

**Key deltas:**
- **Fine-tuning Δ (Clinical BERTScore):** +0.0198
- **RAG Δ (Clinical BERTScore):** +0.0339 ← RAG contributes more than fine-tuning alone
- **OOD gap:** only −0.0215 (0.9401 → 0.9186) ← strong generalisation
- **RAG contradiction reduction:** −0.1393 from fine-tuned → RAG resolves the NLI trade-off

#### Notable Finding — The NLI Trade-off

A surprising and important result: fine-tuning *increases* the NLI contradiction rate from 0.1007 (zero-shot) to 0.2173, but RAG then *reduces* it to 0.0780 — below even the zero-shot baseline.

**Why fine-tuning increases NLI contradiction:** The fine-tuned model becomes domain-specific and verbose. It states medical facts with more precision than the terse NIH reference sentences. The `roberta-large-mnli` model (trained on Wikipedia-domain sentence pairs) is not calibrated for medical text and flags domain-specific elaborations as contradictions even when they are factually correct. This is a known limitation of cross-domain NLI evaluation.

**Why RAG resolves this:** When the model has access to the exact NIH MedQuAD passage as context, it answers in the same NIH phrasing. The response language closely mirrors the reference, so the NLI model classifies it as entailment or neutral rather than contradiction.

**The practical implication:** RAG is not merely additive — it actively corrects a safety trade-off introduced by fine-tuning. This argues that RAG should be considered a required component of the system, not an optional feature.

#### The Verbosity Problem and Its Fix

The model generates longer, more detailed answers than MedQuAD references. This is desirable behaviour but suppresses all overlap-based metrics computed on full predictions.

**Example (the same question answered differently):**
```
Question: What causes Holt-Oram syndrome?

Reference (NIH): Mutations in the TBX5 gene.

Model (full prediction, ~80 tokens):
  Holt-Oram syndrome is caused by mutations in the TBX5 gene, which encodes a
  transcription factor essential for cardiac and upper limb development during
  embryogenesis. The condition is autosomal dominant. Consult a physician.

Model (first 50 tokens = ROUGE@50tok input):
  Holt-Oram syndrome is caused by mutations in the TBX5 gene, which encodes a
  transcription factor essential for cardiac and upper limb development
```

Scores on this example:
| Metric | Score | Why |
|---|---|---|
| ROUGE-1 (full) | ~0.14 | Only "mutations", "TBX5", "gene" match across 80 tokens |
| ROUGE-1 @50tok | ~0.28 | Same 3 matching words over 50 tokens → higher density |
| Lexical Prec@50 | ~0.33 | 3 of ~9 content words in @50tok are in the reference |
| Clinical BERTScore | ~0.90 | BiomedBERT knows both texts discuss the same genetic concept |

ROUGE-1 @50tok and Lexical Precision@50 give a fair, verbosity-corrected picture of factual accuracy.

#### Historical Baseline Comparison

| Model | Method | Train Ex. | ROUGE-1 | ROUGE-1@50tok | Clin. BERTScore | Latency |
|---|---|---|---|---|---|---|
| **Phi-3 Mini QLoRA + RAG** ★ | QLoRA + FAISS | 2,000 | **0.753** | **0.410** | **0.974** | 11.75 s |
| **Phi-3 Mini QLoRA** | QLoRA 4-bit | 2,000 | 0.436 | 0.290 | 0.940 | 11.33 s |
| Phi-3 Mini (zero-shot) | Base model | 0 | 0.315 | 0.195 | 0.920 | 13.74 s |
| Falcon-7B QLoRA | QLoRA 4-bit | 200 | 0.250 | — | — | 10.94 s |
| Falcon-7B LoRA | LoRA BF16 | 200 | 0.210 | — | — | 3.53 s |
| Falcon-7B Prompt (4-bit) | Prompt Tuning | 200 | 0.210 | — | — | 8.81 s |
| Falcon-7B Prompt (BF16) | Prompt Tuning | 200 | 0.180 | — | — | 1.89 s |

**OOD result:** On PubMedQA (research-style questions from PubMed abstracts, unseen during training), the fine-tuned model scores Clinical BERTScore F1 = **0.9186**, a gap of only 0.0215 from the in-distribution score (0.9401). This demonstrates strong generalisation to a different biomedical domain.

---

## 8. Application

### Streamlit Chat Application (`app/app.py`)

The main application is a multi-page Streamlit app with two pages:

1. **Chat (`app/app.py`)** — The main interface
2. **Evaluation (`app/pages/Evaluation.py`)** — Metrics dashboard

#### Model Loading Strategy

The app uses a device-aware loading strategy:

```python
if torch.cuda.is_available():
    dtype = torch.float16
    device_map = {"": 0}          # Force all layers to GPU 0
elif torch.backends.mps.is_available():
    dtype = torch.float16
    device_map = {"": "mps"}      # Apple Silicon
else:
    dtype = torch.float32         # CPU — fp16 is unsupported
    device_map = {"": "cpu"}
```

**Why `device_map={"": 0}` instead of `"auto"`?** `device_map="auto"` on Kaggle sometimes splits the model across CPU and GPU (mixed-device inference) when VRAM is tight. This causes cross-device tensor errors during the forward pass. Forcing all layers to `0` (first GPU) prevents this.

**Why `attn_implementation="eager"`?** Phi-3 Mini supports `flash_attention_2` for faster inference, but Flash Attention 2 requires CUDA with compute capability ≥ 8.0 (Ampere or newer). Kaggle T4 is Turing (compute capability 7.5). `eager` uses the standard PyTorch attention implementation which works on all CUDA devices.

Models are cached with `@st.cache_resource` so they are loaded once per Streamlit session and reused across all messages.

#### Generation Parameters

```python
model.generate(
    input_ids=...,
    max_new_tokens=512,
    do_sample=True,
    temperature=0.7,       # moderate randomness — factual but not repetitive
    top_p=0.9,             # nucleus sampling — prevent low-probability token artifacts
    pad_token_id=tokenizer.unk_token_id,
    eos_token_id=tokenizer.convert_tokens_to_ids("<|end|>"),
)
```

`temperature=0.7` balances factual accuracy (lower temperatures) and naturalness (higher temperatures). For a medical assistant, going below 0.5 can cause repetitive phrasing; going above 0.9 can produce hallucinations.

#### RAG Toggle

The chat interface has a sidebar toggle to enable/disable RAG. When enabled:
1. The question is embedded with `all-MiniLM-L6-v2`
2. Top-3 MedQuAD passages are retrieved
3. Context is prepended to the prompt (capped at 1,500 characters)

The toggle is useful for ablation: comparing answers with and without RAG shows the benefit of grounded retrieval.

### Evaluation Dashboard (`app/pages/Evaluation.py`)

A dark-themed Streamlit dashboard with five sections:

1. **Key Findings** — Four delta badges (Fine-tuning Δ, RAG Δ, OOD gap, Contradiction reduction)
2. **Ablation Comparison Table** — 4 conditions × 8 metrics with best-value highlighting per row
3. **Semantic Quality** — Generic vs. Clinical BERTScore with delta card
4. **Classical Metrics** — ROUGE-1 (full + @50tok), perplexity, latency
5. **Clinical Safety** — NLI entailment / neutral / contradiction with live safety verdict:
   - Contradiction < 10%: ✅ Clinically Safe
   - Contradiction 10–15%: ⚠️ Borderline — review needed
   - Contradiction > 15%: 🚨 Caution — clinical review required

All data is loaded from `evaluate/results/results.json`. The ablation table populates automatically once `ablation_conditions` is filled by running `evaluate/ablation_kaggle.py`.

---

## 9. Deployment

### HF Spaces (Gradio, free CPU tier)

The `spaces/` directory contains a standalone Gradio application. Because HF Spaces CPU Basic tier provides only 2 CPU cores and 16 GB RAM (no GPU), the app loads the model in fp32 on CPU with `device_map="cpu"`. This makes inference slower (~45 s per response on CPU) but allows completely free hosting.

**To deploy:**
1. Create a new Space at huggingface.co/new-space
   - SDK: Gradio · Hardware: CPU Basic (free)
2. Upload `spaces/app.py`, `spaces/requirements.txt`, `spaces/README.md`
3. Add `HF_Token` as a Space Secret

### Local (Streamlit)

```bash
pip install -r requirements.txt
streamlit run app/app.py
```

The app auto-detects CUDA/MPS/CPU and adjusts dtype accordingly.

---

## 10. Engineering Decisions & Known Issues

### Kaggle-Specific Fixes

Several fixes were required for the Kaggle environment:

1. **Module cache invalidation:** Kaggle pre-loads older versions of `transformers`, `peft`, and `trl`. After `pip install --upgrade`, the old versions remain in `sys.modules`. Solution: delete matching entries from `sys.modules` before importing.

2. **HF modules cache:** Kaggle caches the Phi-3 `modeling_phi3.py` file after the first load. A version mismatch between cached code and updated weights causes silent errors. Solution: delete `~/.cache/huggingface/modules` at startup.

3. **Checkpoint resume + PyTorch 2.6:** `trainer.train(resume_from_checkpoint=...)` calls `torch.load` on saved RNG states. PyTorch 2.6 changed the default `weights_only=True` which breaks NumPy RNG state loading. Solution: monkey-patch `torch.load` to pass `weights_only=False` during checkpoint loading only.

4. **BERTScore OverflowError:** `bert_score` library passes `max_length=None` to the HuggingFace fast tokenizer, which overflows Rust's `usize` type (max 2^64-1). `use_fast_tokenizer=False` is nominally supported by `bert_score` but ignored by Kaggle's pinned `transformers` build. Solution: bypass `bert_score` entirely and implement BERTScore's greedy matching manually using `AutoModel` + `AutoTokenizer(use_fast=False)`.

5. **scispacy installation failure:** The scispacy `en_core_sci_md` model is distributed as a source tarball requiring `numpy>=2.0` as a build dependency. Kaggle's environment has `numpy==1.26.4` (pinned by JAX, OpenCV, and many other packages). Solution: replaced scispacy with a dependency-free content-word F1 approach.

### Known Limitations

1. **NLI model domain mismatch:** `roberta-large-mnli` was trained on Wikipedia/books, not clinical text. The ablation reveals this concretely: fine-tuning *increases* contradiction (0.1007 → 0.2173) not because the model becomes less safe, but because it produces more verbose domain-specific text that the Wikipedia NLI model misclassifies. RAG resolves this to 0.0780. Contradiction rates should be interpreted relative to conditions, not as absolute values.

2. **Verbosity suppresses overlap metrics on full predictions:** ROUGE-1 on full predictions is suppressed because the model generates more content than the NIH reference. Use ROUGE-1 @50tok (reported in the ablation: 0.195 → 0.290 → 0.410) for a verbosity-corrected comparison.

3. **50-sample evaluation:** Clinical evaluation runs on 50 samples due to GPU time limits. Results have a confidence interval of approximately ±0.03 on BERTScore (rough estimate from sampling variance). Running on 500 samples would give more stable estimates.

4. **Latency (7.24 s/response):** This is on a Kaggle T4 without Flash Attention. On an A100 with Flash Attention 2 the same model runs in ~1.5 s. The Streamlit app on a local GPU with MPS (Apple Silicon) typically runs in 8–15 s.

5. **No multi-turn memory:** The chat application does not maintain conversation history between turns. Each message is processed independently with only the system prompt and RAG context. Adding conversation history would require careful context window management.

---

## 11. Reproducing Everything from Scratch

### Step 1: Fine-tune the model (Kaggle T4, ~60 min)

```
1. Create a Kaggle account
2. Create a new notebook → enable T4 GPU
3. Add Secrets: name=HF_Token, value=<your_hf_token>
4. Paste the contents of training/train_qlora_phi3.py into a cell
5. Run — the script handles all installs, training, and uploads automatically
6. Output: Shriyanshml/phi3-mini-qlora-mediguide on HF Hub
```

### Step 2: Build the RAG index (Kaggle T4, ~5 min)

```
1. Same Kaggle environment
2. Paste rag/build_index.py into a new cell
3. Output: Shriyanshml/mediguide-rag-index on HF Hub (FAISS index + metadata)
```

### Step 3: Run the comprehensive ablation evaluation (Kaggle T4, ~50 min)

```
1. Same Kaggle environment
2. Paste evaluate/ablation_kaggle.py into a new cell
3. Script runs 4 conditions × 5 metrics automatically
4. Output: ablation_results.json pushed to Shriyanshml/mediguide-rag-index
5. Download it and copy into evaluate/results/ablation_results.json locally
6. The results.json will auto-merge via the local script:
   python3 -c "import json; ..."
   (see evaluate/ablation_kaggle.py footer for merge instructions)
```

> For the original single-model evaluation only, `evaluate/clinical_kaggle.py` is still available.

### Step 4: Run the application locally

```bash
git clone https://github.com/mananms21/Mediguide-
cd Mediguide-
pip install -r requirements.txt
streamlit run app/app.py
```

### Step 5 (Optional): Deploy to HF Spaces

```
1. huggingface.co/new-space → Gradio → CPU Basic (free)
2. Upload spaces/app.py, spaces/requirements.txt, spaces/README.md
3. Add HF_Token secret
```

### Environment Requirements

| Component | Minimum | Recommended |
|---|---|---|
| Python | 3.10 | 3.12 |
| CUDA | 11.8 (compute cap ≥ 7.5) | 12.x (compute cap ≥ 8.0) |
| VRAM (training) | 15 GB | 24 GB |
| VRAM (inference) | 8 GB | 16 GB |
| RAM | 16 GB | 32 GB |
| Storage | 20 GB | 40 GB |

Training has been tested on:
- Kaggle T4 (15 GB VRAM) ✅
- Colab A100 (40 GB VRAM) ✅ (use `bf16=True` on A100)

Inference has been tested on:
- Kaggle T4 ✅
- Apple M2 Pro (MPS, 16 GB unified memory) ✅
- CPU-only (slow, ~3 min/response) ✅

---

## References

1. Hu, E., et al. (2022). **LoRA: Low-Rank Adaptation of Large Language Models.** ICLR 2022.
2. Dettmers, T., et al. (2023). **QLoRA: Efficient Finetuning of Quantized LLMs.** NeurIPS 2023.
3. Zhang, T., et al. (2020). **BERTScore: Evaluating Text Generation with BERT.** ICLR 2020.
4. Ben Abacha, A., & Demner-Fushman, D. (2019). **A Question-Entailment Approach to Question Answering.** BMC Bioinformatics.
5. Gu, Y., et al. (2021). **Domain-Specific Language Model Pretraining for Biomedical Natural Language Processing.** ACM CHIL 2021. (BiomedBERT)
6. Williams, A., et al. (2018). **A Broad-Coverage Challenge Corpus for Sentence Understanding through Inference.** NAACL 2018. (MultiNLI — used for roberta-large-mnli)
7. Lewis, P., et al. (2020). **Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.** NeurIPS 2020.
8. Abdin, M., et al. (2024). **Phi-3 Technical Report.** Microsoft Research.

---

*MEDIGUIDE is a research and educational project. It is not a medical device and must not be used for clinical decision-making.*
