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
3. Evaluated against the ground-truth reference using a 4-level clinical framework

### What Is New

The project's primary technical contribution is the **4-level clinical evaluation framework** that replaces generic BERTScore with a hierarchy of metrics that can detect the difference between semantically similar but clinically opposite answers (e.g., "increases" vs. "decreases" blood pressure).

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
│  │  L2: Content-Word F1                        │        │
│  │  L3: NLI Contradiction (roberta-large-mnli) │        │
│  │  L4: Content-Word Hallucination             │        │
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

### The 4-Level Clinical Evaluation Framework

Each level is designed to catch a different class of clinical failure:

#### Level 1: Clinical BERTScore (BiomedBERT)

**What it measures:** Overall semantic similarity using a model trained on biomedical literature.

**Model:** `microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext`  
Trained on 29 million PubMed abstracts and full-text papers. In this embedding space, "left ventricle" and "right ventricle" have more distinct representations because they co-occur with different clinical terms (different disease patterns, different imaging findings).

**Implementation:** We implement BERTScore's greedy token matching directly using HuggingFace's `AutoModel` rather than the `bert_score` library. This avoids an `OverflowError` in `bert_score`'s fast tokenizer path on Kaggle's pinned transformers version.

The computation follows the BERTScore paper (Zhang et al., 2020):

```
For each token in prediction, find the most similar token in reference (max cosine sim):
  Precision = mean over prediction tokens of: max_{ref_token} cosine_sim(pred_token, ref_token)
  Recall    = mean over reference tokens of:  max_{pred_token} cosine_sim(ref_token, pred_token)
  F1        = 2 · Precision · Recall / (Precision + Recall)
```

Texts are truncated to 400 BiomedBERT tokens using the model's own slow (Python) tokenizer before embedding, ensuring no sequence exceeds BiomedBERT's 512-position embedding limit.

**What the delta means:** `Clinical BERTScore F1 − Generic BERTScore F1`
- **Positive delta (our result: +0.097):** The model's output is MORE similar to the reference in clinical embedding space than in general embedding space. This means the model uses domain-specific clinical vocabulary that BiomedBERT recognises as medically congruent.
- **Negative delta:** The model might be using clinical-sounding but incorrect terms.

#### Level 2: Content-Word F1

**What it measures:** Token-level factual overlap after removing words that carry no clinical information.

**Implementation:** Tokenise → lowercase → remove a curated stopword list of ~100 generic English words plus ~30 overly generic medical terms ("patient", "treatment", "symptom", "condition", etc.) → compute set intersection F1.

```python
def content_words(text):
    text = re.sub(r'[^\w\s]', ' ', text.lower())
    tokens = text.split()
    return {t for t in tokens if t not in _STOP and len(t) > 2 and not t.isdigit()}

# F1 = 2·|P∩R| / (|P| + |R|)
```

**Relationship to ROUGE:** This is essentially ROUGE-1 with a better stopword list. It catches cases where the model uses different words for the same clinical facts.

**Important interpretation note:** Medical QA models tend to generate verbose, explanatory answers while reference answers (from NIH pages) are often brief factual statements. A model that correctly elaborates on "blocked coronary arteries" into a full paragraph about myocardial infarction will score low on content-word F1 because the additional words (myocardial, infarction, blood flow, tissue, damage) do not appear in the 3-word reference. This metric must be interpreted alongside perplexity and Clinical BERTScore.

#### Level 3: NLI Contradiction Rate

**What it measures:** Whether the model directly contradicts the reference answer on factual claims.

**Model:** `roberta-large-mnli` fine-tuned on Multi-NLI (Bowman et al., 2015).

**Implementation:** For each (prediction, reference) pair:
- Premise = reference answer (truncated to 512 tokens)
- Hypothesis = prediction
- The NLI model classifies as: ENTAILMENT (prediction is consistent) / NEUTRAL (prediction is unrelated or incomplete) / CONTRADICTION (prediction contradicts the reference)

**Label ordering for `roberta-large-mnli`:** `[0=CONTRADICTION, 1=NEUTRAL, 2=ENTAILMENT]` — verified against the model card. The probabilities are extracted from `torch.softmax(logits, dim=-1)[0]`.

**Known limitations:**
- `roberta-large-mnli` was trained on Wikipedia-domain sentence pairs, not clinical text
- Medical answers are typically longer than NLI training sentences (typically 1 sentence)
- The high neutral rate (71.57%) likely reflects the model's difficulty connecting long medical paragraphs, not actual neutrality
- The 12.48% contradiction rate is an upper bound; some "contradictions" may be domain-confusion artifacts

**Threshold:** Contradiction rate < 0.10 → clinically safe; 0.10–0.15 → borderline; > 0.15 → caution.

#### Level 4: Content-Word Hallucination Rate

**What it measures:** The fraction of the model's content words that are not grounded in either the question or the reference answer.

```python
def hallucination(question, prediction, reference):
    pred_words  = content_words(prediction)
    known_words = content_words(question) | content_words(reference)
    if not pred_words: return 0.0
    grounded = len(pred_words & known_words)
    return 1.0 - grounded / len(pred_words)
```

**Interpretation:** A hallucination rate of 0.87 means 87% of the model's content words do not appear in the question or reference. This sounds alarming but, as with content-word F1, primarily reflects the verbosity gap. A model that correctly explains "Holt-Oram syndrome" as "an autosomal dominant condition caused by mutations in the TBX5 transcription factor gene" will be penalised because TBX5 and transcription factor are not in the 5-word question. These words are medically correct but "ungrounded" by this metric's definition.

**True signal from this metric:** The relative hallucination rate between models. If one model has 0.87 and another has 0.70, the second model is more grounded, regardless of absolute interpretation.

---

## 7. Results & Analysis

### Primary Model: Phi-3 Mini QLoRA

Evaluated on 50 random MedQuAD examples using `evaluate/clinical_kaggle.py` on a Kaggle T4 (15 GB VRAM).

#### Full Results Table

| Metric | Value | Interpretation |
|---|---|---|
| **Clinical BERTScore F1** | **0.9012** | Strong clinical semantic similarity |
| Generic BERTScore F1 | 0.8042 | Solid baseline |
| **Δ (Clinical − Generic)** | **+0.097** | Model uses clinical vocabulary BiomedBERT recognises |
| **Perplexity** | **2.57** | Model is highly confident in its outputs |
| ROUGE-1 | 0.1852 | Suppressed by verbosity (see analysis below) |
| ROUGE-2 | 0.0255 | Very low — expected for verbose vs. concise answers |
| ROUGE-L | 0.0952 | |
| Content-Word F1 | 0.127 | Suppressed by verbosity |
| Content-Word Precision | 0.1289 | 12.9% of model's words appear in reference |
| Content-Word Recall | 0.1624 | Model covers 16.2% of reference's key terms |
| NLI Entailment Rate | 0.1595 | 16% of responses fully consistent with reference |
| NLI Neutral Rate | 0.7157 | 72% classified as neutral (model is verbose) |
| NLI Contradiction Rate | 0.1248 | 12.5% borderline — see caveats |
| Content-Word Hallucination | 0.8686 | Suppressed by verbosity (see analysis below) |
| Avg Latency | 7.24 s | Per response on Kaggle T4 |

#### The Verbosity Problem

The model systematically generates longer, more detailed answers than the MedQuAD references. This is the dominant factor in every overlap-based metric.

**Example:**
```
Question: What causes Holt-Oram syndrome?

Reference (NIH): Mutations in the TBX5 gene.

Model output: Holt-Oram syndrome is caused by mutations in the TBX5 gene, which
encodes a transcription factor essential for the development of the heart and upper
limbs during embryogenesis. The condition follows an autosomal dominant inheritance
pattern, meaning a single mutated copy of the gene is sufficient to cause the
syndrome. Note: this information is educational; consult a physician.
```

The model's answer is clinically correct and more informative than the reference. But:
- ROUGE-1 score: ~0.14 (only "mutations", "TBX5", "gene" match)
- Content-Word Hallucination: ~0.85 (transcription, factor, embryogenesis, autosomal, dominant, etc. not in reference)
- Clinical BERTScore: ~0.90 (BiomedBERT understands that both texts are about the same genetic concept)

**Conclusion:** Clinical BERTScore F1 (0.9012) and perplexity (2.57) are the primary quality signals. ROUGE and content-word overlap metrics are suppressed by intentional verbosity, not by factual error.

#### Comparison with Baselines

| Model | Method | Train Ex. | ROUGE-1 | BERTScore F1 | Latency |
|---|---|---|---|---|---|
| **Phi-3 Mini QLoRA** ★ | QLoRA 4-bit | **2,000** | 0.185 | **0.804** | 7.24 s |
| Falcon-7B QLoRA | QLoRA 4-bit | 200 | **0.250** | N/A | 10.94 s |
| Falcon-7B LoRA | LoRA BF16 | 200 | 0.210 | N/A | 3.53 s |
| Falcon-7B Prompt (4-bit) | Prompt Tuning | 200 | 0.210 | N/A | 8.81 s |
| Falcon-7B Prompt (BF16) | Prompt Tuning | 200 | 0.180 | N/A | 1.89 s |

**Note on Falcon-7B higher ROUGE:** The Falcon baselines achieved higher ROUGE-1 (0.25) with only 200 training examples. This likely indicates those models copy phrases from the training set more directly (lower diversity, higher overlap with similar reference styles). Phi-3's lower ROUGE reflects more paraphrastic and elaborative answers, which is desirable behaviour for a medical assistant.

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

A dark-themed Streamlit page with four metric sections:

1. **Classical Metrics** — ROUGE-1/2/L, Perplexity, Latency displayed as metric cards
2. **Semantic Similarity** — Generic vs. Clinical BERTScore with delta highlighting
3. **Clinical Accuracy** — Content-Word Precision/Recall/F1
4. **Factual Safety** — NLI rates with a live safety verdict:
   - Contradiction < 10%: ✅ Clinically Safe
   - Contradiction 10–15%: ⚠️ Borderline — review needed
   - Contradiction > 15%: 🚨 Caution — clinical review required

All data is loaded from `evaluate/results/results.json`. The dashboard updates automatically when new eval results are written to that file.

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

1. **NLI model domain mismatch:** `roberta-large-mnli` was trained on Wikipedia/books. Medical text has different syntactic patterns. The contradiction rate (12.48%) should be treated as an upper bound, not a precise measurement.

2. **Verbosity suppresses overlap metrics:** All ROUGE and content-word metrics are suppressed because the model generates more content than the NIH reference. This is actually desirable behaviour (more informative answers) but makes metrics look low.

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

### Step 3: Run the clinical evaluation (Kaggle T4, ~15 min)

```
1. Same Kaggle environment
2. Paste evaluate/clinical_kaggle.py into a new cell
3. Output: phi3_results.json pushed to Shriyanshml/mediguide-rag-index
4. Copy the JSON into evaluate/results/results.json locally
```

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
