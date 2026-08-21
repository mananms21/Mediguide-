"""
╔══════════════════════════════════════════════════════════════════╗
║      MEDIGUIDE — Phi-3 Mini QLoRA Fine-tuning Script             ║
║                                                                  ║
║  Platform  : Kaggle T4 GPU (free tier)                           ║
║  Runtime   : ~60–90 minutes                                      ║
║  Dataset   : MedQuAD (2,000 train / 200 eval examples)           ║
║  Base Model: microsoft/Phi-3-mini-4k-instruct (3.8B)             ║
║  Output    : Shriyanshml/phi3-mini-qlora-mediguide               ║
║                                                                  ║
║  ⚠️  BEFORE RUNNING:                                              ║
║    1. Enable GPU (Settings → Accelerator → GPU T4 x1)            ║
║    2. Add HF_TOKEN as a Kaggle Secret                            ║
║       (Add-ons → Secrets → New Secret → name: HF_TOKEN)         ║
╚══════════════════════════════════════════════════════════════════╝
"""

# ── STEP 0: Install / upgrade packages ────────────────────────────
import subprocess, sys, shutil, pathlib

print("📦 Installing packages…")
subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q",
     "bitsandbytes>=0.46.1",
     "torchao>=0.16.0",
     "trl==0.11.4",            
     "transformers==4.46.3",   
     "peft>=0.12.0",
     "accelerate>=0.34.0",
     "datasets>=2.21.0",
     "rouge-score",
     "bert-score",
     "sentence-transformers>=2.7.0",
     "faiss-gpu",
     "huggingface_hub>=0.24.0",
    ],
    check=True,
)

# 🔥 CRITICAL KAGGLE FIX: Force Python to forget Kaggle's pre-loaded packages
for mod in list(sys.modules.keys()):
    if any(name in mod for name in ['transformers', 'peft', 'trl']):
        del sys.modules[mod]
sys.path.insert(0, '/usr/local/lib/python3.12/dist-packages')

# Clear cached modeling_phi3.py
modules_cache = pathlib.Path.home() / ".cache" / "huggingface" / "modules"
if modules_cache.exists():
    shutil.rmtree(modules_cache)
    print("🗑️  Cleared HF modules cache")

print("✅ Packages ready\n")


# ── STEP 1: Imports ───────────────────────────────────────────────
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0" 
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import json, time, pickle, warnings, gc
import numpy as np
import faiss
import torch
import pandas as pd
from datasets import Dataset, load_dataset
import transformers
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
from transformers.trainer_utils import get_last_checkpoint
import trl
from trl import SFTTrainer, SFTConfig
from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
)
from huggingface_hub import login, HfApi
from sentence_transformers import SentenceTransformer
from rouge_score import rouge_scorer as rouge_lib

warnings.filterwarnings("ignore")


# ── STEP 2: Configuration ─────────────────────────────────────────
HF_USERNAME   = "Shriyanshml"

# Read HF token from Kaggle Secrets (not os.environ — Kaggle uses its own API)
try:
    from kaggle_secrets import UserSecretsClient
    _secrets  = UserSecretsClient()
    HF_TOKEN  = _secrets.get_secret("HF_Token")   # must match exactly what you named it
except Exception:
    HF_TOKEN  = os.environ.get("HF_TOKEN") or os.environ.get("HF_Token")  # fallback

BASE_MODEL_ID = "microsoft/Phi-3-mini-4k-instruct"
OUTPUT_MODEL  = f"{HF_USERNAME}/phi3-mini-qlora-mediguide"
RAG_DATASET   = f"{HF_USERNAME}/mediguide-rag-index"

TRAIN_SIZE    = 2000
EVAL_SIZE     = 200
MAX_SEQ_LEN   = 512
NUM_EPOCHS    = 3
LR            = 2e-4
BATCH_SIZE    = 4
GRAD_ACCUM    = 4

SYSTEM_PROMPT = (
    "You are MEDIGUIDE, a knowledgeable medical assistant trained on "
    "authoritative NIH sources. Provide accurate, evidence-based answers "
    "to medical questions in a clear, empathetic tone. Always end your "
    "response with a brief disclaimer that this information is educational "
    "and patients should consult a qualified healthcare professional for "
    "personal medical advice."
)

# Authenticate
if not HF_TOKEN:
    raise EnvironmentError(
        "HF token not found.\n"
        "Make sure:\n"
        "  1. You added the secret via Add-ons → Secrets\n"
        "  2. The secret name matches exactly: 'HF_Token'\n"
        "  3. You ticked the checkbox to attach it to this notebook\n"
    )
login(token=HF_TOKEN)
print(f"✅ Logged in to HuggingFace as {HF_USERNAME}\n")


# ── STEP 3: Load & Preprocess MedQuAD ────────────────────────────
def load_medquad() -> pd.DataFrame:
    try:
        print("📥 Trying HuggingFace Hub: keivalya/MedQuad-MedicalQnADataset…")
        raw    = load_dataset("keivalya/MedQuad-MedicalQnADataset")
        frames = [raw[s].to_pandas() for s in raw.keys()]
        df     = pd.concat(frames, ignore_index=True)
        df.columns = [c.lower() for c in df.columns]
        if "question" not in df.columns:
            df = df.rename(columns={"questions": "question", "answers": "answer",
                                    "input": "question", "output": "answer"})
        print(f"   ✅ Loaded {len(df):,} rows from HF Hub")
        return df
    except Exception as e:
        print(f"   ⚠️  HF Hub failed: {e}")

    import glob, os
    kaggle_pattern = "/kaggle/input/medquad*/**/*.csv"
    csv_files = glob.glob(kaggle_pattern, recursive=True)
    if csv_files:
        print(f"📥 Found Kaggle input files: {csv_files[:3]}")
        frames = [pd.read_csv(f) for f in csv_files]
        df     = pd.concat(frames, ignore_index=True)
        df.columns = [c.lower() for c in df.columns]
        if "question" not in df.columns:
            df = df.rename(columns={"questions": "question", "answers": "answer"})
        print(f"   ✅ Loaded {len(df):,} rows from Kaggle input")
        return df

    raise RuntimeError(
        "Could not load MedQuAD dataset.\n"
        "Option 1 (recommended): It should have loaded from HF Hub automatically.\n"
        "Option 2: In your Kaggle notebook, click Data → Add Input → search 'medquad pythonafroz' → Add."
    )

df = load_medquad()
print(f"   Columns available: {list(df.columns)}")

# Clean
df = df.dropna(subset=["question", "answer"])
df["question"] = df["question"].str.strip()
df["answer"]   = df["answer"].str.strip()
df = df[df["answer"].str.len() > 80]
df = df[df["question"].str.len() > 10]
df = df.drop_duplicates(subset=["question"]).reset_index(drop=True)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
print(f"   After cleaning: {len(df):,} rows\n")

assert len(df) >= TRAIN_SIZE + EVAL_SIZE, (
    f"Not enough rows! Got {len(df)}, need {TRAIN_SIZE + EVAL_SIZE}."
)

train_df = df.iloc[:TRAIN_SIZE].copy()
eval_df  = df.iloc[TRAIN_SIZE : TRAIN_SIZE + EVAL_SIZE].copy()
print(f"   Train: {len(train_df):,} | Eval: {len(eval_df):,}")


def phi3_format(question: str, answer: str) -> str:
    """Phi-3 instruct chat template."""
    return (
        f"<|system|>\n{SYSTEM_PROMPT}<|end|>\n"
        f"<|user|>\n{question}<|end|>\n"
        f"<|assistant|>\n{answer}<|end|>"
    )

train_texts = [phi3_format(r.question, r.answer) for r in train_df.itertuples()]
eval_texts  = [phi3_format(r.question, r.answer) for r in eval_df.itertuples()]

train_dataset = Dataset.from_dict({"text": train_texts})
eval_dataset  = Dataset.from_dict({"text": eval_texts})


# ── STEP 4: Load Phi-3 Mini & Apply LoRA ───────────────────────────
from transformers import BitsAndBytesConfig
from peft import prepare_model_for_kbit_training

print(f"🤖 Loading {BASE_MODEL_ID} in 4-bit QLoRA…")

# 1. Configure 4-bit quantization (Shrinks model to ~2.5 GB)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_ID,
    quantization_config=bnb_config,
    torch_dtype=torch.float16,
    device_map={"": 0},
    attn_implementation="eager",
)
model.config.use_cache = False

# 2. Prepare the model for 4-bit training (Handles gradient checkpointing automatically)
model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
tokenizer.pad_token    = tokenizer.unk_token
tokenizer.padding_side = "right"

print(f"   GPU memory used: {torch.cuda.memory_allocated()/1e9:.1f} GB / 15 GB\n")

# 3. Apply LoRA to the Phi-3 fused attention layer
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["qkv_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()


# ── STEP 6: Training ──────────────────────────────────────────────
import inspect
print("\n🚀 Starting LoRA training…")

_sft_sig = set(inspect.signature(SFTConfig.__init__).parameters)

_base_cfg = dict(
    output_dir="./phi3-mediguide-checkpoints",
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=16,
    warmup_steps=int(0.03 * (TRAIN_SIZE // 16) * NUM_EPOCHS),
    learning_rate=LR,
    fp16=True,
    logging_steps=25,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    report_to="none",
    optim="paged_adamw_8bit",
)
if "max_seq_length"      in _sft_sig: _base_cfg["max_seq_length"]      = MAX_SEQ_LEN
if "dataset_text_field" in _sft_sig: _base_cfg["dataset_text_field"] = "text"
if "packing"            in _sft_sig: _base_cfg["packing"]            = False

if "max_seq_length" not in _sft_sig:
    tokenizer.model_max_length = MAX_SEQ_LEN

sft_config = SFTConfig(**_base_cfg)

_trainer_sig = set(inspect.signature(SFTTrainer.__init__).parameters)
_trainer_kw = dict(
    model=model,
    args=sft_config,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)
if "processing_class" in _trainer_sig:
    _trainer_kw["processing_class"] = tokenizer
elif "tokenizer" in _trainer_sig:
    _trainer_kw["tokenizer"] = tokenizer

if "dataset_text_field" in _trainer_sig and "dataset_text_field" not in _base_cfg:
    _trainer_kw["dataset_text_field"] = "text"
if "packing" in _trainer_sig and "packing" not in _base_cfg:
    _trainer_kw["packing"] = False
if "max_seq_length" in _trainer_sig and "max_seq_length" not in _base_cfg:
    _trainer_kw["max_seq_length"] = MAX_SEQ_LEN

trainer = SFTTrainer(**_trainer_kw)

# Flush any orphaned memory right before starting the massive training loop
gc.collect()
torch.cuda.empty_cache()

# ── CHECKPOINT RESUME LOGIC ──
last_checkpoint = None
if os.path.isdir(_base_cfg["output_dir"]):
    last_checkpoint = get_last_checkpoint(_base_cfg["output_dir"])
    if last_checkpoint is not None:
        print(f"🔄 Resuming training from checkpoint: {last_checkpoint}")
    else:
        print("🆕 Starting training from scratch.")

# 🔥 FIX FOR PYTORCH 2.6+ CHECKPOINT RESUMING
# Temporarily revert torch.load so it can read Numpy RNG states
_original_load = torch.load
def _legacy_load(*args, **kwargs):
    kwargs["weights_only"] = False
    return _original_load(*args, **kwargs)
torch.load = _legacy_load

train_start = time.time()
if last_checkpoint is not None:
    trainer.train(resume_from_checkpoint=last_checkpoint)
else:
    trainer.train()

# Restore standard torch.load behavior
torch.load = _original_load

train_time  = time.time() - train_start
print(f"✅ Training complete! ({train_time/60:.1f} min)\n")

# ── STEP 7: ROUGE Evaluation ──────────────────────────────────────
print("📊 Running ROUGE evaluation on 50 eval examples…")

model.eval()
device  = "cuda" if torch.cuda.is_available() else "cpu"
scorer  = rouge_lib.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

r1_list, r2_list, rL_list, lat_list = [], [], [], []

for row in eval_df.head(50).itertuples():
    prompt = (
        f"<|system|>\n{SYSTEM_PROMPT}<|end|>\n"
        f"<|user|>\n{row.question}<|end|>\n"
        f"<|assistant|>\n"
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    t0 = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=150,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.2,
        )
    latency = time.time() - t0

    generated = tokenizer.decode(
        outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True
    ).strip()

    s = scorer.score(row.answer, generated)
    r1_list.append(s["rouge1"].fmeasure)
    r2_list.append(s["rouge2"].fmeasure)
    rL_list.append(s["rougeL"].fmeasure)
    lat_list.append(latency)

eval_results = {
    "model":           OUTPUT_MODEL,
    "base_model":      BASE_MODEL_ID,
    "method":          "QLoRA (4-bit NF4)",
    "train_examples":  TRAIN_SIZE,
    "eval_examples":   50,
    "rouge1":          round(float(np.mean(r1_list)), 4),
    "rouge2":          round(float(np.mean(r2_list)), 4),
    "rougeL":          round(float(np.mean(rL_list)), 4),
    "latency_s":       round(float(np.mean(lat_list)), 2),
    "adapter_size_mb": None,
    "bertscore_f1":    None,
    "perplexity":      None,
}

print("\n📈 Results:")
for k, v in eval_results.items():
    print(f"   {k:20s}: {v}")

with open("phi3_qlora_results.json", "w") as f:
    json.dump(eval_results, f, indent=2)


# ── STEP 8: Build FAISS RAG Index ─────────────────────────────────
print("\n🧠 Building FAISS index from full MedQuAD dataset…")

all_docs = [
    {
        "question":   str(r.question).strip(),
        "answer":     str(r.answer).strip(),
        "source":     str(getattr(r, "source", "MedQuAD")),
        "focus_area": str(getattr(r, "focus_area", "General")),
    }
    for r in df.itertuples()
]

encoder   = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
questions = [d["question"] for d in all_docs]

print(f"   Encoding {len(all_docs):,} documents…")
embeddings = encoder.encode(
    questions,
    show_progress_bar=True,
    batch_size=128,
    normalize_embeddings=True,
).astype(np.float32)

faiss.normalize_L2(embeddings)
dim   = embeddings.shape[1]
index = faiss.IndexFlatIP(dim)
index.add(embeddings)

os.makedirs("rag_index", exist_ok=True)
faiss.write_index(index, "rag_index/faiss_index.bin")
with open("rag_index/medquad_docs.pkl", "wb") as f:
    pickle.dump(all_docs, f)

print(f"✅ FAISS index: {index.ntotal:,} vectors of dim {dim}\n")


# ── STEP 9: Push Everything to HuggingFace Hub ───────────────────
print(f"📤 Pushing model to {OUTPUT_MODEL}…")
trainer.model.push_to_hub(OUTPUT_MODEL, private=False)
tokenizer.push_to_hub(OUTPUT_MODEL, private=False)
print("✅ Model pushed!")

api = HfApi()

print(f"\n📤 Pushing RAG index to {RAG_DATASET}…")
api.create_repo(RAG_DATASET, repo_type="dataset", exist_ok=True, private=False)
for fname, fpath in [
    ("faiss_index.bin",    "rag_index/faiss_index.bin"),
    ("medquad_docs.pkl",   "rag_index/medquad_docs.pkl"),
    ("phi3_results.json",  "phi3_qlora_results.json"),
]:
    api.upload_file(
        path_or_fileobj=fpath,
        path_in_repo=fname,
        repo_id=RAG_DATASET,
        repo_type="dataset",
    )
    print(f"   ✅ {fname}")

print(f"""
╔══════════════════════════════════════════════════════════╗
║  🎉 ALL DONE!                                           ║
║                                                         ║
║  Model : https://huggingface.co/{OUTPUT_MODEL:<25s}  ║
║  Index : https://huggingface.co/datasets/{RAG_DATASET}  ║
║                                                         ║
║  Next steps (on your local machine):                    ║
║    1. python rag/build_index.py --mode download         ║
║    2. streamlit run app/app.py                          ║
║    3. Deploy spaces/ folder to HF Spaces                ║
╚══════════════════════════════════════════════════════════╝
""")