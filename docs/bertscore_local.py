"""
MEDIGUIDE — Local BERTScore Evaluation (CPU-friendly)
Runs on Mac without GPU. Uses 5 examples to keep runtime ≈ 10 min.

Usage:
    HF_TOKEN=hf_xxx python evaluate/bertscore_local.py
"""

import json, os, sys, time
from pathlib import Path

ROOT = Path(__file__).parent.parent

# ── Auth ──────────────────────────────────────────────────────────
HF_TOKEN = os.environ.get("HF_TOKEN", "")
if HF_TOKEN:
    from huggingface_hub import login
    login(token=HF_TOKEN, add_to_git_credential=False)

MODEL_ID  = "Shriyanshml/phi3-mini-qlora-mediguide"
N_SAMPLES = 5          # keep small — 3.8B on CPU is ~2 min/sample
MAX_TOKENS = 80

SYSTEM_PROMPT = (
    "You are MEDIGUIDE, a knowledgeable medical assistant. "
    "Answer the question briefly and accurately."
)

# ── Load eval examples ────────────────────────────────────────────
print("📥 Loading eval examples from MedQuAD…")
from datasets import load_dataset
import pandas as pd

raw    = load_dataset("keivalya/MedQuad-MedicalQnADataset")
df     = raw["train"].to_pandas()
df.columns = [c.lower() for c in df.columns]
df     = df.dropna(subset=["question", "answer"])
df     = df[df["answer"].str.len() > 80].sample(N_SAMPLES * 5, random_state=42)
df     = df.drop_duplicates("question").head(N_SAMPLES)
eval_set = df[["question", "answer"]].to_dict("records")
print(f"   ✅ {len(eval_set)} eval examples loaded")

# ── Load model ────────────────────────────────────────────────────
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftConfig, PeftModel

# Pick best available device (MPS for Apple Silicon, else CPU)
if torch.backends.mps.is_available():
    DEVICE     = "mps"
    DTYPE      = torch.float16
    device_map = {"": "mps"}
    print("⚡ Using Apple MPS (Metal) backend")
else:
    DEVICE     = "cpu"
    DTYPE      = torch.float32
    device_map = {"": "cpu"}
    print("🐢 Using CPU (slow but works — ~2 min/sample)")

print(f"\n🤖 Loading {MODEL_ID}…")
peft_cfg   = PeftConfig.from_pretrained(MODEL_ID)
base_id    = peft_cfg.base_model_name_or_path

model = AutoModelForCausalLM.from_pretrained(
    base_id,
    dtype=DTYPE,
    device_map=device_map,
    attn_implementation="eager",
)
model = PeftModel.from_pretrained(model, MODEL_ID, device_map=device_map)
model.eval()

tok = AutoTokenizer.from_pretrained(base_id)
tok.pad_token = tok.unk_token
print("✅ Model loaded\n")

# ── Generation ────────────────────────────────────────────────────
def generate(question: str) -> tuple[str, float]:
    prompt = (
        f"<|system|>\n{SYSTEM_PROMPT}<|end|>\n"
        f"<|user|>\n{question}<|end|>\n"
        f"<|assistant|>\n"
    )
    inputs = tok(prompt, return_tensors="pt").to(DEVICE)
    t0 = time.time()
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=MAX_TOKENS,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tok.eos_token_id,
            repetition_penalty=1.2,
        )
    latency = time.time() - t0
    answer  = tok.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
    return answer, latency

preds, refs, latencies = [], [], []
for i, sample in enumerate(eval_set, 1):
    print(f"  [{i}/{N_SAMPLES}] Generating…", end=" ", flush=True)
    pred, lat = generate(sample["question"])
    preds.append(pred)
    refs.append(sample["answer"])
    latencies.append(lat)
    print(f"✓ ({lat:.1f}s)  |  {pred[:80]}…")

# ── BERTScore ─────────────────────────────────────────────────────
print("\n📐 Computing BERTScore (bert-base-uncased, CPU)…")
from bert_score import score as bs_score
P, R, F1 = bs_score(preds, refs, lang="en", verbose=True, device="cpu")

bertscore_p  = round(float(P.mean()), 4)
bertscore_r  = round(float(R.mean()), 4)
bertscore_f1 = round(float(F1.mean()), 4)

# ── ROUGE ─────────────────────────────────────────────────────────
print("📐 Computing ROUGE…")
import numpy as np
from rouge_score import rouge_scorer as rs

scorer = rs.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
r1, r2, rL = [], [], []
for p, r in zip(preds, refs):
    s = scorer.score(r, p)
    r1.append(s["rouge1"].fmeasure)
    r2.append(s["rouge2"].fmeasure)
    rL.append(s["rougeL"].fmeasure)

rouge1 = round(float(np.mean(r1)), 4)
rouge2 = round(float(np.mean(r2)), 4)
rougeL = round(float(np.mean(rL)), 4)
avg_lat = round(float(np.mean(latencies)), 2)

# ── Report ────────────────────────────────────────────────────────
print(f"""
╔══════════════════════════════════════════╗
║  BERTScore Results — Phi-3 Mini QLoRA   ║
╠══════════════════════════════════════════╣
║  BERTScore P  : {bertscore_p:<8}               ║
║  BERTScore R  : {bertscore_r:<8}               ║
║  BERTScore F1 : {bertscore_f1:<8}  ← key metric  ║
║  ROUGE-1      : {rouge1:<8}               ║
║  ROUGE-2      : {rouge2:<8}               ║
║  ROUGE-L      : {rougeL:<8}               ║
║  Avg latency  : {avg_lat:<8}s              ║
║  Samples      : {N_SAMPLES:<8}               ║
╚══════════════════════════════════════════╝
""")

# ── Update results.json ───────────────────────────────────────────
results_path = ROOT / "evaluate" / "results" / "results.json"
with open(results_path) as f:
    data = json.load(f)

for m in data["models"]:
    if m["model_id"] == MODEL_ID:
        m["bertscore_p"]  = bertscore_p
        m["bertscore_r"]  = bertscore_r
        m["bertscore_f1"] = bertscore_f1
        m["rouge1"]       = rouge1
        m["rouge2"]       = rouge2
        m["rougeL"]       = rougeL
        m["latency_s"]    = avg_lat
        m["eval_examples"] = N_SAMPLES
        print(f"✅ Updated results.json for {m['name']}")
        break

from datetime import datetime
data["last_updated"] = datetime.utcnow().isoformat()
with open(results_path, "w") as f:
    json.dump(data, f, indent=2)

print(f"\n📄 Saved → {results_path}")
print("\n💡 Re-run with N_SAMPLES=50 on Kaggle for full evaluation.")
