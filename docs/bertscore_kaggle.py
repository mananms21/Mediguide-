"""
MEDIGUIDE — BERTScore Evaluation on Kaggle T4
Run this as a NEW cell AFTER the training cell has completed, or in a fresh kernel.
"""

# ── Packages ──
import subprocess, sys
print("📦 Installing evaluation dependencies...")
subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q", 
     "bert-score", 
     "rouge-score",
     "torchao>=0.16.0",       
     "peft>=0.12.0",          
     "transformers==4.46.3",
     "datasets"
    ], 
    check=True
)

# 🔥 CRITICAL KAGGLE FIX: Force Python to use the newly installed versions
for mod in list(sys.modules.keys()):
    if any(name in mod for name in ['transformers', 'peft', 'torchao', 'datasets']):
        del sys.modules[mod]
sys.path.insert(0, '/usr/local/lib/python3.12/dist-packages')

# ── Imports ──
import json, time, torch
import numpy as np
from datasets import load_dataset
from bert_score import score as bs_score
from rouge_score import rouge_scorer as rs
import pandas as pd
from kaggle_secrets import UserSecretsClient
from huggingface_hub import login
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig

# ── Config ────────────────────────────────────────────────────────────
N_EVAL     = 50          # number of eval examples
MAX_TOKENS = 150         # tokens to generate per answer
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"

SYSTEM_PROMPT = (
    "You are MEDIGUIDE, a knowledgeable medical assistant trained on "
    "authoritative NIH sources. Provide accurate, evidence-based answers "
    "to medical questions in a clear, empathetic tone. Always end your "
    "response with a brief disclaimer that this is not a substitute for "
    "professional medical advice."
)

# ── Authenticate & Load Model (Fresh Kernel Safe) ─────────────────────
print("\n🔑 Authenticating...")
login(token=UserSecretsClient().get_secret("HF_Token"))

MODEL_ID = "Shriyanshml/phi3-mini-qlora-mediguide"
print(f"🤖 Loading adapter from {MODEL_ID} and its base model...")

peft_cfg = PeftConfig.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    peft_cfg.base_model_name_or_path,
    torch_dtype=torch.float16,      # <-- FIXED: changed dtype to torch_dtype
    device_map="auto", 
    attn_implementation="eager"
)
model = PeftModel.from_pretrained(model, MODEL_ID)

tokenizer = AutoTokenizer.from_pretrained(peft_cfg.base_model_name_or_path)
tokenizer.pad_token = tokenizer.unk_token
model.eval()

# ── Load Dataset ──────────────────────────────────────────────────────
print("\n📥 Loading MedQuAD Dataset...")
raw = load_dataset("keivalya/MedQuad-MedicalQnADataset")
eval_df = raw["train"].to_pandas()
eval_df.columns = [c.lower() for c in eval_df.columns]
eval_df = eval_df.dropna(subset=["question","answer"])
eval_df = eval_df[eval_df["answer"].str.len() > 80]

# Sample eval set
sample = eval_df.sample(N_EVAL, random_state=99).to_dict("records")

# ── Generation ────────────────────────────────────────────────────────
def generate(question: str) -> tuple[str, float]:
    prompt = (
        f"<|system|>\n{SYSTEM_PROMPT}<|end|>\n"
        f"<|user|>\n{question}<|end|>\n"
        f"<|assistant|>\n"
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    t0 = time.time()
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=MAX_TOKENS,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.2,
        )
    latency = time.time() - t0
    answer = tokenizer.decode(
        out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True
    ).strip()
    return answer, latency

print(f"\n🔮 Generating {N_EVAL} responses on {DEVICE}…")
preds, refs, latencies = [], [], []

for i, row in enumerate(sample, 1):
    pred, lat = generate(row["question"])
    preds.append(pred)
    refs.append(row["answer"])
    latencies.append(lat)
    if i % 10 == 0 or i == 1:
        print(f"  [{i:02d}/{N_EVAL}] {lat:.1f}s | {pred[:70]}…")

avg_lat = round(float(np.mean(latencies)), 2)
print(f"\n✅ Generation done — avg latency: {avg_lat}s/sample")

# ── ROUGE ─────────────────────────────────────────────────────────────
print("\n📐 Computing ROUGE…")
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

# ── BERTScore ─────────────────────────────────────────────────────────
print("📐 Computing BERTScore (bert-base-uncased)…")
P, R, F1 = bs_score(preds, refs, lang="en", verbose=True, device=DEVICE)

bertscore_p  = round(float(P.mean()), 4)
bertscore_r  = round(float(R.mean()), 4)
bertscore_f1 = round(float(F1.mean()), 4)

# ── Print results ─────────────────────────────────────────────────────
print(f"""
╔═══════════════════════════════════════════════╗
║   MEDIGUIDE — Phi-3 Mini QLoRA Evaluation     ║
╠═══════════════════════════════════════════════╣
║  BERTScore P  : {bertscore_p:<8}                 ║
║  BERTScore R  : {bertscore_r:<8}                  ║
║  BERTScore F1 : {bertscore_f1:<8}  ← key metric   ║
╠═══════════════════════════════════════════════╣
║  ROUGE-1      : {rouge1:<8}                  ║
║  ROUGE-2      : {rouge2:<8}                  ║
║  ROUGE-L      : {rougeL:<8}                  ║
╠═══════════════════════════════════════════════╣
║  Avg latency  : {avg_lat:<8}s                ║
║  Eval samples : {N_EVAL:<8}                  ║
╚═══════════════════════════════════════════════╝
""")

# ── Save phi3_results.json and push to HF ─────────────────────────────
results = {
    "model":          "Shriyanshml/phi3-mini-qlora-mediguide",
    "base_model":     "microsoft/Phi-3-mini-4k-instruct",
    "method":         "QLoRA (4-bit NF4)",
    "train_examples": 2000,
    "eval_examples":  N_EVAL,
    "rouge1":         rouge1,
    "rouge2":         rouge2,
    "rougeL":         rougeL,
    "bertscore_p":    bertscore_p,
    "bertscore_r":    bertscore_r,
    "bertscore_f1":   bertscore_f1,
    "latency_s":      avg_lat,
}

with open("phi3_results.json", "w") as f:
    json.dump(results, f, indent=2)

# Push updated results to HF dataset
from huggingface_hub import HfApi
api = HfApi()
api.upload_file(
    path_or_fileobj="phi3_results.json",
    path_in_repo="phi3_results.json",
    repo_id="Shriyanshml/mediguide-rag-index",
    repo_type="dataset",
    commit_message="Add full BERTScore evaluation results",
)
print("✅ phi3_results.json pushed to Shriyanshml/mediguide-rag-index")