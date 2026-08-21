"""
MEDIGUIDE — Full Clinical Evaluation on Kaggle T4
==================================================
Runs all evaluation levels in one cell. Paste entirely into Kaggle.

Metrics computed:
  Existing : ROUGE-1/2/L, latency, perplexity, generic BERTScore
  New      : Clinical BERTScore (BiomedBERT), NER Entity F1,
             NLI Contradiction Rate, Hallucination Rate
"""

# ── STEP 0: Install ───────────────────────────────────────────────────
import subprocess, sys

print("📦 Installing dependencies…")
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
    "bert-score", "rouge-score", "datasets",
    "peft>=0.12.0", "accelerate>=0.34.0",
    "torchao>=0.16.0",
    # scispacy: scientific/medical NER — REQUIRED for entity F1 & hallucination metrics
    # en_core_web_md (general English) does NOT recognize medical entities
    "scispacy",
], check=True)

# Install scispacy medium scientific model (~100 MB)
# This model tags scientific entities generically (type="ENTITY") covering
# diseases, drugs, chemicals, anatomy — exactly what we need
print("📥 Installing scispacy en_core_sci_md (medical NER)…")
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
    "https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.3/en_core_sci_md-0.5.3.tar.gz"
], check=True)
print("✅ Packages ready\n")

# ── STEP 1: Imports ───────────────────────────────────────────────────
import json, time, torch, warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

from datasets import load_dataset
from rouge_score import rouge_scorer as rs
from bert_score import score as generic_bs
from transformers import (
    AutoModelForCausalLM, AutoTokenizer,
    AutoModelForSequenceClassification,
)
from peft import PeftModel, PeftConfig
from kaggle_secrets import UserSecretsClient
from huggingface_hub import login, HfApi

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🖥️  Device: {DEVICE} | VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB\n"
      if DEVICE == "cuda" else f"🖥️  Device: {DEVICE}\n")

# ── STEP 2: Authenticate & Load Fine-tuned Model ──────────────────────
print("🔑 Authenticating…")
login(token=UserSecretsClient().get_secret("HF_Token"))

MODEL_ID = "Shriyanshml/phi3-mini-qlora-mediguide"
print(f"🤖 Loading {MODEL_ID}…")

peft_cfg = PeftConfig.from_pretrained(MODEL_ID)
gen_model = AutoModelForCausalLM.from_pretrained(
    peft_cfg.base_model_name_or_path,
    torch_dtype=torch.float16,
    device_map={"": 0},  # Forced to GPU 0 to prevent cross-device Kaggle errors
    attn_implementation="eager",
)
gen_model = PeftModel.from_pretrained(gen_model, MODEL_ID)
gen_model.eval()

tokenizer = AutoTokenizer.from_pretrained(peft_cfg.base_model_name_or_path)
tokenizer.pad_token = tokenizer.unk_token
print(f"   ✅ Model loaded | GPU: {torch.cuda.memory_allocated()/1e9:.1f} GB used\n")

# ── STEP 3: Load Dataset ──────────────────────────────────────────────
N_EVAL  = 50
MAX_TOK = 150

SYSTEM_PROMPT = (
    "You are MEDIGUIDE, a knowledgeable medical assistant trained on "
    "authoritative NIH sources. Provide accurate, evidence-based answers "
    "to medical questions in a clear, empathetic tone. Always end your "
    "response with a brief disclaimer that this is not a substitute for "
    "professional medical advice."
)

print("📥 Loading MedQuAD…")
raw = load_dataset("keivalya/MedQuad-MedicalQnADataset")
df  = raw["train"].to_pandas()
df.columns = [c.lower() for c in df.columns]
df  = df.dropna(subset=["question", "answer"])
df  = df[df["answer"].str.len() > 80]
sample = df.sample(N_EVAL, random_state=99).to_dict("records")
print(f"   ✅ {N_EVAL} eval examples sampled\n")

# ── STEP 4: Generation + Perplexity ──────────────────────────────────
def generate_and_score(question, reference):
    """Returns (prediction, latency_s, perplexity)."""
    prompt = (
        f"<|system|>\n{SYSTEM_PROMPT}<|end|>\n"
        f"<|user|>\n{question}<|end|>\n"
        f"<|assistant|>\n"
    )

    # ── Generation ─────────────────────────────────────────────────
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    t0 = time.time()
    with torch.no_grad():
        out = gen_model.generate(
            **inputs,
            max_new_tokens=MAX_TOK,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.2,
        )
    latency = time.time() - t0
    pred = tokenizer.decode(
        out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True
    ).strip()

    # ── Perplexity: model probability of the reference given question ─
    # Lower = model assigns higher probability to the correct answer
    full_text = prompt + reference + tokenizer.eos_token
    full_enc  = tokenizer(full_text, return_tensors="pt").to(DEVICE)
    n_prompt  = inputs.input_ids.shape[1]
    labels    = full_enc.input_ids.clone()
    labels[0, :n_prompt] = -100  # mask prompt — only score reference tokens

    with torch.no_grad():
        loss = gen_model(**full_enc, labels=labels).loss
    perplexity = torch.exp(loss).item()

    return pred, latency, perplexity


print(f"🔮 Generating {N_EVAL} responses…")
questions, preds, refs, latencies, perplexities = [], [], [], [], []

for i, row in enumerate(sample, 1):
    pred, lat, ppl = generate_and_score(row["question"], row["answer"])
    questions.append(row["question"])
    preds.append(pred)
    refs.append(row["answer"])
    latencies.append(lat)
    perplexities.append(min(ppl, 1000.0))  # cap outliers

    if i % 10 == 0 or i == 1:
        print(f"  [{i:02d}/{N_EVAL}] {lat:.1f}s | ppl={ppl:.1f} | {pred[:65]}…")

avg_lat = round(float(np.mean(latencies)), 2)
avg_ppl = round(float(np.mean(perplexities)), 2)
print(f"\n✅ Generation done — avg latency: {avg_lat}s | avg perplexity: {avg_ppl}\n")

# ── STEP 5: ROUGE ─────────────────────────────────────────────────────
print("📐 [1/6] ROUGE…")
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
print(f"   ROUGE-1={rouge1}  ROUGE-2={rouge2}  ROUGE-L={rougeL}")

# ── STEP 6: Generic BERTScore (roberta-large baseline) ────────────────
print("\n📐 [2/6] Generic BERTScore (roberta-large)…")

# Truncate texts to ~2000 chars to avoid BERT's strict 512 token limit
short_preds = [p[:2000] for p in preds]
short_refs  = [r[:2000] for r in refs]

P, R, F1 = generic_bs(short_preds, short_refs, lang="en", verbose=True, device=DEVICE)
bertscore_p  = round(float(P.mean()), 4)
bertscore_r  = round(float(R.mean()), 4)
bertscore_f1 = round(float(F1.mean()), 4)
print(f"   P={bertscore_p}  R={bertscore_r}  F1={bertscore_f1}")

# ── STEP 7: Clinical BERTScore (BiomedBERT) ───────────────────────────
print("\n📐 [3/6] Clinical BERTScore (BiomedBERT — clinically aware)…")
BIOMEDBERT = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"
P_c, R_c, F1_c = generic_bs(
    short_preds, short_refs,   # <-- Use truncated strings here too!
    model_type=BIOMEDBERT,
    num_layers=12,
    lang="en", verbose=True,
    device=DEVICE,
    rescale_with_baseline=False,
)
clinical_bertscore_p  = round(float(P_c.mean()), 4)
clinical_bertscore_r  = round(float(R_c.mean()), 4)
clinical_bertscore_f1 = round(float(F1_c.mean()), 4)
print(f"   P={clinical_bertscore_p}  R={clinical_bertscore_r}  F1={clinical_bertscore_f1}")
print(f"   Δ vs generic BERTScore: {clinical_bertscore_f1 - bertscore_f1:+.4f}")
# ── STEP 8: Medical NER Entity F1 (scispacy) ──────────────────────────
# IMPORTANT: en_core_web_md (general English) does NOT work here —
# it recognises PERSON/ORG/DATE, not medical entities. Use scispacy.
print("\n📐 [4/6] Medical NER Entity F1 (scispacy en_core_sci_md)…")
import spacy

try:
    nlp = spacy.load("en_core_sci_md")
    NER_BACKEND = "scispacy:en_core_sci_md"
except OSError:
    # Fallback: if scispacy failed to install, use a curated medical
    # keyword extractor based on common biomedical term patterns.
    # Still far better than en_core_web_md for medical F1.
    import re
    print("   ⚠️  scispacy not available — using regex medical keyword fallback")
    NER_BACKEND = "regex-medical-fallback"
    nlp = None

# Common non-medical words that general NER wrongly tags as entities
_STOP = {"the","a","an","is","are","was","were","be","been","have",
         "has","had","do","does","did","will","would","could","should",
         "may","might","shall","can","not","no","yes","and","or","but",
         "if","then","that","this","these","those","with","for","from",
         "by","at","to","in","on","of","as","it","its"}

def extract_ents(text):
    if nlp is not None:
        doc = nlp(text[:5000])
        return {ent.text.lower().strip() for ent in doc.ents
                if len(ent.text.strip()) > 1 and ent.text.lower() not in _STOP}
    else:
        # Regex fallback: multi-word sequences of title-case or known medical words
        # Captures patterns like "Type 2 Diabetes", "aortic valve", etc.
        tokens = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Za-z]+){0,3}\b", text)
        return {t.lower() for t in tokens if t.lower() not in _STOP and len(t) > 3}

print(f"   NER backend: {NER_BACKEND}")

ep_list, er_list, ef_list = [], [], []
for pred, ref in zip(preds, refs):
    pe = extract_ents(pred)
    re_ = extract_ents(ref)
    if not pe and not re_:
        ep_list.append(1.0); er_list.append(1.0); ef_list.append(1.0)
        continue
    if not pe:
        ep_list.append(0.0); er_list.append(0.0); ef_list.append(0.0)
        continue
    if not re_:
        ep_list.append(1.0); er_list.append(0.0); ef_list.append(0.0)
        continue
    tp = len(pe & re_)
    p  = tp / len(pe)
    r  = tp / len(re_)
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    ep_list.append(p); er_list.append(r); ef_list.append(f1)

entity_precision = round(float(np.mean(ep_list)), 4)
entity_recall    = round(float(np.mean(er_list)), 4)
entity_f1        = round(float(np.mean(ef_list)), 4)
print(f"   Precision={entity_precision}  Recall={entity_recall}  F1={entity_f1}")

# ── STEP 9: Hallucination Rate ────────────────────────────────────────
print("\n📐 [5/6] Hallucination / Specificity Score…")
hall_rates = []
for q, p, r in zip(questions, preds, refs):
    pred_ents  = extract_ents(p)
    known_ents = extract_ents(q) | extract_ents(r)
    if not pred_ents:
        hall_rates.append(0.0)
    else:
        grounded = len(pred_ents & known_ents)
        hall_rates.append(1.0 - grounded / len(pred_ents))
hallucination_rate = round(float(np.mean(hall_rates)), 4)
print(f"   Hallucination Rate: {hallucination_rate:.4f}  "
      f"(Specificity: {1-hallucination_rate:.4f})")

# ── STEP 10: NLI Factual Consistency ─────────────────────────────────
print("\n📐 [6/6] NLI Factual Consistency (roberta-large-mnli)…")
print("   Loading NLI model (~1.4 GB)…")
NLI_MODEL = "roberta-large-mnli"
nli_tok = AutoTokenizer.from_pretrained(NLI_MODEL)
nli_mdl = AutoModelForSequenceClassification.from_pretrained(NLI_MODEL).to(DEVICE).eval()

# Free up generation model memory before loading NLI
torch.cuda.empty_cache()

def nli_predict(premise, hypothesis):
    """premise=reference (truth), hypothesis=prediction"""
    enc = nli_tok(
        premise, hypothesis,
        return_tensors="pt", truncation=True, max_length=512
    ).to(DEVICE)
    with torch.no_grad():
        logits = nli_mdl(**enc).logits
    probs = torch.softmax(logits, dim=-1)[0]
    # roberta-large-mnli label order: 0=CONTRADICTION, 1=NEUTRAL, 2=ENTAILMENT
    return probs[0].item(), probs[1].item(), probs[2].item()

cont_list, neut_list, ent_list = [], [], []
for pred, ref in zip(preds, refs):
    c, n, e = nli_predict(ref[:1024], pred[:1024])
    cont_list.append(c); neut_list.append(n); ent_list.append(e)

contradiction_rate = round(float(np.mean(cont_list)), 4)
neutral_rate       = round(float(np.mean(neut_list)), 4)
entailment_rate    = round(float(np.mean(ent_list)),  4)
print(f"   Entailment={entailment_rate}  Neutral={neutral_rate}  Contradiction={contradiction_rate}")

# ── STEP 11: Print Full Report ────────────────────────────────────────
print(f"""
╔══════════════════════════════════════════════════════════════╗
║     MEDIGUIDE — Full Clinical Evaluation Report              ║
║     Model: Phi-3 Mini QLoRA  |  Samples: {N_EVAL}               ║
╠══════════════════════════════════════════════════════════════╣
║  CLASSICAL METRICS                                           ║
║   ROUGE-1         : {rouge1:<8}  ROUGE-2   : {rouge2:<8}     ║
║   ROUGE-L         : {rougeL:<8}  Perplexity: {avg_ppl:<8}     ║
║   Latency         : {avg_lat}s/sample                         ║
╠══════════════════════════════════════════════════════════════╣
║  SEMANTIC SIMILARITY                                         ║
║   Generic BERTScore F1  : {bertscore_f1:<8} (roberta-large)      ║
║   Clinical BERTScore F1 : {clinical_bertscore_f1:<8} (BiomedBERT)     ║
║   Delta                 : {clinical_bertscore_f1 - bertscore_f1:+.4f}   (↓ = harder to fake) ║
╠══════════════════════════════════════════════════════════════╣
║  CLINICAL ACCURACY (NEW)                                     ║
║   NER Entity Precision : {entity_precision:<8}                  ║
║   NER Entity Recall    : {entity_recall:<8}                  ║
║   NER Entity F1        : {entity_f1:<8}                  ║
╠══════════════════════════════════════════════════════════════╣
║  FACTUAL SAFETY (NEW)                                        ║
║   Entailment Rate    : {entailment_rate:<8} (consistent answers)    ║
║   Neutral Rate       : {neutral_rate:<8} (incomplete but safe) ║
║   Contradiction Rate : {contradiction_rate:<8} ← clinical danger    ║
║   Hallucination Rate : {hallucination_rate:<8} (novel entities)    ║
╚══════════════════════════════════════════════════════════════╝

💡 Interpretation:
   Contradiction rate < 0.10 → clinically safe
   Entity F1 > 0.50          → good term accuracy
   Clinical BERTScore ≈ Generic → model uses general language (not domain jargon)
   Clinical BERTScore << Generic → model used wrong clinical terms
""")

# ── STEP 12: Build results dict & push ───────────────────────────────
results = {
    "model":          "Shriyanshml/phi3-mini-qlora-mediguide",
    "base_model":     "microsoft/Phi-3-mini-4k-instruct",
    "method":         "QLoRA (4-bit NF4)",
    "train_examples": 2000,
    "eval_examples":  N_EVAL,
    # Classical
    "rouge1":         rouge1,
    "rouge2":         rouge2,
    "rougeL":         rougeL,
    "perplexity":     avg_ppl,
    "latency_s":      avg_lat,
    # Semantic
    "bertscore_p":    bertscore_p,
    "bertscore_r":    bertscore_r,
    "bertscore_f1":   bertscore_f1,
    # Clinical semantic
    "clinical_bertscore_p":  clinical_bertscore_p,
    "clinical_bertscore_r":  clinical_bertscore_r,
    "clinical_bertscore_f1": clinical_bertscore_f1,
    # Clinical accuracy
    "entity_precision":   entity_precision,
    "entity_recall":      entity_recall,
    "entity_f1":          entity_f1,
    # Factual safety
    "entailment_rate":    entailment_rate,
    "neutral_rate":       neutral_rate,
    "contradiction_rate": contradiction_rate,
    # Hallucination
    "hallucination_rate": hallucination_rate,
}

with open("phi3_results.json", "w") as f:
    json.dump(results, f, indent=2)

api = HfApi()
api.upload_file(
    path_or_fileobj="phi3_results.json",
    path_in_repo="phi3_results.json",
    repo_id="Shriyanshml/mediguide-rag-index",
    repo_type="dataset",
    commit_message="Full clinical evaluation — BiomedBERT + NER + NLI + hallucination",
)
print("✅ phi3_results.json pushed to HF Hub")
print("\n📌 Copy phi3_results.json contents into evaluate/results/results.json locally.")