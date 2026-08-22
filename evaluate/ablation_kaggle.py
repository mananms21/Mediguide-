"""
MEDIGUIDE — Comprehensive Ablation Evaluation  (v2)
====================================================

4 conditions, same 5-metric suite:
  1. Zero-shot     — base Phi-3 Mini, no adapter, no RAG
  2. Fine-tuned    — Phi-3 Mini + QLoRA adapter, no RAG
  3. + RAG         — Phi-3 Mini + QLoRA + on-the-fly FAISS retrieval
  4. OOD           — Fine-tuned on PubMedQA (external benchmark, different domain)

Metrics:
  Classical : ROUGE-1/2/L (full predictions)
              ROUGE-1     (truncated @50 tokens)   ← fixes verbosity bias
  Semantic  : Clinical BERTScore F1  (BiomedBERT manual greedy matching)
              Generic BERTScore F1   (roberta-large)
  Lexical   : Lexical Precision@50   (content-word precision on trunc preds)
  Safety    : NLI Contradiction Rate (roberta-large-mnli)
  Meta      : Perplexity | Latency

Run on  : Kaggle T4 GPU (15 GB VRAM)
Runtime : ~50 min total
"""

# ── STEP 0: Install ────────────────────────────────────────────────────
import subprocess, sys, warnings
warnings.filterwarnings("ignore")

print("📦 Installing dependencies…")
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
    "bert-score", "rouge-score", "datasets",
    "peft>=0.12.0", "accelerate>=0.34.0",
    "sentence-transformers>=2.7.0",
    "faiss-cpu",
], check=True)
print("✅ Packages ready\n")


# ── STEP 1: Imports & Config ───────────────────────────────────────────
import os, json, time, re, gc
import numpy as np
import torch
import faiss

from datasets import load_dataset, Dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, AutoModel,
    AutoTokenizer as _AuxTok,
    AutoModelForSequenceClassification,
    BitsAndBytesConfig,
)
from peft import PeftModel
from rouge_score import rouge_scorer as rouge_lib
from bert_score import score as generic_bs
from sentence_transformers import SentenceTransformer
from huggingface_hub import login, HfApi

DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"
HF_USERNAME   = "Shriyanshml"
BASE_MODEL    = "microsoft/Phi-3-mini-4k-instruct"
ADAPTER_ID    = f"{HF_USERNAME}/phi3-mini-qlora-mediguide"
RAG_DATASET   = f"{HF_USERNAME}/mediguide-rag-index"
N_EVAL        = 50   # samples per condition
TOP_K_RAG     = 3
RAG_CHAR_CAP  = 1500
TRUNC_TOKENS  = 50   # for ROUGE@50tok and Lexical Precision@50

print(f"🖥️  Device: {DEVICE} | VRAM: "
      f"{torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB\n")

SYSTEM_PROMPT = (
    "You are MEDIGUIDE, a knowledgeable medical assistant trained on "
    "authoritative NIH sources. Provide accurate, evidence-based answers "
    "to medical questions in a clear, empathetic tone. Always end your "
    "response with a brief disclaimer that this information is educational "
    "and patients should consult a qualified healthcare professional for "
    "personal medical advice."
)


# ── STEP 2: Authenticate ───────────────────────────────────────────────
print("🔑 Authenticating…")
try:
    from kaggle_secrets import UserSecretsClient
    HF_TOKEN = UserSecretsClient().get_secret("HF_Token")
except Exception:
    HF_TOKEN = os.environ.get("HF_TOKEN") or os.environ.get("HF_Token")
if not HF_TOKEN:
    raise EnvironmentError("Add HF_Token to Kaggle Secrets (Add-ons → Secrets)")
login(token=HF_TOKEN)


# ── STEP 3: Load MedQuAD — same preprocessing as training ─────────────
print("📥 Loading MedQuAD…")
raw = load_dataset("keivalya/MedQuad-MedicalQnADataset")
df = raw["train"].to_pandas()
df.columns = [c.lower() for c in df.columns]
df = df.dropna(subset=["question", "answer"])
df["question"] = df["question"].str.strip()
df["answer"]   = df["answer"].str.strip()
df = df[df["answer"].str.len() > 80]
df = df[df["question"].str.len() > 10]
df = df.drop_duplicates(subset=["question"]).reset_index(drop=True)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Test set: 50 samples AFTER the 2,200 used for train+eval (same seed as training)
test_df = df.iloc[2200 : 2200 + N_EVAL].copy().reset_index(drop=True)
mq_questions = test_df["question"].tolist()
mq_refs      = test_df["answer"].tolist()
print(f"   ✅ {N_EVAL} MedQuAD test samples (from index 2200+, no train overlap)\n")


# ── STEP 4: Load PubMedQA — external OOD benchmark ────────────────────
print("📥 Loading PubMedQA (OOD benchmark)…")
pqa = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train")
pqa_sub = pqa.shuffle(seed=42).select(range(N_EVAL))
pqa_questions = [x["question"] for x in pqa_sub]
pqa_refs      = [x["long_answer"] for x in pqa_sub]
print(f"   ✅ {N_EVAL} PubMedQA samples loaded\n")


# ── STEP 5: Build on-the-fly RAG index (test-excluded) ────────────────
print("🗂️  Building RAG index from MedQuAD (test samples excluded)…")
test_indices = set(test_df.index.tolist())
rag_df = df[~df.index.isin(test_indices)].reset_index(drop=True)

embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
rag_questions = rag_df["question"].tolist()
rag_answers   = rag_df["answer"].tolist()

print(f"   Embedding {len(rag_df):,} passages…")
rag_embs = embedder.encode(rag_questions, batch_size=512,
                           show_progress_bar=True, device=DEVICE)
rag_embs = rag_embs.astype("float32")

rag_index = faiss.IndexFlatL2(rag_embs.shape[1])
rag_index.add(rag_embs)
print(f"   ✅ FAISS index ready ({len(rag_df):,} passages, dim={rag_embs.shape[1]})\n")

def retrieve_context(query: str) -> str:
    q_emb = embedder.encode([query]).astype("float32")
    _, idxs = rag_index.search(q_emb, TOP_K_RAG)
    passages = [
        f"Q: {rag_questions[i]}\nA: {rag_answers[i]}"
        for i in idxs[0]
    ]
    return "\n\n".join(passages)[:RAG_CHAR_CAP]


# ── STEP 6: Load Phi-3 base model (4-bit) ─────────────────────────────
print("🤖 Loading base Phi-3 Mini (4-bit NF4)…")
bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb,
    torch_dtype=torch.float16,
    device_map={"": 0},
    attn_implementation="eager",
)
model.config.use_cache = False
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
tokenizer.pad_token    = tokenizer.unk_token
tokenizer.padding_side = "right"
EOS_ID = tokenizer.convert_tokens_to_ids("<|end|>")
print(f"   GPU after base load: {torch.cuda.memory_allocated()/1e9:.1f} GB\n")


# ── Generation helper ─────────────────────────────────────────────────
def generate_responses(questions, refs, context_fn=None, label=""):
    """
    Run inference on a list of questions.
    Returns: preds_full, preds_trunc (first TRUNC_TOKENS tokens), ppls, lats
    """
    preds_full, preds_trunc, ppls, lats = [], [], [], []
    print(f"🔮 [{label}] Generating {len(questions)} responses…")
    for i, (q, ref) in enumerate(zip(questions, refs), 1):
        # Build prompt
        if context_fn is not None:
            ctx = context_fn(q)
            user_content = f"[CONTEXT FROM NIH MEDQUAD]\n{ctx}\n\n[QUESTION]\n{q}"
        else:
            user_content = q
        prompt = (
            f"<|system|>\n{SYSTEM_PROMPT}<|end|>\n"
            f"<|user|>\n{user_content}<|end|>\n"
            f"<|assistant|>\n"
        )
        input_ids = tokenizer(prompt, return_tensors="pt",
                              truncation=True, max_length=2048).input_ids.to(DEVICE)
        t0 = time.time()
        with torch.no_grad():
            out = model.generate(
                input_ids,
                max_new_tokens=300,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.unk_token_id,
                eos_token_id=EOS_ID,
            )
        lat = time.time() - t0

        gen_ids = out[0, input_ids.shape[1]:]
        full_text  = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
        trunc_text = tokenizer.decode(gen_ids[:TRUNC_TOKENS],
                                       skip_special_tokens=True).strip()

        # Perplexity on generated tokens only
        with torch.no_grad():
            labels = out.clone()
            labels[0, :input_ids.shape[1]] = -100
            ppl = torch.exp(model(out, labels=labels).loss).item()

        preds_full.append(full_text)
        preds_trunc.append(trunc_text)
        ppls.append(min(ppl, 1000))   # cap runaway values
        lats.append(lat)

        if i == 1 or i % 10 == 0:
            print(f"  [{i:02d}/{len(questions)}] {lat:.1f}s | ppl={ppl:.1f} | "
                  f"{full_text[:65]}…")

    avg_ppl = round(float(np.mean(ppls)), 2)
    avg_lat = round(float(np.mean(lats)), 2)
    print(f"   ✅ Done — avg latency: {avg_lat}s | avg perplexity: {avg_ppl}\n")
    return preds_full, preds_trunc, ppls, lats


# ── STEP 7: Condition 1 — Zero-shot (base model, no adapter, no RAG) ──
print("=" * 65)
print("CONDITION 1 — Zero-shot (base Phi-3, no adapter, no RAG)")
print("=" * 65)
zs_full, zs_trunc, zs_ppls, zs_lats = generate_responses(
    mq_questions, mq_refs, context_fn=None, label="ZERO-SHOT"
)


# ── STEP 8: Condition 2 — Fine-tuned (apply LoRA adapter) ─────────────
print("=" * 65)
print("CONDITION 2 — Fine-tuned (QLoRA adapter, no RAG)")
print("=" * 65)
print("🔌 Applying LoRA adapter…")
model = PeftModel.from_pretrained(model, ADAPTER_ID)
model.eval()
print(f"   GPU after adapter: {torch.cuda.memory_allocated()/1e9:.1f} GB\n")

ft_full, ft_trunc, ft_ppls, ft_lats = generate_responses(
    mq_questions, mq_refs, context_fn=None, label="FINE-TUNED"
)


# ── STEP 9: Condition 3 — Fine-tuned + RAG ────────────────────────────
print("=" * 65)
print("CONDITION 3 — Fine-tuned + RAG")
print("=" * 65)
rag_full, rag_trunc, rag_ppls, rag_lats = generate_responses(
    mq_questions, mq_refs, context_fn=retrieve_context, label="FINE-TUNED+RAG"
)


# ── STEP 10: Condition 4 — OOD on PubMedQA ───────────────────────────
print("=" * 65)
print("CONDITION 4 — OOD: PubMedQA (external benchmark)")
print("=" * 65)
ood_full, ood_trunc, ood_ppls, ood_lats = generate_responses(
    pqa_questions, pqa_refs, context_fn=None, label="OOD-PubMedQA"
)


# ── STEP 11: Unload generation model ──────────────────────────────────
print("🗑️  Unloading generation model to free GPU…")
del model
gc.collect()
torch.cuda.empty_cache()
print(f"   GPU after unload: {torch.cuda.memory_allocated()/1e9:.1f} GB\n")


# ── STEP 12: Classical Metrics ─────────────────────────────────────────
print("📐 [1/5] Classical Metrics (ROUGE full + ROUGE@50tok)…")
rouge = rouge_lib.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

def compute_rouge(preds, refs):
    r1, r2, rL = [], [], []
    for p, r in zip(preds, refs):
        s = rouge.score(r, p)
        r1.append(s["rouge1"].fmeasure)
        r2.append(s["rouge2"].fmeasure)
        rL.append(s["rougeL"].fmeasure)
    return round(np.mean(r1), 4), round(np.mean(r2), 4), round(np.mean(rL), 4)

zs_r1,  zs_r2,  zs_rL  = compute_rouge(zs_full,  mq_refs)
ft_r1,  ft_r2,  ft_rL  = compute_rouge(ft_full,  mq_refs)
rag_r1, rag_r2, rag_rL = compute_rouge(rag_full, mq_refs)
ood_r1, ood_r2, ood_rL = compute_rouge(ood_full, pqa_refs)

# ROUGE@50tok (verbosity-corrected)
zs_r1t,  _, _ = compute_rouge(zs_trunc,  mq_refs)
ft_r1t,  _, _ = compute_rouge(ft_trunc,  mq_refs)
rag_r1t, _, _ = compute_rouge(rag_trunc, mq_refs)
ood_r1t, _, _ = compute_rouge(ood_trunc, pqa_refs)

print(f"   Zero-shot   — R1={zs_r1}  R@50={zs_r1t}")
print(f"   Fine-tuned  — R1={ft_r1}  R@50={ft_r1t}")
print(f"   + RAG       — R1={rag_r1}  R@50={rag_r1t}")
print(f"   OOD         — R1={ood_r1}  R@50={ood_r1t}")


# ── STEP 13: Lexical Precision@50 ────────────────────────────────────
print("\n📐 [2/5] Lexical Precision@50 (content-word precision on trunc preds)…")

_STOP = {
    "the","a","an","is","are","was","were","be","been","being",
    "have","has","had","do","does","did","will","would","could",
    "should","may","might","shall","can","cannot","not","no","yes",
    "and","or","but","if","then","that","this","these","those",
    "with","for","from","by","at","to","in","on","of","as","it",
    "its","they","them","their","we","our","you","your","he","she",
    "his","her","i","my","me","who","which","what","when","where",
    "how","all","both","each","few","more","most","other","some",
    "such","also","often","usually","generally","however","therefore",
    "although","though","while","since","because","about","after",
    "before","during","between","among","above","below","than","so",
    "very","just","even","well","many","much","any","here","there",
    "patient","patients","disease","condition","symptom","symptoms",
    "treatment","treatments","diagnosis","cause","causes","related",
    "include","includes","including","common","type","types","called",
    "known","used","based","found","associated","certain","occur",
    "occurs","help","helps","note","notes","information","medical",
    "health","care","doctor","physician","please","see","refer",
    "disclaimer","professional","advice","consult",
}

def content_words(text: str) -> set:
    text = re.sub(r'[^\w\s]', ' ', text.lower())
    return {t for t in text.split()
            if t not in _STOP and len(t) > 2 and not t.isdigit()}

def lexical_prec(preds_trunc, refs) -> float:
    scores = []
    for p, r in zip(preds_trunc, refs):
        p_w = content_words(p)
        r_w = content_words(r)
        if not p_w:
            scores.append(0.0)
        elif not r_w:
            scores.append(1.0)
        else:
            scores.append(len(p_w & r_w) / len(p_w))
    return round(float(np.mean(scores)), 4)

zs_lp  = lexical_prec(zs_trunc,  mq_refs)
ft_lp  = lexical_prec(ft_trunc,  mq_refs)
rag_lp = lexical_prec(rag_trunc, mq_refs)
ood_lp = lexical_prec(ood_trunc, pqa_refs)

print(f"   Zero-shot={zs_lp}  Fine-tuned={ft_lp}  +RAG={rag_lp}  OOD={ood_lp}")


# ── STEP 14: Generic BERTScore (roberta-large) ────────────────────────
# Use TRUNCATED predictions to avoid verbosity suppressing scores
print("\n📐 [3/5] Generic BERTScore (roberta-large, on truncated@50tok preds)…")

def safe_bertscore(preds, refs):
    short_p = [p[:1500] for p in preds]
    short_r = [r[:1500] for r in refs]
    P, R, F = generic_bs(short_p, short_r, lang="en", device=DEVICE, verbose=False)
    return round(float(P.mean()), 4), round(float(R.mean()), 4), round(float(F.mean()), 4)

zs_bp,  zs_br,  zs_bf  = safe_bertscore(zs_trunc,  mq_refs)
ft_bp,  ft_br,  ft_bf  = safe_bertscore(ft_trunc,  mq_refs)
rag_bp, rag_br, rag_bf = safe_bertscore(rag_trunc, mq_refs)
ood_bp, ood_br, ood_bf = safe_bertscore(ood_trunc, pqa_refs)
torch.cuda.empty_cache()

print(f"   Zero-shot={zs_bf}  Fine-tuned={ft_bf}  +RAG={rag_bf}  OOD={ood_bf}")


# ── STEP 15: Clinical BERTScore (BiomedBERT manual greedy matching) ───
print("\n📐 [4/5] Clinical BERTScore (BiomedBERT manual greedy matching)…")
import torch.nn.functional as F

BIOMEDBERT = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"
_bio_tok = _AuxTok.from_pretrained(BIOMEDBERT, use_fast=False)
_bio_mdl = AutoModel.from_pretrained(BIOMEDBERT).to(DEVICE).eval()
print(f"   BiomedBERT loaded ({sum(p.numel() for p in _bio_mdl.parameters())/1e6:.0f}M params)")

def _bio_embed(texts, max_length=400):
    embs = []
    for text in texts:
        enc = _bio_tok(text, return_tensors="pt", truncation=True,
                       max_length=max_length, padding=False).to(DEVICE)
        with torch.no_grad():
            out = _bio_mdl(**enc)
        token_embs = out.last_hidden_state[0, 1:-1, :]
        if token_embs.shape[0] == 0:
            token_embs = out.last_hidden_state[0, :1, :]
        embs.append(token_embs)
    return embs

def clinical_bertscore(preds, refs):
    p_embs = _bio_embed(preds)
    r_embs = _bio_embed(refs)
    Ps, Rs, F1s = [], [], []
    for pe, re in zip(p_embs, r_embs):
        pe_n = F.normalize(pe, dim=-1)
        re_n = F.normalize(re, dim=-1)
        sim  = torch.mm(pe_n, re_n.T)
        P  = sim.max(dim=1).values.mean().item()
        R  = sim.max(dim=0).values.mean().item()
        f1 = 2 * P * R / (P + R) if (P + R) > 0 else 0.0
        Ps.append(P); Rs.append(R); F1s.append(f1)
    return (round(float(np.mean(Ps)),  4),
            round(float(np.mean(Rs)),  4),
            round(float(np.mean(F1s)), 4))

zs_cp,  zs_cr,  zs_cf  = clinical_bertscore(zs_full,  mq_refs)
ft_cp,  ft_cr,  ft_cf  = clinical_bertscore(ft_full,  mq_refs)
rag_cp, rag_cr, rag_cf = clinical_bertscore(rag_full, mq_refs)
ood_cp, ood_cr, ood_cf = clinical_bertscore(ood_full, pqa_refs)

del _bio_mdl; torch.cuda.empty_cache()
print(f"   Zero-shot={zs_cf}  Fine-tuned={ft_cf}  +RAG={rag_cf}  OOD={ood_cf}")
print(f"   Fine-tuning Δ: {ft_cf - zs_cf:+.4f}  |  RAG Δ: {rag_cf - ft_cf:+.4f}  |"
      f"  OOD gap: {ft_cf - ood_cf:+.4f}")


# ── STEP 16: NLI Contradiction Rate ──────────────────────────────────
print("\n📐 [5/5] NLI Contradiction Rate (roberta-large-mnli)…")
NLI_MODEL = "roberta-large-mnli"
nli_tok = AutoTokenizer.from_pretrained(NLI_MODEL)
nli_mdl = AutoModelForSequenceClassification.from_pretrained(NLI_MODEL).to(DEVICE).eval()
torch.cuda.empty_cache()

def nli_batch(preds, refs):
    """Returns (entailment, neutral, contradiction) rates."""
    e_list, n_list, c_list = [], [], []
    for pred, ref in zip(preds, refs):
        enc = nli_tok(ref[:512], pred[:512], return_tensors="pt",
                      truncation=True, max_length=512).to(DEVICE)
        with torch.no_grad():
            probs = torch.softmax(nli_mdl(**enc).logits, dim=-1)[0]
        # roberta-large-mnli label order: 0=CONTRADICTION, 1=NEUTRAL, 2=ENTAILMENT
        c_list.append(probs[0].item())
        n_list.append(probs[1].item())
        e_list.append(probs[2].item())
    return (round(float(np.mean(e_list)), 4),
            round(float(np.mean(n_list)), 4),
            round(float(np.mean(c_list)), 4))

zs_ent,  zs_neu,  zs_con  = nli_batch(zs_full,  mq_refs)
ft_ent,  ft_neu,  ft_con  = nli_batch(ft_full,  mq_refs)
rag_ent, rag_neu, rag_con = nli_batch(rag_full, mq_refs)
ood_ent, ood_neu, ood_con = nli_batch(ood_full, pqa_refs)

del nli_mdl; torch.cuda.empty_cache()
print(f"   Zero-shot   — Contradiction={zs_con}")
print(f"   Fine-tuned  — Contradiction={ft_con}")
print(f"   + RAG       — Contradiction={rag_con}")
print(f"   OOD         — Contradiction={ood_con}")


# ── STEP 17: Comprehensive Report ────────────────────────────────────
print(f"""
╔═══════════════════════════════════════════════════════════════════════════╗
║       MEDIGUIDE — Ablation Report  |  {N_EVAL} samples per condition          ║
╠══════════════════════════╦═══════════╦════════════╦═════════╦════════════╣
║  Metric                  ║ Zero-shot ║ Fine-tuned ║  + RAG  ║ OOD(PubMQ) ║
╠══════════════════════════╬═══════════╬════════════╬═════════╬════════════╣
║  Clinical BERTScore F1   ║  {zs_cf:<8} ║ {ft_cf:<9}  ║ {rag_cf:<6}  ║ {ood_cf:<9}  ║
║  Generic  BERTScore F1   ║  {zs_bf:<8} ║ {ft_bf:<9}  ║ {rag_bf:<6}  ║ {ood_bf:<9}  ║
╠══════════════════════════╬═══════════╬════════════╬═════════╬════════════╣
║  ROUGE-1  (full)         ║  {zs_r1:<8} ║ {ft_r1:<9}  ║ {rag_r1:<6}  ║ {ood_r1:<9}  ║
║  ROUGE-1  (@50 tok)      ║  {zs_r1t:<8} ║ {ft_r1t:<9}  ║ {rag_r1t:<6}  ║ {ood_r1t:<9}  ║
║  Lexical Precision@50    ║  {zs_lp:<8} ║ {ft_lp:<9}  ║ {rag_lp:<6}  ║ {ood_lp:<9}  ║
╠══════════════════════════╬═══════════╬════════════╬═════════╬════════════╣
║  NLI Contradiction       ║  {zs_con:<8} ║ {ft_con:<9}  ║ {rag_con:<6}  ║ {ood_con:<9}  ║
╠══════════════════════════╬═══════════╬════════════╬═════════╬════════════╣
║  Perplexity (avg)        ║  {round(np.mean(zs_ppls),2):<8} ║ {round(np.mean(ft_ppls),2):<9}  ║ {round(np.mean(rag_ppls),2):<6}  ║ {round(np.mean(ood_ppls),2):<9}  ║
║  Latency s/sample (avg)  ║  {round(np.mean(zs_lats),2):<8} ║ {round(np.mean(ft_lats),2):<9}  ║ {round(np.mean(rag_lats),2):<6}  ║ {round(np.mean(ood_lats),2):<9}  ║
╚══════════════════════════╩═══════════╩════════════╩═════════╩════════════╝

💡 Key findings:
   Fine-tuning Δ (Clinical BERTScore): {ft_cf - zs_cf:+.4f}
   RAG Δ (Clinical BERTScore):         {rag_cf - ft_cf:+.4f}
   OOD gap vs in-distribution:         {ft_cf - ood_cf:+.4f}
   Fine-tuning reduces contradiction:  {zs_con - ft_con:+.4f}
""")


# ── STEP 18: Save & push results ────────────────────────────────────
results = {
    "last_updated": time.strftime("%Y-%m-%d"),
    "eval_samples_per_condition": N_EVAL,
    "conditions": {
        "medquad_zeroshot": {
            "label": "Zero-shot (base Phi-3, no adapter)",
            "dataset": "MedQuAD",
            "clinical_bertscore_f1": zs_cf, "clinical_bertscore_p": zs_cp, "clinical_bertscore_r": zs_cr,
            "bertscore_f1": zs_bf, "bertscore_p": zs_bp, "bertscore_r": zs_br,
            "rouge1": zs_r1, "rouge2": zs_r2, "rougeL": zs_rL,
            "rouge1_50tok": zs_r1t,
            "lexical_precision_50": zs_lp,
            "nli_entailment": zs_ent, "nli_neutral": zs_neu, "nli_contradiction": zs_con,
            "perplexity": round(float(np.mean(zs_ppls)), 2),
            "latency_s": round(float(np.mean(zs_lats)), 2),
        },
        "medquad_finetuned": {
            "label": "Fine-tuned QLoRA (no RAG)",
            "dataset": "MedQuAD",
            "clinical_bertscore_f1": ft_cf, "clinical_bertscore_p": ft_cp, "clinical_bertscore_r": ft_cr,
            "bertscore_f1": ft_bf, "bertscore_p": ft_bp, "bertscore_r": ft_br,
            "rouge1": ft_r1, "rouge2": ft_r2, "rougeL": ft_rL,
            "rouge1_50tok": ft_r1t,
            "lexical_precision_50": ft_lp,
            "nli_entailment": ft_ent, "nli_neutral": ft_neu, "nli_contradiction": ft_con,
            "perplexity": round(float(np.mean(ft_ppls)), 2),
            "latency_s": round(float(np.mean(ft_lats)), 2),
        },
        "medquad_rag": {
            "label": "Fine-tuned QLoRA + RAG",
            "dataset": "MedQuAD",
            "clinical_bertscore_f1": rag_cf, "clinical_bertscore_p": rag_cp, "clinical_bertscore_r": rag_cr,
            "bertscore_f1": rag_bf, "bertscore_p": rag_bp, "bertscore_r": rag_br,
            "rouge1": rag_r1, "rouge2": rag_r2, "rougeL": rag_rL,
            "rouge1_50tok": rag_r1t,
            "lexical_precision_50": rag_lp,
            "nli_entailment": rag_ent, "nli_neutral": rag_neu, "nli_contradiction": rag_con,
            "perplexity": round(float(np.mean(rag_ppls)), 2),
            "latency_s": round(float(np.mean(rag_lats)), 2),
        },
        "pubmedqa_finetuned": {
            "label": "Fine-tuned QLoRA — OOD on PubMedQA",
            "dataset": "PubMedQA (pqa_labeled)",
            "ood_note": "External benchmark: research-style questions, different domain from training",
            "clinical_bertscore_f1": ood_cf, "clinical_bertscore_p": ood_cp, "clinical_bertscore_r": ood_cr,
            "bertscore_f1": ood_bf, "bertscore_p": ood_bp, "bertscore_r": ood_br,
            "rouge1": ood_r1, "rouge2": ood_r2, "rougeL": ood_rL,
            "rouge1_50tok": ood_r1t,
            "lexical_precision_50": ood_lp,
            "nli_entailment": ood_ent, "nli_neutral": ood_neu, "nli_contradiction": ood_con,
            "perplexity": round(float(np.mean(ood_ppls)), 2),
            "latency_s": round(float(np.mean(ood_lats)), 2),
        },
    },
    "deltas": {
        "finetuning_clinical_bertscore": round(ft_cf - zs_cf, 4),
        "rag_clinical_bertscore": round(rag_cf - ft_cf, 4),
        "ood_gap_clinical_bertscore": round(ft_cf - ood_cf, 4),
        "finetuning_contradiction_reduction": round(zs_con - ft_con, 4),
        "rag_contradiction_reduction": round(ft_con - rag_con, 4),
    }
}

with open("ablation_results.json", "w") as f:
    json.dump(results, f, indent=2)

api = HfApi()
api.upload_file(
    path_or_fileobj="ablation_results.json",
    path_in_repo="ablation_results.json",
    repo_id=RAG_DATASET,
    repo_type="dataset",
    commit_message="Ablation results: zero-shot, fine-tuned, +RAG, OOD (PubMedQA)",
)
print("✅ ablation_results.json pushed to HF Hub")
print("📌 Copy contents into evaluate/results/ablation_results.json locally.")
