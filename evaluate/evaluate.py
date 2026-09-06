"""
MEDIGUIDE — Evaluation Script
Computes ROUGE-1/2/L and BERTScore for all models on the MedQuAD test set.

⚠️  Requires GPU. Run on Kaggle T4 after training.

Usage:
    # Evaluate all models (writes to evaluate/results/results.json):
    python evaluate/evaluate.py --all

    # Evaluate a single model:
    python evaluate/evaluate.py --model phi3

    # Just print the current results table (no GPU needed):
    python evaluate/evaluate.py --table
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# ── Model registry ────────────────────────────────────────────────
MODEL_REGISTRY = {
    "phi3": {
        "name":          "Phi-3 Mini QLoRA",
        "model_id":      "Shriyanshml/phi3-mini-qlora-mediguide",
        "base_model":    "microsoft/Phi-3-mini-4k-instruct",
        "method":        "QLoRA (4-bit NF4)",
        "type":          "phi3",
        "train_examples": 2000,
        "adapter_size_mb": 12.6,
    },
}


RESULTS_PATH = Path(__file__).parent / "results" / "results.json"
EVAL_SAMPLES  = 50

SYSTEM_PROMPT = (
    "You are MEDIGUIDE, a knowledgeable medical assistant trained on "
    "authoritative NIH sources. Provide accurate, evidence-based answers "
    "to medical questions in a clear, empathetic tone. Always end your "
    "response with a brief disclaimer that this information is educational "
    "and patients should consult a qualified healthcare professional."
)


# ── Dataset loading ───────────────────────────────────────────────

def load_eval_set(n: int = EVAL_SAMPLES) -> list[dict]:
    """Load n evaluation examples from MedQuAD."""
    from datasets import load_dataset

    raw = load_dataset(
        "pythonafroz/medquad-medical-question-answer-for-ai-research",
        trust_remote_code=True,
    )
    frames = [raw[s].to_pandas() for s in raw.keys()]
    df = pd.concat(frames, ignore_index=True)
    df = df.dropna(subset=["question", "answer"])
    df = df[df["answer"].str.len() > 80].sample(n * 5, random_state=99)
    df = df.drop_duplicates("question").head(n)
    return df[["question", "answer"]].to_dict("records")


# ── Model loading helpers ─────────────────────────────────────────

def _load_phi3(model_id: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel, PeftConfig

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    cfg   = PeftConfig.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        cfg.base_model_name_or_path,
        quantization_config=bnb,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="eager",
    )
    model   = PeftModel.from_pretrained(model, model_id)
    tok     = AutoTokenizer.from_pretrained(cfg.base_model_name_or_path, trust_remote_code=True)
    tok.pad_token = tok.unk_token
    return model, tok


def _load_falcon(model_id: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel, PeftConfig

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    cfg   = PeftConfig.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        cfg.base_model_name_or_path,
        quantization_config=bnb,
        return_dict=True,
        device_map="auto",
        trust_remote_code=True,
    )
    model   = PeftModel.from_pretrained(model, model_id)
    tok     = AutoTokenizer.from_pretrained(cfg.base_model_name_or_path, trust_remote_code=True)
    tok.pad_token = tok.eos_token
    return model, tok


def _prompt_phi3(question: str) -> str:
    return (
        f"<|system|>\n{SYSTEM_PROMPT}<|end|>\n"
        f"<|user|>\n{question}<|end|>\n"
        f"<|assistant|>\n"
    )


def _prompt_falcon(question: str) -> str:
    return f": {question}?\n: "


# ── Inference ─────────────────────────────────────────────────────

def generate_responses(
    model,
    tokenizer,
    eval_set: list[dict],
    model_type: str,
    device: str = "cuda",
) -> tuple[list[str], list[float]]:
    """Run inference and return (predictions, latencies)."""
    import torch

    model.eval()
    preds, latencies = [], []

    for sample in eval_set:
        prompt = (
            _prompt_phi3(sample["question"])
            if model_type == "phi3"
            else _prompt_falcon(sample["question"])
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        t0 = time.time()
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=150,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.2,
            )
        latencies.append(time.time() - t0)
        generated = tokenizer.decode(
            output[0][inputs.input_ids.shape[1]:], skip_special_tokens=True
        ).strip()
        preds.append(generated)

    return preds, latencies


# ── Metrics ───────────────────────────────────────────────────────

def compute_rouge(preds: list[str], refs: list[str]) -> dict:
    from rouge_score import rouge_scorer as rs
    scorer = rs.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    r1, r2, rL = [], [], []
    for p, r in zip(preds, refs):
        s = scorer.score(r, p)
        r1.append(s["rouge1"].fmeasure)
        r2.append(s["rouge2"].fmeasure)
        rL.append(s["rougeL"].fmeasure)
    return {
        "rouge1": round(float(np.mean(r1)), 4),
        "rouge2": round(float(np.mean(r2)), 4),
        "rougeL": round(float(np.mean(rL)), 4),
    }


def compute_bertscore(preds: list[str], refs: list[str]) -> dict:
    from bert_score import score as bs_score
    P, R, F1 = bs_score(preds, refs, lang="en", verbose=False)
    return {
        "bertscore_p":  round(float(P.mean()), 4),
        "bertscore_r":  round(float(R.mean()), 4),
        "bertscore_f1": round(float(F1.mean()), 4),
    }


# ── Main evaluation loop ──────────────────────────────────────────

def evaluate_model(key: str, eval_set: list[dict]) -> dict:
    import torch

    cfg    = MODEL_REGISTRY[key]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n{'='*60}")
    print(f"Evaluating: {cfg['name']}")
    print(f"Model ID  : {cfg['model_id']}")
    print(f"{'='*60}")

    load_fn = _load_phi3 if cfg["type"] == "phi3" else _load_falcon
    try:
        model, tok = load_fn(cfg["model_id"])
    except Exception as e:
        print(f"⚠️  Could not load {cfg['model_id']}: {e}")
        return {"name": cfg["name"], "error": str(e)}

    refs         = [s["answer"] for s in eval_set]
    preds, lats  = generate_responses(model, tok, eval_set, cfg["type"], device)

    rouge_m  = compute_rouge(preds, refs)
    bert_m   = compute_bertscore(preds, refs)

    result = {
        "name":           cfg["name"],
        "model_id":       cfg["model_id"],
        "base_model":     cfg["base_model"],
        "method":         cfg["method"],
        "train_examples": cfg["train_examples"],
        "adapter_size_mb": cfg["adapter_size_mb"],
        **rouge_m,
        **bert_m,
        "latency_s":      round(float(np.mean(lats)), 2),
        "eval_examples":  len(eval_set),
    }

    print(f"  ROUGE-1 : {result['rouge1']}")
    print(f"  ROUGE-L : {result['rougeL']}")
    print(f"  BERTScore F1: {result['bertscore_f1']}")
    print(f"  Avg latency: {result['latency_s']}s")

    # Clean up GPU memory
    import gc
    del model, tok
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result


def load_or_create_results() -> dict:
    if RESULTS_PATH.exists():
        with open(RESULTS_PATH) as f:
            return json.load(f)
    return {"models": [], "last_updated": ""}


def save_results(data: dict) -> None:
    from datetime import datetime
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    data["last_updated"] = datetime.utcnow().isoformat()
    with open(RESULTS_PATH, "w") as f:
        json.dump(data, f, indent=2)
    print(f"\n✅ Results saved to {RESULTS_PATH}")


def print_table(data: dict) -> None:
    models = data.get("models", [])
    if not models:
        print("No results yet. Run with --all to evaluate.")
        return

    cols = ["name", "rouge1", "rouge2", "rougeL", "bertscore_f1", "latency_s", "train_examples"]
    df   = pd.DataFrame(models)[cols]
    df.columns = ["Model", "R-1", "R-2", "R-L", "BERTScore F1", "Latency(s)", "Train Ex."]
    print("\n" + df.to_string(index=False))


# ── CLI ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--all",   action="store_true", help="Evaluate all models")
    parser.add_argument("--model", choices=list(MODEL_REGISTRY.keys()), help="Evaluate one model")
    parser.add_argument("--table", action="store_true", help="Print results table (no GPU)")
    args = parser.parse_args()

    data = load_or_create_results()

    if args.table:
        print_table(data)
        sys.exit(0)

    keys_to_eval = list(MODEL_REGISTRY.keys()) if args.all else ([args.model] if args.model else [])

    if not keys_to_eval:
        parser.print_help()
        sys.exit(1)

    eval_set = load_eval_set(EVAL_SAMPLES)
    print(f"Loaded {len(eval_set)} evaluation examples.")

    existing = {m["model_id"]: m for m in data.get("models", [])}

    for key in keys_to_eval:
        result = evaluate_model(key, eval_set)
        existing[result["model_id"]] = result

    data["models"] = list(existing.values())
    save_results(data)
    print_table(data)
