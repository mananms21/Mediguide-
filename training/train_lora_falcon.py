"""
MEDIGUIDE — Falcon-7B LoRA Fine-tuning (BF16)
==============================================
Baseline experiment: full-precision LoRA on Falcon-7B.
Superseded by QLoRA Phi-3 (see train_qlora_phi3.py).
Kept for reproducibility of baseline results in evaluate/results/results.json.

Run on:  Kaggle T4 / Colab A100
Dataset: keivalya/MedQuad-MedicalQnADataset (200 training examples)
Pushes:  TestCase1/falcon-7b-lora-chat-medical-bot
"""

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer

# ── Config ────────────────────────────────────────────────────────────
BASE_MODEL  = "tiiuae/falcon-7b"
OUTPUT_DIR  = "./falcon-lora-checkpoints"
HF_REPO     = "TestCase1/falcon-7b-lora-chat-medical-bot"
N_TRAIN     = 200
EPOCHS      = 3
LR          = 2e-4

SYSTEM_PROMPT = (
    "You are MEDIGUIDE, a knowledgeable medical assistant. "
    "Provide accurate, evidence-based answers. Always recommend consulting a doctor."
)

# ── Dataset ───────────────────────────────────────────────────────────
raw = load_dataset("keivalya/MedQuad-MedicalQnADataset")
df  = raw["train"].to_pandas()
df.columns = [c.lower() for c in df.columns]
df  = df.dropna(subset=["question","answer"])
df  = df[df["answer"].str.len() > 80].sample(N_TRAIN, random_state=42)

def format_row(row):
    return (
        f"<|system|>\n{SYSTEM_PROMPT}\n"
        f"<|user|>\n{row['question']}\n"
        f"<|assistant|>\n{row['answer']}"
    )

from datasets import Dataset
train_ds = Dataset.from_dict({"text": df.apply(format_row, axis=1).tolist()})

# ── Model ─────────────────────────────────────────────────────────────
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
)

lora_cfg = LoraConfig(
    task_type=TaskType.CAUSAL_LM, r=8, lora_alpha=32,
    lora_dropout=0.1, target_modules=["query_key_value"],
)
model = get_peft_model(model, lora_cfg)
model.print_trainable_parameters()

# ── Training ──────────────────────────────────────────────────────────
args = TrainingArguments(
    output_dir=OUTPUT_DIR, num_train_epochs=EPOCHS,
    per_device_train_batch_size=2, gradient_accumulation_steps=4,
    learning_rate=LR, fp16=False, bf16=True,
    logging_steps=10, save_strategy="epoch",
    push_to_hub=True, hub_model_id=HF_REPO,
)

trainer = SFTTrainer(model=model, args=args, train_dataset=train_ds,
                     tokenizer=tokenizer, dataset_text_field="text",
                     max_seq_length=512)
trainer.train()
trainer.push_to_hub()
print(f"✅ Model pushed to {HF_REPO}")
