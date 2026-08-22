"""
MEDIGUIDE — Falcon-7B Prompt Tuning Baselines
==============================================
Two prompt-tuning experiments on Falcon-7B:
  - Quantized (4-bit): TestCase1/falcon-7b-prompt-chat-medical-bot
  - Full Precision (BF16): TestCase1/falcon-7b-prompt-fp-chat-medical-bot

Superseded by QLoRA Phi-3 (see train_qlora_phi3.py).
Kept for reproducibility of baseline results in evaluate/results/results.json.

Run on: Kaggle T4 / Colab A100
"""

import sys
import torch
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from peft import PromptTuningConfig, PromptTuningInit, get_peft_model, TaskType
from trl import SFTTrainer

# ── Config ────────────────────────────────────────────────────────────
BASE_MODEL = "tiiuae/falcon-7b"
MODE       = sys.argv[1] if len(sys.argv) > 1 else "quantized"  # "quantized" | "fp"

if MODE == "quantized":
    HF_REPO    = "TestCase1/falcon-7b-prompt-chat-medical-bot"
    LOAD_IN_4  = True
    BF16       = False
else:
    HF_REPO    = "TestCase1/falcon-7b-prompt-fp-chat-medical-bot"
    LOAD_IN_4  = False
    BF16       = True

N_TRAIN = 200
EPOCHS  = 3
LR      = 3e-2

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

train_ds = Dataset.from_dict({"text": df.apply(format_row, axis=1).tolist()})

# ── Model ─────────────────────────────────────────────────────────────
from transformers import BitsAndBytesConfig
bnb_cfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16) if LOAD_IN_4 else None

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb_cfg,
    torch_dtype=torch.bfloat16 if BF16 else torch.float16,
    device_map="auto", trust_remote_code=True,
)

prompt_cfg = PromptTuningConfig(
    task_type=TaskType.CAUSAL_LM,
    prompt_tuning_init=PromptTuningInit.TEXT,
    num_virtual_tokens=8,
    prompt_tuning_init_text="Answer medical questions accurately:",
    tokenizer_name_or_path=BASE_MODEL,
)
model = get_peft_model(model, prompt_cfg)
model.print_trainable_parameters()

# ── Training ──────────────────────────────────────────────────────────
args = TrainingArguments(
    output_dir=f"./falcon-prompt-{MODE}-checkpoints",
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=2, gradient_accumulation_steps=4,
    learning_rate=LR, fp16=not BF16, bf16=BF16,
    logging_steps=10, save_strategy="epoch",
    push_to_hub=True, hub_model_id=HF_REPO,
)

trainer = SFTTrainer(model=model, args=args, train_dataset=train_ds,
                     tokenizer=tokenizer, dataset_text_field="text",
                     max_seq_length=512)
trainer.train()
trainer.push_to_hub()
print(f"✅ Model pushed to {HF_REPO}")
