
# MEDIGUIDE: Comparative Fine-Tuning for Medical Question Answering

## Abstract

In this project, we present **MEDIGUIDE**, a conversational AI chatbot fine-tuned to provide accurate, medically grounded responses to patient queries. We experiment with three fine-tuning strategies on the Falcon-7B model: Prompt Tuning, LoRA, and QLoRA. Using a subset of the MedQuAD dataset, we measure each method's performance using ROUGE, Perplexity (PPL), latency, and model size. Our results highlight the trade-offs between quality, speed, and resource efficiency, leading to a recommended deployment strategy tailored for real-world medical chatbot scenarios.

---

## Introduction

With the growing demand for quick medical advice, AI-powered chatbots have gained significant relevance. These systems offer immediate responses to general health queries and assist in triage, while clearly disclaiming that they are not substitutes for professional diagnosis.

## Screenshots 

 ![Screenshot 2025-06-23 132357](https://github.com/user-attachments/assets/3b4604cd-a054-499a-afae-8ff8e6b41148)
 ![Screenshot 2025-06-23 132408](https://github.com/user-attachments/assets/e71f37a3-aa07-485a-8824-bafe7a23f7c0)

### Goals of the Project

- Build a Falcon-7B-based chatbot that answers user medical queries.
- Ensure answers follow clinical tone, accuracy, and include safety disclaimers.
- Fine-tune using Prompt Tuning (both quantised and full precision), LoRA, and QLoRA.
- Compare each method based on output quality, efficiency, and deployability.

---

## Dataset & Preprocessing

### Dataset Used

- Derived from **MedQuAD** - Medical Question Answering Dataset, a collection of QA pairs curated from 12 trusted **NIH** websites (e.g., cancer.gov, GARD).
- HIPAA-equivalent standards ensured.

Each entry includes:
- `question`: Medical query  
- `answer`: Expert-written answer  
- `source`: Source website  
- `focus_area`: Query label  

### Preprocessing Steps

- Removed short answers and low-context examples.
- Reformatted into:

```plaintext
<human>: "question"
<assistant>: "answer"
```

- Split into 200 training and 50 evaluation examples (shuffled).

---

## Fine-Tuning Methods

Models used: **Falcon-7B** (quantised) and **Falcon-1B** (full precision)  
All methods trained on identical data.

### 1. Prompt Tuning
- Injected virtual tokens before the question.
- Only trained prompt embeddings (<1MB).
- Model weights frozen.
- Fast and low-memory training.

### 2. LoRA (Full Precision)
- Adapter layers injected into attention blocks.
- 16-bit precision (BF16).
- Moderate cost, better fluency.

### 3. QLoRA (Quantized LoRA)
- 4-bit quantization using `bitsandbytes`.
- LoRA adapters via PEFT.
- Most memory-efficient, slightly higher latency.

### Common Libraries

- Hugging Face Transformers  
- PEFT, TRL, BitsAndBytes  
- PyTorch, Accelerate

---

## Evaluation Metrics

- **ROUGE (1, 2, L)**: Token overlap with gold answer
- **Perplexity (PPL)**: Confidence in generated text
- **Latency**: Avg. inference time per question
- **Model Size**: Trainable parameters (adapter size)

---

## Results

| Method             | ROUGE-1 | ROUGE-2 | ROUGE-L | Perplexity | Latency (s) | Adapter Size |
|--------------------|---------|---------|---------|------------|-------------|---------------|
| Prompt Tuning (Q)  | 0.21    | 0.04    | 0.12    | 5.85       | 8.81        | ~0.43 MB      |
| Prompt Tuning (FP) | 0.18    | 0.02    | 0.10    | 6.89       | 1.89        | ~0.20 MB      |
| LoRA (FP)          | 0.21    | 0.04    | 0.12    | 5.31       | 3.53        | ~12 MB        |
| QLoRA (4-bit)      | 0.25    | 0.07    | 0.15    | 3.45       | 10.94       | ~18 MB        |

---

## Discussion

### Prompt Tuning

- Lightweight, fast, but lower performance.
- Suitable for mobile or edge deployment.

### LoRA

- Balanced quality and latency.
- Simpler than full fine-tuning.

### QLoRA

- Best performance (PPL = 3.45, highest ROUGE).
- Slightly slower due to quantization.
- Very memory-efficient (runs on 16GB T4 GPU).

---

## Trade-offs & Deployment Strategy

### Summary

- **Prompt Tuning**: Fastest, smallest; weakest accuracy.
- **LoRA**: Balanced in quality and resource needs.
- **QLoRA**: Best accuracy; acceptable latency.

### 1. Prompt Tuning

- **Size**: ~500KB  
- **Latency**: ~3s  
- **Memory**: Minimal  
- **Ideal for**: Edge/mobile  
- **Trade-offs**: Weakest clinical fluency

### 2. LoRA (Full Precision)

- **Size**: ~30MB adapter + FP16 base (13–16GB VRAM)  
- **Latency**: ~7.5s  
- **Ideal for**: Hosted GPU servers (AWS, GCP, HF)  
- **Trade-offs**: Balanced output and latency

### 3. QLoRA (Quantized + LoRA)

- **Size**: ~30MB + 5–6GB quantized model  
- **Latency**: ~10.9s  
- **Ideal for**: Cloud GPU (e.g., HF Spaces)  
- **Trade-offs**: Best overall trade-off

---

## Conclusion & Future Work

**Conclusion**:  
We developed a medically-aware chatbot using Falcon-7B, fine-tuned via PEFT techniques. **QLoRA** showed the best trade-off between resource use and output quality.

**Future Extensions**:

- Train on 10k+ examples for generalizability.
- Try smaller models with more epochs.
- Explore more PEFTs (e.g., prefix tuning, adapters).
- Tune hyperparameters rigorously.
- Deploy using the recommended strategies.

---

## Hardware Used (Kaggle)

- T4 GPU (15GB VRAM)  
- 29GB RAM  
- 58GB Disk  

---

## Dataset

[MedQuAD Dataset on Kaggle](https://www.kaggle.com/datasets/pythonafroz/medquad-medical-question-answer-for-ai-research)

---

## Model Links (Hugging Face)

- [QLoRA](https://huggingface.co/TestCase1/falcon-7b-qlora-chat-medical-bot)  
- [LoRA](https://huggingface.co/TestCase1/falcon-7b-lora-chat-medical-bot)  
- [Prompt FP](https://huggingface.co/TestCase1/falcon-7b-prompt-fp-chat-medical-bot)  
- [Prompt Quantised](https://huggingface.co/TestCase1/falcon-7b-prompt-chat-medical-bot)
