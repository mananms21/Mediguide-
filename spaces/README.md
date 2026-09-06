---
title: MediGuide — Medical AI Chatbot
emoji: ⚕
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: "4.44.0"
app_file: app.py
pinned: true
license: mit
short_description: Phi-3 Mini QLoRA medical chatbot with RAG
tags:
  - medical
  - chatbot
  - llm
  - qlora
  - peft
  - rag
  - phi3
  - zerogpu
---

# ⚕ MediGuide — Medical AI Chatbot

A fine-tuned **Phi-3 Mini (3.8B)** medical chatbot with **Retrieval-Augmented Generation (RAG)**
over 14,782 NIH MedQuAD Q&A pairs.

## Ablation Results

| Condition | Clinical BERTScore | ROUGE-1 @50tok | NLI Contradiction |
|---|---|---|---|
| Zero-shot | 0.9203 | 0.1949 | 0.1007 |
| Fine-tuned | 0.9401 | 0.2903 | 0.2173 |
| **Fine-tuned + RAG** ★ | **0.9740** | **0.4104** | **0.0780** |
| OOD (PubMedQA) | 0.9186 | 0.2392 | 0.1374 |

## Tech Stack

`microsoft/Phi-3-mini-4k-instruct` · `QLoRA 4-bit NF4` · `PEFT` · `FAISS` · `SentenceTransformers` · `Gradio` · `ZeroGPU`

## Links

- 🤗 [Model](https://huggingface.co/Shriyanshml/phi3-mini-qlora-mediguide)
- 📦 [RAG Index](https://huggingface.co/datasets/Shriyanshml/mediguide-rag-index)
- 💻 [GitHub](https://github.com/mananms21/Mediguide-)
