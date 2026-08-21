---
title: MEDIGUIDE — Medical AI Chatbot
emoji: 🏥
colorFrom: blue
colorTo: teal
sdk: gradio
sdk_version: "4.44.0"
app_file: app.py
pinned: true
license: mit
short_description: Fine-tuned Phi-3 Mini medical chatbot with RAG over MedQuAD
tags:
  - medical
  - chatbot
  - llm
  - qlora
  - peft
  - rag
  - phi3
---

# 🏥 MEDIGUIDE — Medical AI Chatbot

A fine-tuned **Phi-3 Mini (3.8B)** medical chatbot with **Retrieval-Augmented Generation (RAG)**
over the MedQuAD dataset (16,000+ NIH-sourced Q&A pairs).

## Features

- **QLoRA Fine-tuned** on 2,000 MedQuAD examples using 4-bit NF4 quantization
- **RAG Pipeline** via FAISS + SentenceTransformers — retrieves relevant references before answering
- **Medical Disclaimer** on all responses (HIPAA-equivalent standards)

## Models Compared

| Method | ROUGE-1 | Perplexity | Latency |
|---|---|---|---|
| Phi-3 Mini QLoRA (this app) | TBD | TBD | ~5s |
| Falcon-7B QLoRA (baseline) | 0.25 | 3.45 | 10.94s |

## Tech Stack

`microsoft/Phi-3-mini-4k-instruct` · `PEFT` · `FAISS` · `SentenceTransformers` · `Gradio`
