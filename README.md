# 🏥 MEDIGUIDE:  Fine-Tuned Medical Chatbot
<div align="center">

[![Python](https://img.shields.io/badge/Python-3.8+-green.svg)](https://python.org)
[![Hugging Face](https://img.shields.io/badge/🤗-Hugging%20Face-yellow)](https://huggingface.co)
[![Kaggle](https://img.shields.io/badge/Kaggle-Dataset-blue)](https://www.kaggle.com/datasets/pythonafroz/medquad-medical-question-answer-for-ai-research)

*An AI-powered medical chatbot built with advanced fine-tuning techniques*


</div>

---

## 📖 Abstract

**MEDIGUIDE** is a conversational AI chatbot fine-tuned to provide accurate, medically grounded responses to patient queries. We experiment with three cutting-edge fine-tuning strategies on the Falcon-7B model: **Prompt Tuning**, **LoRA**, and **QLoRA**. Using a curated subset of the MedQuAD dataset, we comprehensively evaluate each method using ROUGE, Perplexity (PPL), latency, and model size metrics. Our results reveal crucial trade-offs between quality, speed, and resource efficiency, culminating in evidence-based deployment recommendations for real-world medical chatbot scenarios.

---

## 🎯 Introduction

With the exponential growth in demand for accessible medical advice, AI-powered chatbots have emerged as a transformative solution in healthcare technology. These intelligent systems provide:

- 🔍 **Immediate responses** to general health queries
- 🏥 **Triage assistance** for healthcare providers
- ⚠️ **Clear disclaimers** emphasizing they're not substitutes for professional diagnosis
- 🌐 **24/7 availability** for basic medical information

---

## 🖼️ Screenshots & Demo

<div align="center">

### 📺 [**Watch Full Demo Video**](https://drive.google.com/file/d/1-h_btasIkoq7TZpBmwGoKlfcLfSM9aFW/view?usp=drive_link)

</div>

![MEDIGUIDE Interface 1](https://github.com/user-attachments/assets/3b4604cd-a054-499a-afae-8ff8e6b41148)

![MEDIGUIDE Interface 2](https://github.com/user-attachments/assets/e71f37a3-aa07-485a-8824-bafe7a23f7c0)


---

## 🎯 Project Goals

<div align="center">

| 🎯 **Objective** | 📋 **Description** |
|---|---|
| 🤖 **AI Development** | Build a Falcon-7B-based chatbot for medical query responses |
| 🏥 **Clinical Standards** | Ensure clinical tone, accuracy, and comprehensive safety disclaimers |
| 🔬 **Method Comparison** | Fine-tune using Prompt Tuning, LoRA, and QLoRA techniques |
| 📊 **Performance Analysis** | Compare methods on quality, efficiency, and deployability metrics |

</div>

---

## 📊 Dataset & Preprocessing

### 🗂️ Dataset Overview

Our foundation is the **MedQuAD** (Medical Question Answering Dataset) - a meticulously curated collection sourced from 12 trusted **NIH** websites including:

- 🎗️ **cancer.gov** - Cancer-related queries
- 🧬 **GARD** - Genetic and Rare Diseases
- 🏥 **NIH Clinical Center** - General medical information
- *...and 9 other authoritative sources*

**🔒 Privacy Compliance**: HIPAA-equivalent standards rigorously enforced

### 📋 Data Structure

```json
{
  "question": "What are the symptoms of diabetes?",
  "answer": "Expert-written clinical response...",
  "source": "Source website identifier",
  "focus_area": "Endocrinology"
}
```

### 🛠️ Preprocessing Pipeline

1. **🔍 Quality Filtering**: Removed short answers and low-context examples
2. **📝 Format Standardization**: 
   ```
   <human>: "What are the symptoms of diabetes?"
   <assistant>: "Diabetes symptoms include frequent urination..."
   ```
3. **📊 Dataset Split**: 
   - 🎯 **Training**: 200 examples
   - 🧪 **Evaluation**: 50 examples
   - 🔀 **Shuffled** for optimal distribution

---

## 🔬 Fine-Tuning Methodologies

<div align="center">

### 🤖 **Base Models**
**Falcon-7B** (Quantized) | **Falcon-1B** (Full Precision)

*All methods trained on identical datasets for fair comparison*

</div>

### 1. 🎯 Prompt Tuning

<div align="center">

| **Feature** | **Details** |
|---|---|
| 🔧 **Mechanism** | Virtual tokens injected before questions |
| 📏 **Training Scope** | Prompt embeddings only (<1MB) |
| 🔒 **Base Model** | Weights frozen |
| ⚡ **Advantages** | Ultra-fast training, minimal memory |

</div>

### 2. 🔗 LoRA (Low-Rank Adaptation)

<div align="center">

| **Feature** | **Details** |
|---|---|
| 🔧 **Mechanism** | Adapter layers in attention blocks |
| 💾 **Precision** | 16-bit (BF16) |
| ⚖️ **Trade-off** | Moderate cost, enhanced fluency |
| 🎯 **Target** | Balanced performance-efficiency |

</div>

### 3. 🚀 QLoRA (Quantized LoRA)

<div align="center">

| **Feature** | **Details** |
|---|---|
| 🔧 **Mechanism** | 4-bit quantization via `bitsandbytes` |
| 🔗 **Adapters** | LoRA adapters through PEFT |
| 💾 **Efficiency** | Maximum memory optimization |
| ⏱️ **Trade-off** | Slight latency increase |

</div>

### 🛠️ Technology Stack

<div align="center">

| **Category** | **Libraries** |
|---|---|
| 🤗 **Core ML** | Hugging Face Transformers |
| 🔧 **Fine-tuning** | PEFT, TRL, BitsAndBytes |
| 🔥 **Framework** | PyTorch, Accelerate |

</div>

---

## 📈 Evaluation Metrics

<div align="center">

| 📊 **Metric** | 📋 **Description** | 🎯 **Purpose** |
|---|---|---|
| **ROUGE (1, 2, L)** | Token overlap with reference answers | Text similarity assessment |
| **Perplexity (PPL)** | Model confidence in generated text | Fluency measurement |
| **Latency** | Average inference time per query | Performance benchmarking |
| **Model Size** | Trainable parameters (adapter size) | Resource efficiency |

</div>

---

## 🏆 Results & Performance Analysis

<div align="center">

### 📊 **Comprehensive Performance Comparison**

| **Method** | **ROUGE-1** | **ROUGE-2** | **ROUGE-L** | **Perplexity** | **Latency (s)** | **Adapter Size** |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Prompt Tuning (Q)** | `0.21` | `0.04` | `0.12` | `5.85` | `8.81` | `~0.43 MB` |
| **Prompt Tuning (FP)** | `0.18` | `0.02` | `0.10` | `6.89` | `1.89` | `~0.20 MB` |
| **LoRA (FP)** | `0.21` | `0.04` | `0.12` | `5.31` | `3.53` | `~12 MB` |
| **🏆 QLoRA (4-bit)** | `0.45` | `0.17` | `0.30` | `3.45` | `10.94` | `~18 MB` |

</div>

### 📈 Performance Insights

<details>
<summary>🔍 Click for detailed analysis</summary>

#### 🥇 **QLoRA - Champion Performance**
- **Best Overall**: Highest ROUGE scores across all metrics
- **Superior Fluency**: Lowest perplexity (3.45)
- **Acceptable Latency**: 10.94s - reasonable for quality gained

#### 🥈 **LoRA - Balanced Approach**
- **Moderate Performance**: Decent ROUGE scores
- **Good Fluency**: PPL of 5.31
- **Fast Response**: 3.53s latency

#### 🥉 **Prompt Tuning - Efficiency Focus**
- **Lightweight**: Smallest model size
- **Variable Performance**: FP version faster, Q version more accurate
- **Edge-Friendly**: Minimal resource requirements

</details>

---

## 💡 Method Analysis & Trade-offs

### 🎯 Prompt Tuning
<div align="center">

| **Pros** ✅ | **Cons** ❌ |
|---|---|
| Ultra-lightweight deployment | Lower accuracy scores |
| Fastest inference times | Limited clinical fluency |
| Minimal memory footprint | Reduced contextual understanding |
| **Perfect for**: Edge/Mobile deployment | **Limitation**: Professional use cases |

</div>

### 🔗 LoRA (Full Precision)
<div align="center">

| **Pros** ✅ | **Cons** ❌ |
|---|---|
| Balanced quality-performance | Moderate resource requirements |
| Good clinical accuracy | Higher memory usage |
| Reasonable inference speed | Complex deployment setup |
| **Perfect for**: Hosted GPU servers | **Limitation**: Resource constraints |

</div>

### 🚀 QLoRA (Quantized + LoRA)
<div align="center">

| **Pros** ✅ | **Cons** ❌ |
|---|---|
| **Best accuracy** across all metrics | Slower inference times |
| Memory-efficient quantization | Complex quantization setup |
| Superior clinical fluency | Higher computational overhead |
| **Perfect for**: Cloud GPU deployment | **Limitation**: Real-time applications |

</div>

---

## 🚀 Deployment Strategy & Recommendations

<div align="center">

### 🎯 **Deployment Decision Matrix**

</div>

### 1. 📱 **Edge/Mobile Deployment**
```yaml
Method: Prompt Tuning
Size: ~500KB
Latency: ~3s
Memory: Minimal
Use Case: Mobile apps, offline operation
Trade-off: Reduced clinical accuracy for portability
```

### 2. 🖥️ **Hosted GPU Servers**
```yaml
Method: LoRA (Full Precision)
Size: ~30MB adapter + 13-16GB VRAM
Latency: ~7.5s
Platform: AWS, GCP, Azure
Use Case: Web applications, API services
Trade-off: Balanced performance and resources
```

### 3. ☁️ **Cloud GPU (Recommended)**
```yaml
Method: QLoRA (Quantized + LoRA)
Size: ~30MB + 5-6GB quantized model
Latency: ~10.9s
Platform: Hugging Face Spaces, Colab Pro
Use Case: Research, high-accuracy applications
Trade-off: Best overall quality-efficiency balance
```

---

## 🎉 Conclusion & Future Roadmap

### 🏁 **Key Findings**

We successfully developed **MEDIGUIDE**, a medically-aware chatbot using Falcon-7B with advanced PEFT techniques. **QLoRA emerged as the optimal choice**, delivering the best trade-off between resource efficiency and output quality.

### 🛣️ **Future Development Pipeline**

<div align="center">

| **Phase** | **Objective** | **Timeline** |
|---|---|---|
| 🔄 **Scale-Up** | Train on 10k+ examples for enhanced generalizability | Q3 2025 |
| 🧪 **Model Optimization** | Experiment with smaller models + extended training | Q4 2025 |
| 🔬 **Advanced PEFT** | Explore prefix tuning, adapters, and novel techniques | Q1 2026 |
| ⚙️ **Hyperparameter Tuning** | Rigorous optimization for peak performance | Q2 2026 |
| 🚀 **Production Deployment** | Implement recommended deployment strategies | Q3 2026 |

</div>

---

## 🖥️ Hardware Specifications

<div align="center">

### 💻 **Development Environment (Kaggle)**

| **Component** | **Specification** |
|---|---|
| 🎮 **GPU** | NVIDIA T4 (15GB VRAM) |
| 💾 **RAM** | 29GB System Memory |
| 💿 **Storage** | 58GB SSD |
| ⚡ **Platform** | Kaggle Notebooks |

</div>

---

## 📚 Resources & Links

### 📊 Dataset
- 🔗 [**MedQuAD Dataset on Kaggle**](https://www.kaggle.com/datasets/pythonafroz/medquad-medical-question-answer-for-ai-research)

### 🤗 Model Links (Hugging Face)

<div align="center">

| **Model** | **Link** | **Description** |
|---|---|---|
| 🚀 **QLoRA** | [TestCase1/falcon-7b-qlora-chat-medical-bot](https://huggingface.co/TestCase1/falcon-7b-qlora-chat-medical-bot) | Best performance model |
| 🔗 **LoRA** | [TestCase1/falcon-7b-lora-chat-medical-bot](https://huggingface.co/TestCase1/falcon-7b-lora-chat-medical-bot) | Balanced approach |
| 🎯 **Prompt FP** | [TestCase1/falcon-7b-prompt-fp-chat-medical-bot](https://huggingface.co/TestCase1/falcon-7b-prompt-fp-chat-medical-bot) | Full precision prompt tuning |
| ⚡ **Prompt Quantized** | [TestCase1/falcon-7b-prompt-chat-medical-bot](https://huggingface.co/TestCase1/falcon-7b-prompt-chat-medical-bot) | Quantized prompt tuning |

</div>

---
