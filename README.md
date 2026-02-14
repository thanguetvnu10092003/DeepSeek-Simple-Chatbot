---
title: PDF RAG DeepSeek OCR Chatbot
emoji: 📄
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: "5.9.1"
python_version: "3.11"
app_file: app.py
pinned: false
license: mit
---

<div align="center">

# 📚 PDF RAG Chatbot with DeepSeek OCR

<p align="center">
  <a href="https://python.org"><img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python"></a>
  <a href="https://gradio.app"><img src="https://img.shields.io/badge/Gradio-5.x-FF6F00?style=for-the-badge&logo=gradio&logoColor=white" alt="Gradio"></a>
  <a href="https://langchain-ai.github.io/langgraph/"><img src="https://img.shields.io/badge/LangGraph-Agentic_RAG-8B5CF6?style=for-the-badge&logo=chainlink&logoColor=white" alt="LangGraph"></a>
</p>

<p align="center">
  <a href="https://huggingface.co/spaces/toanthangle/pdf-rag-deepseek-ocr-chatbot"><img src="https://img.shields.io/badge/🤗%20Live%20Demo-Hugging%20Face-FFD21E?style=for-the-badge" alt="Hugging Face"></a>
  <a href="https://console.groq.com/"><img src="https://img.shields.io/badge/Groq-LLM_API-F55036?style=for-the-badge&logo=groq&logoColor=white" alt="Groq"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-22C55E?style=for-the-badge" alt="License"></a>
</p>

<br>

> 🤖 **An intelligent AI chatbot** that lets you **chat with your PDF documents & images**  
> powered by **Agentic RAG** (LangGraph) — with multi-step reasoning, self-correction & hallucination checking.

<br>

### 🚀 [Try the Live Demo →](https://huggingface.co/spaces/toanthangle/pdf-rag-deepseek-ocr-chatbot)

<br>

[Features](#-key-features) · [Architecture](#-agentic-rag-architecture) · [Quick Start](#-quick-start) · [Usage](#-usage) · [Pricing](#-pricing)

</div>

<br>

---

<br>

## ⚡ Key Features

<table>
<tr>
<td width="50%">

### 🤖 Agentic RAG Mode
- **Smart query routing** — auto-classifies simple vs complex
- **Query decomposition** — breaks down multi-part questions
- **Document grading** — filters irrelevant retrieval results
- **Self-correction** — rewrites queries & retries (max 2x)
- **Hallucination check** — verifies answers are grounded
- **Reasoning steps** — view agent's thinking in UI

</td>
<td width="50%">

### 🔍 Hybrid RAG Engine
- **Dual vectorstore** — small (500) & large (1500) chunks
- **Semantic search** — sentence-transformers embeddings
- **BM25 keyword search** — combined for hybrid retrieval
- **Multi-file query** — search across selected files
- **OCR support** — DeepSeek OCR for scanned PDFs & images
- **Dual mode toggle** — switch Agentic ↔ Traditional RAG

</td>
</tr>
</table>

<br>

---

<br>

## 🏗️ Agentic RAG Architecture

```
                          ┌──────────────┐
                          │  User Query  │
                          └──────┬───────┘
                                 ▼
                        ┌────────────────┐
                        │  🧭 Router    │  Classify: simple / complex
                        └───────┬────────┘
                                │
                ┌───────────────┴───────────────┐
                ▼                               ▼
         ┌─────────────┐              ┌─────────────────┐
         │   Simple    │              │  🔀 Decompose   │  Split into sub-questions
         └──────┬──────┘              └────────┬────────┘
                │                              │
                └──────────────┬───────────────┘
                               ▼
                      ┌────────────────┐
                      │  🔍 Retrieve  │  Hybrid Search (Semantic + BM25)
                      └───────┬────────┘
                              ▼
                      ┌────────────────┐
                      │  📊 Grade     │  Filter irrelevant documents
                      └───────┬────────┘
                              │
                   ┌──────────┴──────────┐
                   ▼                     ▼
            ┌────────────┐       ┌──────────────┐
            │  ✅ ≥30%   │       │  ❌ <30%     │
            │  relevant  │       │  retry < 2   │
            └─────┬──────┘       └──────┬───────┘
                  │                     ▼
                  │              ┌──────────────┐
                  │              │  ✏️ Rewrite  │  Reformulate query
                  │              └──────┬───────┘
                  │                     │
                  │                     └──► (back to 🔍 Retrieve)
                  ▼
           ┌────────────────┐
           │  💬 Generate  │  Answer with context
           └───────┬────────┘
                   ▼
          ┌─────────────────┐
          │  🛡️ Hallucinate │  Verify grounding
          │     Check       │
          └───────┬─────────┘
                  ▼
            ┌───────────┐
            │  ✨ Answer │
            └───────────┘
```

<br>

---

<br>

## 🚀 Quick Start

### 1️⃣ Clone & Install

```bash
git clone https://github.com/thanguetvnu10092003/DeepSeek-Simple-Chatbot.git
cd DeepSeek-Simple-Chatbot
```

<details>
<summary><b>Option A: Conda (recommended)</b></summary>

```bash
conda env create -f environment.yaml
conda activate chatbot
```
</details>

<details>
<summary><b>Option B: pip</b></summary>

```bash
pip install -r requirements.txt
```
</details>

### 2️⃣ Configure API Keys

Create a `.env` file:

```env
GROQ_API_KEY=your_groq_api_key
REPLICATE_API_TOKEN=your_replicate_token
```

| Provider | Purpose | Get Key |
|----------|---------|---------|
| **Groq** | LLM (free) | [console.groq.com](https://console.groq.com/) |
| **Replicate** | OCR (paid) | [replicate.com/account](https://replicate.com/account/api-tokens) |

### 3️⃣ Launch

```bash
python app.py
```

Open **http://127.0.0.1:7860** in your browser 🎉

<br>

---

<br>

## 📖 Usage

| Step | Action | Details |
|------|--------|---------|
| **1** | 📤 Upload | Drag & drop PDF / PNG / JPG files (max 50MB each) |
| **2** | 🔧 OCR | Toggle ON for scanned PDFs & images (~$0.001/page) |
| **3** | 🤖 Mode | **Agentic RAG** (default, smart) or **Traditional RAG** (fast) |
| **4** | 💬 Chat | Ask questions, select specific files via dropdown |
| **5** | 🧠 Reasoning | Expand "Agent Reasoning Steps" to see agent thinking |

<br>

---

<br>

## 📁 Project Structure

```
📦 DeepSeek-Simple-Chatbot
│
├── 🎯 app.py               → Gradio UI, mode toggle, reasoning panel
├── 🤖 agentic_rag.py        → LangGraph workflow (6 agent nodes)
├── 🔍 rag.py               → Hybrid RAG engine + agentic integration
├── 🧠 llm.py               → Groq LLM wrapper + 5 agentic methods
├── 📄 pdf_ocr_loader.py    → PDF/image loader with DeepSeek OCR
│
├── 📋 requirements.txt     → Python dependencies
├── 📋 environment.yaml     → Conda environment config
└── 🔑 .env                 → API keys (create manually)
```

<br>

---

<br>

## 💰 Pricing

| Operation | Cost | Notes |
|-----------|------|-------|
| 📄 PDF Text Extraction | **Free** | Built-in PyMuPDF |
| 👁️ OCR (per page) | ~$0.001 | DeepSeek via Replicate |
| 🤖 LLM Inference | **Free** | Groq API (rate limited) |
| 🔍 Embeddings | **Free** | Local sentence-transformers |

> **💡 Tip:** Agentic RAG uses ~2-4x more LLM calls than Traditional RAG for routing, grading, and hallucination checking — but since Groq is free, there's **no extra cost**.

<br>

---

<br>

## 🛠️ Tech Stack

<p align="center">
  <img src="https://img.shields.io/badge/LangChain-Framework-1C3C3C?style=flat-square&logo=langchain&logoColor=white" alt="LangChain">
  <img src="https://img.shields.io/badge/LangGraph-Agent_Orchestration-8B5CF6?style=flat-square" alt="LangGraph">
  <img src="https://img.shields.io/badge/ChromaDB-Vector_Store-00A67E?style=flat-square" alt="ChromaDB">
  <img src="https://img.shields.io/badge/Sentence_Transformers-Embeddings-FF6F00?style=flat-square" alt="Sentence Transformers">
  <img src="https://img.shields.io/badge/BM25-Keyword_Search-2196F3?style=flat-square" alt="BM25">
  <img src="https://img.shields.io/badge/Groq-LLM_API-F55036?style=flat-square&logo=groq" alt="Groq">
  <img src="https://img.shields.io/badge/Gradio-Web_UI-FF6F00?style=flat-square&logo=gradio" alt="Gradio">
  <img src="https://img.shields.io/badge/Replicate-OCR_API-0A0A0A?style=flat-square" alt="Replicate">
</p>

<br>

---

<br>

## 📜 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

<br>

---

<div align="center">

<br>

**Made with ❤️ by [thanguetvnu10092003](https://github.com/thanguetvnu10092003)**

⭐ Star this repo if you find it useful!

</div>
