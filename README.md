---
title: PDF RAG DeepSeek OCR Chatbot
emoji: 📄
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: "5.9.1"
app_file: app.py
pinned: false
license: mit
---

<div align="center">

# 📚 PDF RAG Chatbot with DeepSeek OCR

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Gradio](https://img.shields.io/badge/Gradio-UI-FF6F00?style=for-the-badge&logo=gradio&logoColor=white)](https://gradio.app)
[![LangGraph](https://img.shields.io/badge/LangGraph-Agentic_RAG-8B5CF6?style=for-the-badge)](https://langchain-ai.github.io/langgraph/)
[![Hugging Face](https://img.shields.io/badge/🤗%20Hugging%20Face-Spaces-yellow?style=for-the-badge)](https://huggingface.co/spaces/toanthangle/pdf-rag-deepseek-ocr-chatbot)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

**An intelligent chatbot that enables conversations with PDF documents and images using Agentic RAG (LangGraph) and DeepSeek OCR.**

### 🚀 [Try the Live Demo](https://huggingface.co/spaces/toanthangle/pdf-rag-deepseek-ocr-chatbot)

[Features](#-features) • [Architecture](#-agentic-rag-architecture) • [Installation](#-installation) • [Usage](#-usage) • [Project Structure](#-project-structure) • [Pricing](#-pricing)

</div>

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🤖 **Agentic RAG** | LangGraph-powered agent with multi-step reasoning, self-correction, and hallucination check |
| 📁 **Multi-file Upload** | Drag and drop multiple PDF/image files at once |
| 🧠 **Smart Query Routing** | Automatically classifies and decomposes complex queries |
| 🔄 **Dual Vectorstore** | Uses 2 chunk sizes (500 & 1500) for different query types |
| 🔍 **Hybrid Search** | Combines semantic search + BM25 for better results |
| 👁️ **OCR Support** | Process scanned PDFs and images with DeepSeek OCR (via Replicate) |
| 📂 **Multi-file Query** | Select one or multiple specific files to query |
| 🔁 **Self-Correction** | Agent rewrites queries and retries when retrieval is poor (max 2 retries) |
| 📊 **Reasoning Steps** | View agent's thinking process in the UI |
| ⚡ **Dual Mode** | Switch between Agentic RAG (smart) and Traditional RAG (fast) |

---

## 🏗️ Agentic RAG Architecture

```
┌─────────────────────────────────────────────────────┐
│                   User Query                        │
└──────────────────────┬──────────────────────────────┘
                       ▼
              ┌────────────────┐
              │  Router Node   │ ← Classify: simple / complex
              └───────┬────────┘
                      │
          ┌───────────┴───────────┐
          ▼                       ▼
    ┌───────────┐         ┌──────────────┐
    │  Simple   │         │  Decompose   │ ← Split into sub-questions
    └─────┬─────┘         └──────┬───────┘
          │                      │
          └──────────┬───────────┘
                     ▼
            ┌────────────────┐
            │ Retrieve Node  │ ← Hybrid Search (Semantic + BM25)
            └───────┬────────┘
                    ▼
            ┌────────────────┐
            │  Grader Node   │ ← Filter irrelevant documents
            └───────┬────────┘
                    │
         ┌──────────┴──────────┐
         ▼                     ▼
   ┌───────────┐        ┌───────────────┐
   │  ≥30%     │        │  <30% relevant│
   │ relevant  │        │  retry < 2    │
   └─────┬─────┘        └──────┬────────┘
         │                     ▼
         │              ┌──────────────┐
         │              │ Rewrite Node │ ← Reformulate query
         │              └──────┬───────┘
         │                     │
         │                     └──► (back to Retrieve)
         ▼
   ┌────────────────┐
   │ Generator Node │ ← Generate answer with context
   └───────┬────────┘
           ▼
   ┌─────────────────────┐
   │ Hallucination Check │ ← Verify grounding
   └───────┬─────────────┘
           ▼
     ┌───────────┐
     │  Answer   │
     └───────────┘
```

---

## 🚀 Installation

### Prerequisites

- Python 3.10+
- Conda or pip

### Quick Setup

**1. Clone the repository**

```bash
git clone https://github.com/thanguetvnu10092003/DeepSeek-Simple-Chatbot.git
cd DeepSeek-Simple-Chatbot
```

**2. Create environment**

Using Conda (recommended):
```bash
conda env create -f environment.yaml
conda activate chatbot
```

Or using pip:
```bash
pip install -r requirements.txt
```

**3. Configure API keys**

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key
REPLICATE_API_TOKEN=your_replicate_token
```

> **📌 Get your API keys:**
> - Groq API: https://console.groq.com/
> - Replicate: https://replicate.com/account/api-tokens

---

## 📖 Usage

### Start the Application

```bash
python app.py
```

### Access the Interface

Open your browser and navigate to: **http://127.0.0.1:7860**

### Workflow

1. **Upload Documents**
   - Drag and drop one or multiple PDF/PNG/JPG files
   - Enable OCR for scanned PDFs and images (~$0.001/page)

2. **Choose RAG Mode**
   - **Agentic RAG** (default): Smart multi-step agent with self-correction
   - **Traditional RAG**: Fast single-pass hybrid search

3. **Chat with Your Documents**
   - Select specific files or leave empty to search all
   - Ask any question about your documents
   - View agent reasoning steps in the accordion panel

---

## 📁 Project Structure

```
📦 DeepSeek-Simple-Chatbot
├── 📄 app.py               # Gradio UI with mode toggle & reasoning display
├── 📄 agentic_rag.py        # LangGraph Agentic RAG workflow (6 nodes)
├── 📄 rag.py               # RAG system with hybrid search + agentic integration
├── 📄 llm.py               # Groq LLM wrapper + agentic methods
├── 📄 pdf_ocr_loader.py    # PDF loader with OCR support
├── 📄 requirements.txt     # Python dependencies
├── 📄 environment.yaml     # Conda environment config
└── 📄 .env                 # API keys (create manually)
```

---

## 💰 Pricing

| Operation | Cost |
|-----------|------|
| PDF Text Extraction | **Free** |
| OCR per Page | ~$0.001 |
| Image OCR | ~$0.001 |
| LLM (Groq) | **Free** |

> **Note:** Agentic RAG uses ~2-4x more API calls than Traditional RAG due to query routing, document grading, and hallucination checking. Groq API is free so this has no cost impact.

---

## 📜 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

---

<div align="center">

**Made with ❤️ by [thanguetvnu10092003](https://github.com/thanguetvnu10092003)**

</div>
