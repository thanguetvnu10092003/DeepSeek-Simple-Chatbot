<div align="center">

# 📚 PDF RAG Chatbot with DeepSeek OCR

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Gradio](https://img.shields.io/badge/Gradio-UI-FF6F00?style=for-the-badge&logo=gradio&logoColor=white)](https://gradio.app)
[![Hugging Face](https://img.shields.io/badge/🤗%20Hugging%20Face-Spaces-yellow?style=for-the-badge)](https://huggingface.co/spaces/toanthangle/pdf-rag-deepseek-ocr-chatbot)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

**An intelligent chatbot that enables conversations with PDF documents and images using RAG (Retrieval-Augmented Generation) and DeepSeek OCR.**

### 🚀 [Try the Live Demo](https://huggingface.co/spaces/toanthangle/pdf-rag-deepseek-ocr-chatbot)

[Features](#-features) • [Installation](#-installation) • [Usage](#-usage) • [Project Structure](#-project-structure) • [Pricing](#-pricing)

</div>

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 📁 **Multi-file Upload** | Drag and drop multiple PDF/image files at once |
| 🧠 **Smart Query Classification** | Automatically optimizes retrieval based on query type |
| 🔄 **Dual Vectorstore** | Uses 2 chunk sizes (500 & 1500) for different query types |
| 🔍 **Hybrid Search** | Combines semantic search + BM25 for better results |
| 👁️ **OCR Support** | Process scanned PDFs and images with DeepSeek OCR (via Replicate) |
| 📂 **Multi-file Query** | Select one or multiple specific files to query |
| ⏱️ **Rate Limit Handling** | Automatic retry when hitting Replicate API rate limits |
| 🔒 **Duplicate Detection** | Automatically detects and skips duplicate files |

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
python main.py
```

### Access the Interface

Open your browser and navigate to: **http://127.0.0.1:7860**

### Workflow

1. **Upload Documents**
   - Drag and drop one or multiple PDF/PNG/JPG files
   - Enable OCR for scanned PDFs and images (~$0.001/page)

2. **Chat with Your Documents**
   - Select specific files or leave empty to search all
   - Ask any question about your documents

---

## 📁 Project Structure

```
📦 DeepSeek-Simple-Chatbot
├── 📄 main.py              # Gradio UI and file upload handling
├── 📄 rag.py               # RAG system with hybrid search
├── 📄 llm.py               # Groq LLM wrapper
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

---

## 📜 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

---

<div align="center">

**Made with ❤️ by [thanguetvnu10092003](https://github.com/thanguetvnu10092003)**

</div>
