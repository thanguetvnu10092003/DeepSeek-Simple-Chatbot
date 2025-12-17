# PDF RAG Chatbot - Smart Adaptive Retrieval

A powerful chatbot that enables intelligent conversations with PDF documents and images using RAG (Retrieval-Augmented Generation) and DeepSeek OCR with adaptive retrieval strategies.

## ✨ Features

- **🧠 LLM-based Query Classification** - Automatically classifies questions to optimize retrieval
- **📊 Dual Vectorstore** - Uses two chunk sizes (500 & 1500) for different query types
- **🎯 Adaptive Strategy** - Automatically selects the best retrieval strategy
- **🔤 OCR Support** - Process scanned PDFs and images using DeepSeek OCR
- **💰 Cost Optimization** - Optional OCR to minimize API costs
- **🔄 Smart Caching** - Caches query classifications to reduce API calls

## 🛠️ Installation

### Prerequisites
- Python 3.10+
- Conda (recommended) or pip

### Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd DeepSeek-OCR-Chatbot
   ```

2. **Create environment**
   ```bash
   # Using conda (recommended)
   conda env create -f environment.yaml
   conda activate chatbot
   
   # Or using pip
   pip install -r requirements.txt
   ```

3. **Configure API keys**
   
   Create a `.env` file in the project root:
   ```env
   GROQ_API_KEY=your_groq_api_key_here
   REPLICATE_API_TOKEN=your_replicate_token_here
   ```

   - Get Groq API key: https://console.groq.com/
   - Get Replicate token: https://replicate.com/account/api-tokens

## 🚀 Usage

1. **Start the application**
   ```bash
   python main.py
   ```

2. **Open in browser**
   Navigate to `http://127.0.0.1:7860`

3. **Upload documents**
   - Drag and drop PDF, PNG, or JPG files
   - Enable OCR for scanned documents (costs ~$0.001/page)

4. **Start chatting**
   - Ask questions about your documents
   - The system automatically selects the best retrieval strategy

## 📁 Project Structure

```
├── main.py              # Gradio UI and main application
├── rag.py               # RAG system with adaptive retrieval
├── llm.py               # Groq LLM wrapper with caching
├── pdf_ocr_loader.py    # PDF loading with OCR support
├── requirements.txt     # Python dependencies
├── environment.yaml     # Conda environment
└── .env                 # API keys (create this file)
```

## 🔧 Architecture

### Query Classification
The system uses LLM to classify queries into:
- **Overview** - Summarize multiple files
- **Specific** - Information from one file/topic
- **Exercise** - Problem solving with detailed context
- **Question** - General questions

### Adaptive Retrieval
Based on classification, the system adjusts:
- Chunk size (500 or 1500 characters)
- Number of retrieved chunks (30-60)
- Prompt template optimization

## 💰 Cost Estimation

| Operation | Cost |
|-----------|------|
| Text PDF | Free (no OCR needed) |
| OCR per page | ~$0.001 |
| Image OCR | ~$0.001 |
| LLM queries | Free (Groq) |

## 🔒 Security Notes

- Never commit `.env` file to git
- API keys are excluded via `.gitignore`
- Database files are stored locally and excluded from git

## 📝 License

MIT License

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
