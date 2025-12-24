# PDF RAG Chatbot với DeepSeek OCR

Chatbot thông minh cho phép chat với tài liệu PDF và ảnh sử dụng RAG (Retrieval-Augmented Generation) và DeepSeek OCR.

## Tính năng

- **Upload nhiều file** - Kéo thả nhiều file PDF/ảnh cùng lúc
- **Phân loại query thông minh** - Tự động tối ưu retrieval dựa trên loại câu hỏi
- **Dual Vectorstore** - Sử dụng 2 chunk sizes (500 & 1500) cho các loại query khác nhau
- **Hybrid Search** - Kết hợp semantic search + BM25
- **OCR Support** - Xử lý PDF scan và ảnh bằng DeepSeek OCR (Replicate)
- **Multi-file Query** - Chọn 1 hoặc nhiều file cụ thể để hỏi
- **Rate Limit Handling** - Tự động retry khi gặp rate limit từ Replicate API
- **Duplicate Detection** - Tự động phát hiện và bỏ qua file trùng lặp

## Cài đặt

### Yêu cầu
- Python 3.10+
- Conda hoặc pip

### Setup

1. **Clone repository**
   ```bash
   git clone https://github.com/thanguetvnu10092003/DeepSeek-Simple-Chatbot.git
   cd DeepSeek-Simple-Chatbot
   ```

2. **Tạo environment**
   ```bash
   # Sử dụng conda
   conda env create -f environment.yaml
   conda activate chatbot
   
   # Hoặc sử dụng pip
   pip install -r requirements.txt
   ```

3. **Cấu hình API keys**
   
   Tạo file `.env`:
   ```env
   GROQ_API_KEY=your_groq_api_key
   REPLICATE_API_TOKEN=your_replicate_token
   ```

   - Groq API: https://console.groq.com/
   - Replicate: https://replicate.com/account/api-tokens

## Sử dụng

1. **Khởi động**
   ```bash
   python main.py
   ```

2. **Mở trình duyệt**: http://127.0.0.1:7860

3. **Upload tài liệu**
   - Kéo thả 1 hoặc nhiều file PDF/PNG/JPG
   - Bật OCR cho PDF scan và ảnh (~$0.001/trang)

4. **Chat**
   - Chọn file cụ thể hoặc để trống để tìm tất cả
   - Hỏi bất kỳ câu hỏi nào về tài liệu

## Cấu trúc project

```
├── main.py              # Gradio UI và xử lý upload
├── rag.py               # RAG system với hybrid search
├── llm.py               # Groq LLM wrapper
├── pdf_ocr_loader.py    # PDF loader với OCR
├── requirements.txt     # Dependencies
├── environment.yaml     # Conda environment
└── .env                 # API keys (tự tạo)
```

## Chi phí

| Thao tác | Chi phí |
|----------|---------|
| PDF text | Miễn phí |
| OCR/trang | ~$0.001 |
| Ảnh OCR | ~$0.001 |
| LLM (Groq) | Miễn phí |

## License

MIT License
