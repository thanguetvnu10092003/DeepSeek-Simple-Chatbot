---
title: PDF RAG DeepSeek OCR Chatbot
emoji: 📄
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 5.9.1
app_file: app.py
pinned: false
license: mit
---

# PDF RAG DeepSeek OCR Chatbot

Chat với tài liệu PDF và ảnh sử dụng **Agentic RAG** (LangGraph) + DeepSeek OCR.

## Tính năng

- **Agentic RAG**: Agent thông minh với LangGraph (routing, grading, self-correction)
- **Traditional RAG**: Hybrid Search (Semantic + BM25) - chế độ nhanh
- Upload nhiều file PDF/ảnh cùng lúc
- OCR cho PDF scan và ảnh (DeepSeek)
- Multi-file query selection
- Dual Vectorstore (Small/Large chunks)
- Reasoning Steps hiển thị trong UI
- Toggle giữa Agentic/Traditional mode

## Sử dụng

1. Kéo thả file PDF hoặc ảnh
2. Bật OCR nếu cần (có phí)
3. Chọn mode: Agentic RAG (chính xác) hoặc Traditional RAG (nhanh)
4. Chat với tài liệu
5. Xem reasoning steps trong accordion panel

## API Keys Required

Cần thêm secrets trong Space settings:
- `GROQ_API_KEY`
- `REPLICATE_API_TOKEN`
