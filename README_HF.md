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

Chat với tài liệu PDF và ảnh sử dụng AI (RAG + DeepSeek OCR).

## Tính năng

- Upload nhiều file PDF/ảnh cùng lúc
- OCR cho PDF scan và ảnh (DeepSeek)
- Hybrid Search (Semantic + BM25)
- Multi-file query selection
- Dual Vectorstore (Small/Large chunks)

## Sử dụng

1. Kéo thả file PDF hoặc ảnh
2. Bật OCR nếu cần (có phí)
3. Chat với tài liệu

## API Keys Required

Cần thêm secrets trong Space settings:
- `GROQ_API_KEY`
- `REPLICATE_API_TOKEN`
