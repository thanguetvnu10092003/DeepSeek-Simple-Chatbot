# FastAPI Backend - PDF RAG DeepSeek OCR Chatbot

import time
import os
import logging
import json
from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
from typing import Optional, List
from pathlib import Path
from datetime import datetime

import replicate
from langchain_chroma import Chroma
from langchain_core.documents import Document

from rag import LangChainPDFRAG
from chat_history import ChatHistoryManager

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
MAX_FILE_SIZE_MB = 50
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024
UPLOAD_DIR = "./uploads"

# Initialize systems
rag_system = LangChainPDFRAG()
history_manager = ChatHistoryManager()
os.makedirs(UPLOAD_DIR, exist_ok=True)

app = FastAPI(title="PDF RAG DeepSeek OCR Chatbot")

# Serve static files
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")


# ===== Pydantic Models =====

class ChatRequest(BaseModel):
    message: str
    conversation_id: Optional[str] = None
    selected_files: Optional[List[str]] = None
    use_agentic: bool = True


class ConversationCreate(BaseModel):
    pass


# ===== API Routes =====

@app.get("/")
async def serve_index():
    response = FileResponse("static/index.html")
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response


@app.get("/api/files")
async def list_files():
    """Danh sách files đã upload vào vectorstore và thông tin tồn tại vật lý"""
    files = rag_system.get_all_files()
    file_info = []
    for f in files:
        has_file = os.path.exists(os.path.join(UPLOAD_DIR, f))
        file_info.append({"name": f, "hasPreview": has_file})
    return {"files": file_info}


@app.delete("/api/files/{filename}")
async def delete_file(filename: str):
    """Xóa file khỏi vectorstore và thư mục uploads"""
    # 1. Xóa khỏi ChromaDB & BM25
    rag_system.delete_file(filename)
    
    # 2. Xóa file vật lý
    file_path = os.path.join(UPLOAD_DIR, filename)
    if os.path.exists(file_path):
        try:
            os.remove(file_path)
            logger.info(f"Deleted physical file: {file_path}")
        except Exception as e:
            logger.error(f"Cannot delete physical file {filename}: {e}")
            
    return {"status": "ok", "message": f"Deleted {filename}"}


@app.get("/api/conversations")
async def list_conversations():
    """Danh sách cuộc trò chuyện"""
    convs = history_manager.list_conversations()
    return {"conversations": convs}


@app.post("/api/conversations")
async def create_conversation():
    """Tạo cuộc trò chuyện mới"""
    conv_id = history_manager.create_conversation()
    return {"id": conv_id, "title": "New Chat"}


@app.get("/api/conversations/{conv_id}")
async def get_conversation(conv_id: str):
    """Load cuộc trò chuyện"""
    data = history_manager.load_conversation(conv_id)
    if not data:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return data


@app.delete("/api/conversations/{conv_id}")
async def delete_conversation(conv_id: str):
    """Xóa cuộc trò chuyện"""
    history_manager.delete_conversation(conv_id)
    return {"status": "ok"}


@app.post("/api/upload")
async def upload_files(
    files: List[UploadFile] = File(...),
    enable_ocr: bool = Form(False)
):
    """Upload và xử lý files"""
    results = []
    existing_db_files = rag_system.get_all_files()

    for upload_file in files:
        file_name = upload_file.filename
        file_ext = Path(file_name).suffix.lower()

        # Validate extension
        if file_ext not in ['.pdf', '.png', '.jpg', '.jpeg']:
            results.append({"name": file_name, "status": "error", "message": "Unsupported format"})
            continue

        # Check duplicate
        if file_name in existing_db_files:
            results.append({"name": file_name, "status": "skipped", "message": "File already exists"})
            continue

        # Image without OCR
        if file_ext in ['.png', '.jpg', '.jpeg'] and not enable_ocr:
            results.append({"name": file_name, "status": "needs_ocr", "message": "OCR required"})
            continue

        # Save temp file
        temp_path = os.path.join(UPLOAD_DIR, file_name)
        try:
            content = await upload_file.read()
            if len(content) > MAX_FILE_SIZE_BYTES:
                results.append({"name": file_name, "status": "error", "message": f"File too large (>{MAX_FILE_SIZE_MB}MB)"})
                continue

            with open(temp_path, "wb") as f:
                f.write(content)

            start_time = time.time()

            if file_ext == '.pdf':
                num_chunks, ocr_pages, skipped_pages = rag_system.add_pdf(temp_path, enable_ocr)
                elapsed = time.time() - start_time

                if num_chunks == 0:
                    results.append({
                        "name": file_name, "status": "warning",
                        "message": f"OCR needed ({skipped_pages} scanned pages)"
                    })
                else:
                    results.append({
                        "name": file_name, "status": "ok",
                        "chunks": num_chunks, "ocr_pages": ocr_pages,
                        "time": f"{elapsed:.1f}s"
                    })

            elif file_ext in ['.png', '.jpg', '.jpeg']:
                # OCR image via Replicate
                max_retries = 3
                text = ""
                for retry in range(max_retries):
                    try:
                        with open(temp_path, "rb") as f:
                            text = replicate.run(
                                "lucataco/deepseek-ocr:cb3b474fbfc56b1664c8c7841550bccecbe7b74c30e45ce938ffca1180b4dff5",
                                input={"image": f}
                            )
                        break
                    except Exception as api_error:
                        if "429" in str(api_error) and retry < max_retries - 1:
                            time.sleep((retry + 1) * 5)
                        else:
                            raise api_error

                doc = Document(page_content=text, metadata={"filename": file_name, "type": "image"})
                splits_small = rag_system.splitter_small.split_documents([doc])
                splits_large = rag_system.splitter_large.split_documents([doc])

                if not rag_system.vectorstore:
                    rag_system.vectorstore = Chroma.from_documents(
                        splits_small, rag_system.embeddings,
                        persist_directory=rag_system.persist_directory, collection_name="small_chunks"
                    )
                else:
                    rag_system.vectorstore.add_documents(splits_small)

                large_dir = rag_system.persist_directory + "_large"
                if not rag_system.vectorstore_large:
                    rag_system.vectorstore_large = Chroma.from_documents(
                        splits_large, rag_system.embeddings,
                        persist_directory=large_dir, collection_name="large_chunks"
                    )
                else:
                    rag_system.vectorstore_large.add_documents(splits_large)

                rag_system._update_bm25(splits_small)
                elapsed = time.time() - start_time

                results.append({
                    "name": file_name, "status": "ok",
                    "chunks": len(splits_small), "ocr_pages": 1,
                    "time": f"{elapsed:.1f}s"
                })

        except Exception as e:
            logger.error(f"Error processing {file_name}: {e}")
            results.append({"name": file_name, "status": "error", "message": str(e)[:80]})
        finally:
            # We don't delete the uploaded file so we can serve it later for preview.
            pass

    all_files_raw = rag_system.get_all_files()
    all_files = []
    for f in all_files_raw:
        has_file = os.path.exists(os.path.join(UPLOAD_DIR, f))
        all_files.append({"name": f, "hasPreview": has_file})
    
    return {"results": results, "all_files": all_files}


@app.post("/api/chat")
async def chat(request: ChatRequest):
    """Chat endpoint - returns SSE stream"""
    message = request.message.strip()
    if not message:
        raise HTTPException(status_code=400, detail="Message is empty")

    conv_id = request.conversation_id
    if not conv_id:
        conv_id = history_manager.create_conversation()

    # Load existing messages
    conv_data = history_manager.load_conversation(conv_id)
    messages = conv_data.get("messages", []) if conv_data else []

    # Check if this is the first message (for auto-naming)
    is_first_message = len(messages) == 0

    # Add user message
    messages.append({"role": "user", "content": message})

    def generate():
        try:
            target_files = request.selected_files if request.selected_files else None
            mode_label = "Agentic RAG" if request.use_agentic else "Traditional RAG"

            start_time = time.time()
            reasoning_steps = []

            if request.use_agentic:
                answer, sources, reasoning_steps = rag_system.query_agentic(
                    message, target_files=target_files
                )
            else:
                answer, sources = rag_system.query(message, target_files=target_files)

            elapsed = time.time() - start_time

            # Build response with sources
            response = answer + "\n\n"

            if sources:
                response += "---\n**References:**\n"
                file_sources = {}
                for doc in sources[:15]:
                    meta = doc.metadata
                    fname = meta.get('filename', 'Unknown')
                    page = meta.get('page', 'N/A')
                    if fname not in file_sources:
                        file_sources[fname] = []
                    file_sources[fname].append(page)

                for fname, pages in file_sources.items():
                    if len(pages) > 3:
                        pages_sorted = sorted([p for p in pages if isinstance(p, int)])
                        if pages_sorted:
                            response += f"• {fname} - Page {pages_sorted[0]}-{pages_sorted[-1]}\n"
                        else:
                            response += f"• {fname}\n"
                    else:
                        pages_str = ", ".join(str(p) for p in sorted(set(pages)) if p != 'N/A')
                        if pages_str:
                            response += f"• {fname} - Page {pages_str}\n"
                        else:
                            response += f"• {fname}\n"

            response += f"\n*Process time: {elapsed:.2f}s | Mode: {mode_label}*"

            # Save messages
            messages.append({"role": "assistant", "content": response})
            history_manager.save_conversation(conv_id, messages)

            # Auto-naming
            title = None
            if is_first_message:
                try:
                    title = rag_system.llm.generate_chat_title(message)
                    history_manager.update_title(conv_id, title)
                except Exception as e:
                    logger.warning(f"Auto-naming failed: {e}")
                    title = message[:50] + ("..." if len(message) > 50 else "")
                    history_manager.update_title(conv_id, title)

            # Send complete response as SSE
            result = {
                "conversation_id": conv_id,
                "response": response,
                "title": title,
                "reasoning_steps": reasoning_steps if reasoning_steps else [],
                "mode": mode_label,
            }
            yield f"data: {json.dumps(result, ensure_ascii=False)}\n\n"

        except Exception as e:
            logger.error(f"Chat error: {e}")
            error_result = {
                "conversation_id": conv_id,
                "response": f"Error: {str(e)}",
                "title": None,
                "reasoning_steps": [],
                "mode": "Error",
            }
            yield f"data: {json.dumps(error_result, ensure_ascii=False)}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
