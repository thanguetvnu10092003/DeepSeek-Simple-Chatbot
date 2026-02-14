# HuggingFace Spaces - PDF RAG DeepSeek OCR Chatbot

import time
import os
import logging
from dotenv import load_dotenv

load_dotenv()

import gradio as gr
import replicate

from datetime import datetime
from pathlib import Path
from langchain_chroma import Chroma
from langchain_core.documents import Document
from rag import LangChainPDFRAG

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
MAX_FILE_SIZE_MB = 50
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024


def create_rag_system():
    """Create a new RAG system instance"""
    return LangChainPDFRAG()


def validate_file(file) -> tuple[bool, str]:
    """Validate uploaded file"""
    if file is None:
        return False, "Vui lòng chọn file!"
    
    file_path = file.name
    file_size = os.path.getsize(file_path)
    
    if file_size > MAX_FILE_SIZE_BYTES:
        size_mb = file_size / (1024 * 1024)
        return False, f"**File quá lớn!**\n\nKích thước: {size_mb:.1f}MB\nTối đa cho phép: {MAX_FILE_SIZE_MB}MB"
    
    valid_extensions = ['.pdf', '.png', '.jpg', '.jpeg']
    file_ext = Path(file_path).suffix.lower()
    if file_ext not in valid_extensions:
        return False, f"**Định dạng không hỗ trợ!**\n\nChỉ hỗ trợ: {', '.join(valid_extensions)}"
    
    return True, ""


def process_files(files, enable_ocr, rag_system, uploaded_files, progress=gr.Progress()):
    """Process multiple uploaded files with OCR detection"""
    if not files or len(files) == 0:
        return "Vui lòng chọn ít nhất 1 file!", gr.update(), rag_system, uploaded_files, gr.update()
    
    if not isinstance(files, list):
        files = [files]
    
    existing_db_files = rag_system.get_all_files()
    results = []
    skipped_duplicates = []
    files_needing_ocr = []
    total_processed = 0
    
    for idx, file in enumerate(files):
        is_valid, error_message = validate_file(file)
        if not is_valid:
            results.append(f"[LOI] **{Path(file.name).name if file else 'Unknown'}** - Loi")
            continue
        
        file_path = file.name
        file_name = Path(file_path).name
        
        if file_name in existing_db_files:
            skipped_duplicates.append(file_name)
            continue
        
        if file_path.lower().endswith(('.png', '.jpg', '.jpeg')) and not enable_ocr:
            files_needing_ocr.append(file_name)
            continue
        
        start_time = time.time()
        
        try:
            progress((idx / len(files)), desc=f"Xu ly {file_name} ({idx+1}/{len(files)})...")
            
            if file_path.lower().endswith('.pdf'):
                num_chunks, ocr_pages, skipped_pages = rag_system.add_pdf(file_path, enable_ocr, progress)
                elapsed = time.time() - start_time
                
                if num_chunks == 0:
                    results.append(f"[CANH BAO] **{file_name}** - Can OCR ({skipped_pages} trang scan)")
                else:
                    uploaded_files.append({
                        "name": file_name, "type": "PDF", "chunks": num_chunks,
                        "ocr_used": ocr_pages > 0, "ocr_pages": ocr_pages,
                        "skipped_pages": skipped_pages, "time": datetime.now().strftime("%H:%M:%S")
                    })
                    r = f"[OK] **{file_name}** - {num_chunks} chunks"
                    if ocr_pages > 0:
                        r += f" (OCR: {ocr_pages} trang)"
                    r += f" [{elapsed:.1f}s]"
                    results.append(r)
                    total_processed += 1
            
            elif file_path.lower().endswith(('.png', '.jpg', '.jpeg')):
                max_retries = 3
                for retry in range(max_retries):
                    try:
                        with open(file_path, "rb") as f:
                            text = replicate.run(
                                "lucataco/deepseek-ocr:cb3b474fbfc56b1664c8c7841550bccecbe7b74c30e45ce938ffca1180b4dff5",
                                input={"image": f}
                            )
                        break
                    except Exception as api_error:
                        if "429" in str(api_error) and retry < max_retries - 1:
                            wait_time = (retry + 1) * 5
                            progress((idx / len(files)), desc=f"Rate limit - doi {wait_time}s...")
                            time.sleep(wait_time)
                        else:
                            raise api_error
                
                doc = Document(page_content=text, metadata={"filename": file_name, "type": "image"})
                splits_small = rag_system.splitter_small.split_documents([doc])
                splits_large = rag_system.splitter_large.split_documents([doc])
                
                if not rag_system.vectorstore:
                    rag_system.vectorstore = Chroma.from_documents(splits_small, rag_system.embeddings,
                        persist_directory=rag_system.persist_directory, collection_name="small_chunks")
                else:
                    rag_system.vectorstore.add_documents(splits_small)
                
                large_dir = rag_system.persist_directory + "_large"
                if not rag_system.vectorstore_large:
                    rag_system.vectorstore_large = Chroma.from_documents(splits_large, rag_system.embeddings,
                        persist_directory=large_dir, collection_name="large_chunks")
                else:
                    rag_system.vectorstore_large.add_documents(splits_large)
                
                rag_system._update_bm25(splits_small)
                elapsed = time.time() - start_time
                
                uploaded_files.append({
                    "name": file_name, "type": "Image", "chunks": len(splits_small),
                    "ocr_used": True, "time": datetime.now().strftime("%H:%M:%S")
                })
                results.append(f"[OK] **{file_name}** - {len(splits_small)} chunks (OCR) [{elapsed:.1f}s]")
                total_processed += 1
                
        except Exception as e:
            logger.error(f"Error processing {file_name}: {e}")
            results.append(f"[ERROR] **{file_name}** - Error: {str(e)[:40]}")
    
    progress(1.0, desc="Hoan thanh!")
    
    summary_parts = []
    
    if files_needing_ocr:
        ocr_warning = f"**[CẦN OCR]** Các file sau cần OCR:\n"
        for fname in files_needing_ocr:
            ocr_warning += f"- {fname}\n"
        summary_parts.append(ocr_warning)
    
    if skipped_duplicates:
        dup_warning = f"**[ĐÃ TỒN TẠI]** Bỏ qua: {', '.join(skipped_duplicates)}"
        summary_parts.append(dup_warning)
    
    if results:
        result_text = f"### Kết quả ({total_processed} file xử lý):\n\n"
        for r in results:
            result_text += f"{r}\n\n"
        summary_parts.append(result_text)
    
    if summary_parts:
        summary = "\n\n".join(summary_parts)
    else:
        summary = "Không có file được xử lý."
    
    all_db_files = rag_system.get_all_files()
    session_names = [f['name'] for f in uploaded_files]
    existing_display = [f for f in all_db_files if f not in session_names]
    
    file_list = ""
    if uploaded_files:
        file_list += "### File vừa upload:\n\n"
        for i, f in enumerate(uploaded_files, 1):
            file_list += f"**{i}. {f['name']}** ({f['type']})\n"
            file_list += f"   - {f['chunks']} chunks | {f['time']}\n"
            if f.get("ocr_used"):
                file_list += f"   - OCR: Có\n"
            file_list += "\n"
    
    if existing_display:
        file_list += "### File đã có sẵn:\n\n"
        for i, fname in enumerate(existing_display, 1):
            file_list += f"{i}. {fname}\n\n"
    
    return summary, gr.update(value=file_list), rag_system, uploaded_files, gr.update(choices=all_db_files, value=[])


def chat_response(message, history, rag_system, selected_files, use_agentic):
    """Handle chat messages with optional file filter and RAG mode selection"""
    if not message.strip():
        return history, rag_system, ""

    history.append({"role": "user", "content": message})
    
    mode_label = "Agentic RAG" if use_agentic else "Traditional RAG"
    
    if selected_files and len(selected_files) > 0:
        if len(selected_files) == 1:
            history.append({"role": "assistant", "content": f"[{mode_label}] Đang tìm kiếm trong **{selected_files[0]}**..."})
        else:
            history.append({"role": "assistant", "content": f"[{mode_label}] Đang tìm kiếm trong **{len(selected_files)}** file..."})
    else:
        history.append({"role": "assistant", "content": f"[{mode_label}] Đang phân tích câu hỏi (tất cả file)..."})
    yield history, rag_system, ""

    start_time = time.time()
    
    target_files = selected_files if selected_files and len(selected_files) > 0 else None
    reasoning_text = ""
    
    if use_agentic:
        answer, sources, reasoning_steps = rag_system.query_agentic(message, target_files=target_files)
        # Format reasoning steps for display
        if reasoning_steps:
            reasoning_parts = ["### Agent Reasoning Steps\n"]
            for i, step in enumerate(reasoning_steps, 1):
                reasoning_parts.append(f"{i}. {step}")
            reasoning_text = "\n".join(reasoning_parts)
    else:
        answer, sources = rag_system.query(message, target_files=target_files)
        reasoning_text = "*Traditional RAG mode - no reasoning steps*"

    elapsed = time.time() - start_time

    response = f"{answer}\n\n"

    if sources:
        response += f"---\n**Nguồn tham khảo:**\n"
        file_sources = {}

        for doc in sources[:15]:
            meta = doc.metadata
            filename = meta.get('filename', 'Unknown')
            page = meta.get('page', 'N/A')

            if filename not in file_sources:
                file_sources[filename] = []
            file_sources[filename].append(page)

        for filename, pages in file_sources.items():
            if len(pages) > 3:
                pages_sorted = sorted([p for p in pages if isinstance(p, int)])
                if pages_sorted:
                    response += f"• {filename} - Trang {pages_sorted[0]}-{pages_sorted[-1]}\n"
                else:
                    response += f"• {filename}\n"
            else:
                pages_str = ", ".join(str(p) for p in sorted(set(pages)) if p != 'N/A')
                if pages_str:
                    response += f"• {filename} - Trang {pages_str}\n"
                else:
                    response += f"• {filename}\n"

    response += f"\n*Thời gian xử lý: {elapsed:.2f}s | Mode: {mode_label}*"

    history[-1]["content"] = response
    yield history, rag_system, reasoning_text


def clear_chat():
    """Clear chat history only"""
    return []


# Create RAG system instance
initial_rag = create_rag_system()

CSS = """
    .status-box {padding: 15px; border-radius: 8px; background: #1e293b; color: #f1f5f9;}
    .file-list-box {padding: 10px; border-radius: 8px; background: #334155; color: #f1f5f9; min-height: 150px; max-height: 350px; overflow-y: auto !important;}
    .file-list-box * {overflow: visible !important;}
    .file-list-box div, .file-list-box span {max-height: none !important;}
    .info-box {padding: 12px; border-radius: 8px; background: #0f172a; color: #cbd5e1; border: 2px solid #3b82f6; margin: 10px 0;}
    #file-filter-dropdown ul {max-height: 250px !important; overflow-y: auto !important;}
    #file-filter-dropdown .options {max-height: 250px !important; overflow-y: auto !important;}
    .reasoning-box {padding: 12px; border-radius: 8px; background: #0c1222; color: #94a3b8; border: 1px solid #1e3a5f;
        font-family: 'Consolas', 'Monaco', monospace; font-size: 0.85em; max-height: 300px; overflow-y: auto !important;}
    .reasoning-box .step {padding: 4px 0; border-bottom: 1px solid #1e293b;}
    .mode-badge {display: inline-block; padding: 3px 10px; border-radius: 12px; font-size: 0.8em; font-weight: bold;}
    .mode-agentic {background: linear-gradient(135deg, #6366f1, #8b5cf6); color: white;}
    .mode-traditional {background: linear-gradient(135deg, #059669, #10b981); color: white;}
"""

with gr.Blocks(title="PDF RAG DeepSeekOCR Chatbot", theme=gr.themes.Soft(), css=CSS) as demo:
    rag_state = gr.State(lambda: initial_rag)
    files_state = gr.State([])
    
    gr.Markdown("""
    # PDF RAG DeepSeekOCR Chatbot
    Chat với tài liệu PDF và ảnh sử dụng AI
    """)
    
    with gr.Tabs():
        with gr.Tab("Chat"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### Upload Tài Liệu")
                    
                    file_input = gr.File(
                        label=f"Kéo thả file vào đây (tối đa {MAX_FILE_SIZE_MB}MB)",
                        file_types=[".pdf", ".png", ".jpg", ".jpeg"],
                        file_count="multiple"
                    )
                    
                    ocr_toggle = gr.Checkbox(
                        label="Sử dụng OCR trả phí",
                        value=False,
                        info="Chi phí: ~$0.001/trang"
                    )
                    
                    process_btn = gr.Button("Xử lý File", variant="primary", size="lg")
                    
                    status_output = gr.Markdown("", elem_classes="status-box")
                    
                    gr.Markdown("---")
                    file_list = gr.Markdown("*Chưa có file nào*", elem_classes="file-list-box")
                
                with gr.Column(scale=2):
                    gr.Markdown("### Chat với Tài Liệu")
                    
                    with gr.Row():
                        file_filter = gr.Dropdown(
                            choices=[],
                            value=[],
                            multiselect=True,
                            label="Chọn file để hỏi",
                            info="Chọn 1 hoặc nhiều file. Để trống = tìm tất cả file",
                            max_choices=None,
                            elem_id="file-filter-dropdown",
                            scale=3
                        )
                        agentic_toggle = gr.Checkbox(
                            label="Agentic RAG",
                            value=True,
                            info="Bật để sử dụng agent thông minh (chậm hơn, chính xác hơn)",
                            scale=1
                        )
                    
                    chatbot = gr.Chatbot(height=450, type="messages")
                    
                    with gr.Row():
                        msg_input = gr.Textbox(
                            placeholder="Nhập câu hỏi của bạn... (Enter để gửi)",
                            show_label=False,
                            scale=4,
                            container=False
                        )
                        submit_btn = gr.Button("Gửi", scale=1, variant="primary")
                    
                    with gr.Row():
                        clear_chat_btn = gr.Button("Xóa Chat", scale=1)
                    
                    with gr.Accordion("Agent Reasoning Steps", open=False):
                        reasoning_display = gr.Markdown(
                            "*Gửi câu hỏi với Agentic RAG để xem reasoning steps*",
                            elem_classes="reasoning-box"
                        )
        
        with gr.Tab("Hướng dẫn sử dụng"):
            gr.Markdown("""
## Hướng dẫn sử dụng PDF RAG Chatbot

### 1. Upload tài liệu
- **Kéo thả** hoặc click để chọn file PDF, PNG, JPG
- Có thể upload **nhiều file** cùng lúc
- Kích thước tối đa: **50MB/file**

### 2. Xử lý OCR (tùy chọn)
- **PDF text thường**: Không cần bật OCR (miễn phí)
- **PDF scan/ảnh chụp**: Bật "Sử dụng OCR trả phí"
- Chi phí OCR: ~$0.001/trang

### 3. Chế độ RAG
- **Agentic RAG** (mặc định): Agent thông minh với khả năng:
  - Phân tích và phân loại câu hỏi tự động
  - Chia câu hỏi phức tạp thành nhiều bước
  - Đánh giá chất lượng tài liệu tìm được
  - Tự viết lại câu hỏi nếu kết quả không tốt (tối đa 2 lần)
  - Kiểm tra hallucination trước khi trả lời
  - Hiển thị reasoning steps bên dưới chat
- **Traditional RAG**: Hybrid search truyền thống (nhanh hơn, ít API calls)

### 4. Chat với tài liệu
- Nhập câu hỏi và nhấn Enter hoặc click "Gửi"
- **Chọn file cụ thể**: Dùng dropdown "Chọn file để hỏi"
- **Tìm tất cả file**: Để trống dropdown
- **Xem reasoning**: Mở accordion "Agent Reasoning Steps"

### 5. Lưu ý
- Agentic RAG chậm hơn (~2-4x) nhưng chính xác hơn cho câu hỏi phức tạp
- File đã upload sẽ được lưu cho các session sau
- Hệ thống tự động phát hiện file trùng lặp
- Với ảnh, phải bật OCR thì mới xử lý được
            """)

    process_btn.click(
        fn=process_files,
        inputs=[file_input, ocr_toggle, rag_state, files_state],
        outputs=[status_output, file_list, rag_state, files_state, file_filter]
    )

    msg_input.submit(
        fn=chat_response,
        inputs=[msg_input, chatbot, rag_state, file_filter, agentic_toggle],
        outputs=[chatbot, rag_state, reasoning_display]
    ).then(
        fn=lambda: "",
        outputs=[msg_input]
    )

    submit_btn.click(
        fn=chat_response,
        inputs=[msg_input, chatbot, rag_state, file_filter, agentic_toggle],
        outputs=[chatbot, rag_state, reasoning_display]
    ).then(
        fn=lambda: "",
        outputs=[msg_input]
    )

    clear_chat_btn.click(
        fn=clear_chat,
        outputs=[chatbot]
    )

    def load_initial_state():
        """Load initial state including file list from vectorstore"""
        existing_files = initial_rag.get_all_files()
        
        if existing_files:
            file_list_text = "### File đã upload:\n\n"
            for idx, fname in enumerate(existing_files, 1):
                file_list_text += f"**{idx}. {fname}**\n\n"
            status = f"**Chatbot đã sẵn sàng!**\n\nĐã tìm thấy **{len(existing_files)}** file từ session trước."
        else:
            file_list_text = "*Chưa có file nào*"
            status = "**Chatbot đã sẵn sàng!**\n\nBật/tắt OCR theo nhu cầu để tiết kiệm chi phí."
        
        return status, file_list_text, gr.update(choices=existing_files, value=[])

    demo.load(
        fn=load_initial_state,
        outputs=[status_output, file_list, file_filter]
    )

if __name__ == "__main__":
    demo.launch()
