import time
import shutil
import os
import logging

import gradio as gr
import replicate

from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from rag import LangChainPDFRAG

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

load_dotenv()

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
    
    # Check file size
    if file_size > MAX_FILE_SIZE_BYTES:
        size_mb = file_size / (1024 * 1024)
        return False, f"**File quá lớn!**\n\nKích thước: {size_mb:.1f}MB\nTối đa cho phép: {MAX_FILE_SIZE_MB}MB"
    
    # Check file extension
    valid_extensions = ['.pdf', '.png', '.jpg', '.jpeg']
    file_ext = Path(file_path).suffix.lower()
    if file_ext not in valid_extensions:
        return False, f"**Định dạng không hỗ trợ!**\n\nChỉ hỗ trợ: {', '.join(valid_extensions)}"
    
    return True, ""


def process_file(file, enable_ocr, rag_system, uploaded_files, progress=gr.Progress()):
    """Process uploaded file"""
    # Validate file first
    is_valid, error_message = validate_file(file)
    if not is_valid:
        return error_message, gr.update(), rag_system, uploaded_files, gr.update()

    start_time = time.time()
    file_path = file.name
    file_name = Path(file_path).name
    
    # Check for duplicate file
    existing_files = rag_system.get_all_files()
    if file_name in existing_files:
        return f"**File đã tồn tại!**\n\nFile **{file_name}** đã được upload trước đó.\nVui lòng chọn file khác hoặc đổi tên file.", gr.update(), rag_system, uploaded_files, gr.update()

    try:
        if file_path.lower().endswith('.pdf'):
            progress(0, desc="Bắt đầu phân tích PDF...")
            num_chunks, ocr_pages, skipped_pages = rag_system.add_pdf(file_path, enable_ocr, progress)
            elapsed = time.time() - start_time

            if num_chunks == 0:
                result = f"**Cảnh báo: {file_name}**\n\n"
                result += f"Không có text được trích xuất!\n"
                result += f"Trang bị bỏ qua: **{skipped_pages}** (PDF scan)\n\n"
                result += f"**Giải pháp:**\n"
                result += f"-Bật **'Sử dụng OCR trả phí'** để đọc PDF scan\n"
                result += f"-Chi phí: ~${skipped_pages * 0.001:.4f}\n"
                return result, gr.update(), rag_system, uploaded_files, gr.update()

            uploaded_files.append({
                "name": file_name,
                "type": "PDF",
                "chunks": num_chunks,
                "ocr_used": ocr_pages > 0,
                "ocr_pages": ocr_pages,
                "skipped_pages": skipped_pages,
                "time": datetime.now().strftime("%H:%M:%S")
            })

            result = f"**Đã xử lý: {file_name}**\n\n"
            result += f"Số chunks (small): **{num_chunks}**\n"
            result += f"Thời gian: **{elapsed:.2f}s**\n"

            if ocr_pages > 0:
                result += f"**OCR được sử dụng:** {ocr_pages} trang\n"
                result += f"Chi phí API: ~${ocr_pages * 0.001:.4f}\n"

            if skipped_pages > 0:
                result += f"**Trang bị bỏ qua:** {skipped_pages} (PDF scan, OCR tắt)\n"

            if ocr_pages == 0 and skipped_pages == 0:
                result += f"**Không cần OCR** (PDF text thông thường) - MIỄN PHÍ \n"

            result += "\nBạn có thể bắt đầu chat ngay bây giờ!"

        elif file_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            if not enable_ocr:
                return "**Không thể xử lý ảnh khi OCR bị tắt!**\n\nVui lòng bật 'Sử dụng OCR trả phí' để xử lý ảnh.", gr.update(), rag_system, uploaded_files

            progress(0, desc="Đang OCR ảnh...")

            with open(file_path, "rb") as f:
                text = replicate.run(
                    "lucataco/deepseek-ocr:cb3b474fbfc56b1664c8c7841550bccecbe7b74c30e45ce938ffca1180b4dff5",
                    input={"image": f}
                )

            doc = Document(page_content=text, metadata={"filename": file_name, "type": "image"})
            splits_small = rag_system.splitter_small.split_documents([doc])
            splits_large = rag_system.splitter_large.split_documents([doc])

            if not rag_system.vectorstore:
                rag_system.vectorstore = Chroma.from_documents(
                    splits_small, rag_system.embeddings,
                    persist_directory=rag_system.persist_directory,
                    collection_name="small_chunks"
                )
            else:
                rag_system.vectorstore.add_documents(splits_small)

            large_dir = rag_system.persist_directory + "_large"
            if not rag_system.vectorstore_large:
                rag_system.vectorstore_large = Chroma.from_documents(
                    splits_large, rag_system.embeddings,
                    persist_directory=large_dir,
                    collection_name="large_chunks"
                )
            else:
                rag_system.vectorstore_large.add_documents(splits_large)

            uploaded_files.append({
                "name": file_name,
                "type": "Image",
                "chunks": len(splits_small),
                "ocr_used": True,
                "time": datetime.now().strftime("%H:%M:%S")
            })

            elapsed = time.time() - start_time
            result = f"**Đã OCR: {file_name}**\n\n"
            result += f"Thời gian: **{elapsed:.2f}s**\n"
            result += f"Số chunks: **{len(splits_small)}**\n"
            result += f"**OCR được sử dụng:** 1 ảnh\n"
            result += f"Chi phí API: ~$0.0010\n\n"
            result += "Bạn có thể chat về nội dung ảnh này!"

        else:
            return "Chỉ hỗ trợ PDF, PNG, JPG!", gr.update(), rag_system, uploaded_files

        # Build file list display with two sections
        all_db_files = rag_system.get_all_files()
        session_file_names = [f['name'] for f in uploaded_files]
        existing_files = [f for f in all_db_files if f not in session_file_names]
        
        file_list = ""
        
        # Section 1: Files uploaded in current session
        if uploaded_files:
            file_list += "### File vừa upload:\n\n"
            for idx, f in enumerate(uploaded_files, 1):
                file_list += f"**{idx}. {f['name']}** ({f['type']})\n"
                file_list += f"   - {f['chunks']} chunks | {f['time']}\n"

                if f.get("ocr_used") and f['type'] == 'PDF':
                    file_list += f"   - OCR: {f.get('ocr_pages', 0)} trang\n"

                if f.get('skipped_pages', 0) > 0:
                    file_list += f"   - Bỏ qua: {f['skipped_pages']} trang\n"

                file_list += "\n"
        
        # Section 2: Files already in database
        if existing_files:
            file_list += "### File đã có sẵn:\n\n"
            for idx, fname in enumerate(existing_files, 1):
                file_list += f"{idx}. {fname}\n\n"

        # Build file choices for dropdown - get ALL files from database (includes old files)
        all_db_files = rag_system.get_all_files()
        
        # Multiselect: return file list as choices, empty as default value (= search all)
        return result, gr.update(value=file_list), rag_system, uploaded_files, gr.update(choices=all_db_files, value=[])

    except Exception as e:
        logger.error(f"Error processing file: {e}")
        return f"**Lỗi:** {str(e)}", gr.update(), rag_system, uploaded_files, gr.update()


def chat_response(message, history, rag_system, selected_files):
    """Handle chat messages with optional file filter (supports multiple files)"""
    if not message.strip():
        return history, rag_system

    history.append({"role": "user", "content": message})
    
    # Show which files are being queried
    if selected_files and len(selected_files) > 0:
        if len(selected_files) == 1:
            history.append({"role": "assistant", "content": f"Đang tìm kiếm trong **{selected_files[0]}**..."})
        else:
            history.append({"role": "assistant", "content": f"Đang tìm kiếm trong **{len(selected_files)}** file..."})
    else:
        history.append({"role": "assistant", "content": "Đang phân tích câu hỏi (tất cả file)..."})
    yield history, rag_system

    start_time = time.time()
    
    # Pass selected files to query
    # If empty list or None -> search all files
    target_files = selected_files if selected_files and len(selected_files) > 0 else None
    answer, sources = rag_system.query(message, target_files=target_files)
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

    response += f"\n*Thời gian xử lý: {elapsed:.2f}s*"

    history[-1]["content"] = response
    yield history, rag_system


def clear_chat():
    """Clear chat history only"""
    return []




# Create RAG system instance - shared across sessions
initial_rag = create_rag_system()

with gr.Blocks(theme=gr.themes.Soft(), title="PDF RAG DeepSeekOCR Chatbot", css="""
    .status-box {padding: 15px; border-radius: 8px; background: #1e293b; color: #f1f5f9;}
    .file-list-box {padding: 10px; border-radius: 8px; background: #334155; color: #f1f5f9; min-height: 150px;}
    .info-box {padding: 12px; border-radius: 8px; background: #0f172a; color: #cbd5e1; border: 2px solid #3b82f6; margin: 10px 0;}
""") as demo:
    # Session state - không dùng global variables
    rag_state = gr.State(lambda: initial_rag)
    files_state = gr.State([])
    
    gr.Markdown("""
    # PDF RAG DeepSeekOCR Chatbot
    """)

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("###  Upload Tài Liệu")

            file_input = gr.File(
                label=f"Kéo thả file vào đây (tối đa {MAX_FILE_SIZE_MB}MB)",
                file_types=[".pdf", ".png", ".jpg", ".jpeg"],
                file_count="single"
            )

            ocr_toggle = gr.Checkbox(
                label=" Sử dụng OCR trả phí",
                value=False,
                info="Chi phí: ~$0.001/trang"
            )

            with gr.Row():
                process_btn = gr.Button(" Xử lý File", variant="primary", size="lg")

            status_output = gr.Markdown("", elem_classes="status-box")

            gr.Markdown("---")
            file_list = gr.Markdown("*Chưa có file nào*", elem_classes="file-list-box")
            


        with gr.Column(scale=2):
            gr.Markdown("###  Chat với Tài Liệu")
            
            # File filter dropdown - supports multiple selection
            file_filter = gr.Dropdown(
                choices=[],
                value=[],
                multiselect=True,
                label="Chọn file để hỏi",
                info="Chọn 1 hoặc nhiều file. Để trống = tìm tất cả file"
            )

            chatbot = gr.Chatbot(
                height=500,
                type="messages",
                show_copy_button=True
            )

            with gr.Row():
                msg_input = gr.Textbox(
                    placeholder="Nhập câu hỏi của bạn... (Enter để gửi)",
                    show_label=False,
                    scale=4,
                    container=False
                )
                submit_btn = gr.Button(" Gửi", scale=1, variant="primary")

            with gr.Row():
                clear_chat_btn = gr.Button(" Xóa Chat")

    # Event handlers với state
    process_btn.click(
        fn=process_file,
        inputs=[file_input, ocr_toggle, rag_state, files_state],
        outputs=[status_output, file_list, rag_state, files_state, file_filter]
    )

    msg_input.submit(
        fn=chat_response,
        inputs=[msg_input, chatbot, rag_state, file_filter],
        outputs=[chatbot, rag_state]
    ).then(
        fn=lambda: "",
        outputs=[msg_input]
    )

    submit_btn.click(
        fn=chat_response,
        inputs=[msg_input, chatbot, rag_state, file_filter],
        outputs=[chatbot, rag_state]
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
            status = "**Chatbot đã sẵn sàng!**\n\n Bật/tắt OCR theo nhu cầu để tiết kiệm chi phí."
        
        # Multiselect: choices = file list, value = empty (search all)
        return status, file_list_text, gr.update(choices=existing_files, value=[])

    demo.load(
        fn=load_initial_state,
        outputs=[status_output, file_list, file_filter]
    )

if __name__ == "__main__":
    logger.info("=" * 50)
    logger.info(" Đang khởi động PDF RAG Chatbot...")
    logger.info("️ Optional OCR - Tiết kiệm chi phí thông minh")
    logger.info("=" * 50)

    demo.launch(
        share=False,
        server_name="127.0.0.1",
        server_port=7860
    )