import time
import shutil
import os
import gradio as gr
import replicate

from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from rag import LangChainPDFRAG


load_dotenv()


def reset_database():
    dirs_to_clean = ["./chroma_db", "./chroma_db_large", "./temp_ocr"]
    for dir_path in dirs_to_clean:
        if os.path.exists(dir_path):
            shutil.rmtree(dir_path)
            print(f"Đã xóa {dir_path}")


reset_database()

rag_system = LangChainPDFRAG()
uploaded_files = []


def process_file(file, enable_ocr, progress=gr.Progress()):
    if file is None:
        return "Vui lòng chọn file!", gr.update()

    start_time = time.time()
    file_path = file.name
    file_name = Path(file_path).name

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
                return result, gr.update()

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
                return "**Không thể xử lý ảnh khi OCR bị tắt!**\n\nVui lòng bật 'Sử dụng OCR trả phí' để xử lý ảnh.", gr.update()

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
            return "Chỉ hỗ trợ PDF, PNG, JPG!", gr.update()

        file_list = "### File đã upload:\n\n"
        for idx, f in enumerate(uploaded_files, 1):
            file_list += f"**{idx}. {f['name']}** ({f['type']})\n"
            file_list += f"   - {f['chunks']} chunks | {f['time']}\n"

            if f.get("ocr_used") and f['type'] == 'PDF':
                file_list += f"   - OCR: {f.get('ocr_pages', 0)} trang\n"

            if f.get('skipped_pages', 0) > 0:
                file_list += f"   - Bỏ qua: {f['skipped_pages']} trang\n"

            file_list += "\n"

        return result, gr.update(value=file_list)

    except Exception as e:
        return f"**Lỗi:** {str(e)}", gr.update()


def chat_response(message, history):
    if not message.strip():
        return history

    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": "Đang phân tích câu hỏi..."})
    yield history

    start_time = time.time()
    answer, sources = rag_system.query(message)
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
    yield history


def clear_chat():
    return []


def clear_all():
    global rag_system, uploaded_files

    dirs_to_clean = ["./chroma_db", "./chroma_db_large"]
    for dir_path in dirs_to_clean:
        if os.path.exists(dir_path):
            shutil.rmtree(dir_path)

    rag_system = LangChainPDFRAG()
    uploaded_files = []

    return (
        "Đã xóa toàn bộ dữ liệu!",
        "*Chưa có file nào*",
        []
    )


with gr.Blocks(theme=gr.themes.Soft(), title="PDF RAG Chatbot", css="""
    .status-box {padding: 15px; border-radius: 8px; background: #1e293b; color: #f1f5f9;}
    .file-list-box {padding: 10px; border-radius: 8px; background: #334155; color: #f1f5f9; min-height: 150px;}
    .info-box {padding: 12px; border-radius: 8px; background: #0f172a; color: #cbd5e1; border: 2px solid #3b82f6; margin: 10px 0;}
""") as demo:
    gr.Markdown("""
    # PDF RAG Chatbot - Smart Adaptive Retrieval

     **Tính năng nâng cao:**
    -  **LLM-based Query Classification** - Tự động phân loại câu hỏi
    -  **Dual Vectorstore** - 2 kích thước chunk (500 & 1500)
    -  **Adaptive Strategy** - Tự động chọn strategy phù hợp
    -  **No Hard-coded Keywords** - Linh hoạt với mọi ngôn ngữ/context
    """)

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("###  Upload Tài Liệu")

            file_input = gr.File(
                label="Kéo thả file vào đây",
                file_types=[".pdf", ".png", ".jpg", ".jpeg"],
                file_count="single"
            )

            ocr_toggle = gr.Checkbox(
                label=" Sử dụng OCR trả phí",
                value=False,
                info="Chi phí: ~$0.001/trang"
            )

            gr.Markdown("""
            <div class="info-box">
             <b>Adaptive Retrieval:</b><br><br>

            Hệ thống tự động:<br>
            •  Phân loại câu hỏi bằng LLM<br>
            •  Chọn chunk size phù hợp<br>
            •  Điều chỉnh số lượng context<br>
            •  Tối ưu prompt cho từng loại<br><br>

            <b>Không cần lo về keyword!</b>
            </div>
            """)

            with gr.Row():
                process_btn = gr.Button(" Xử lý File", variant="primary", size="lg")

            status_output = gr.Markdown("", elem_classes="status-box")

            gr.Markdown("---")
            file_list = gr.Markdown("*Chưa có file nào*", elem_classes="file-list-box")

        with gr.Column(scale=2):
            gr.Markdown("###  Chat với Tài Liệu")

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

    #  Truyền ocr_toggle vào process_file
    process_btn.click(
        fn=process_file,
        inputs=[file_input, ocr_toggle],
        outputs=[status_output, file_list]
    )

    msg_input.submit(
        fn=chat_response,
        inputs=[msg_input, chatbot],
        outputs=[chatbot]
    ).then(
        fn=lambda: "",
        outputs=[msg_input]
    )

    submit_btn.click(
        fn=chat_response,
        inputs=[msg_input, chatbot],
        outputs=[chatbot]
    ).then(
        fn=lambda: "",
        outputs=[msg_input]
    )

    clear_chat_btn.click(
        fn=clear_chat,
        outputs=[chatbot]
    )

    demo.load(
        fn=lambda: ("**Chatbot đã sẵn sàng!**\n\n Bật/tắt OCR theo nhu cầu để tiết kiệm chi phí.",
                    "*Chưa có file nào*"),
        outputs=[status_output, file_list]
    )

if __name__ == "__main__":
    print("\n" + "=" * 50)
    print(" Đang khởi động PDF RAG Chatbot...")
    print(" Database đã được reset hoàn toàn")
    print("️ Optional OCR - Tiết kiệm chi phí thông minh")
    print("=" * 50 + "\n")

    demo.launch(
        share=False,
        server_name="127.0.0.1",
        server_port=7860
    )