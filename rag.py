import os
import gradio as gr

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_chroma import Chroma
from llm import GroqLLM
from pdf_ocr_loader import OCRPDFLoader

class LangChainPDFRAG:
    def __init__(self, persist_directory="./chroma_db"):
        self.llm = GroqLLM()
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        self.persist_directory = persist_directory

        # Adaptive chunk sizes: tạo 2 vectorstores với chunk size khác nhau
        self.splitter_small = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            separators=["\n\n", "\n", ". ", "! ", "? ", ", ", " ", ""]
        )

        self.splitter_large = RecursiveCharacterTextSplitter(
            chunk_size=1500,
            chunk_overlap=300,
            separators=["\n\n", "\n", ". ", "! ", "? ", ", ", " ", ""]
        )

        if os.path.exists(persist_directory) and os.listdir(persist_directory):
            self.vectorstore = Chroma(
                persist_directory=persist_directory,
                embedding_function=self.embeddings,
                collection_name="small_chunks"
            )

            # Large chunks vectorstore
            large_dir = persist_directory + "_large"
            if os.path.exists(large_dir) and os.listdir(large_dir):
                self.vectorstore_large = Chroma(
                    persist_directory=large_dir,
                    embedding_function=self.embeddings,
                    collection_name="large_chunks"
                )
            else:
                self.vectorstore_large = None
        else:
            self.vectorstore = None
            self.vectorstore_large = None

    def add_pdf(self, pdf_path: str, enable_ocr: bool = True, progress=gr.Progress()):
        progress(0, desc="Đang đọc PDF...")
        loader = OCRPDFLoader(pdf_path, enable_ocr=enable_ocr)
        docs, ocr_pages, skipped_pages = loader.load()

        if not docs:
            return 0, 0, skipped_pages

        progress(0.3, desc="Đang chia nhỏ văn bản (small chunks)...")
        splits_small = self.splitter_small.split_documents(docs)

        progress(0.5, desc="Đang chia nhỏ văn bản (large chunks)...")
        splits_large = self.splitter_large.split_documents(docs)

        progress(0.7, desc="Đang lưu vào database (small)...")
        if self.vectorstore is None:
            self.vectorstore = Chroma.from_documents(
                splits_small,
                self.embeddings,
                persist_directory=self.persist_directory,
                collection_name="small_chunks"
            )
        else:
            self.vectorstore.add_documents(splits_small)

        progress(0.9, desc="Đang lưu vào database (large)...")
        large_dir = self.persist_directory + "_large"
        if self.vectorstore_large is None:
            self.vectorstore_large = Chroma.from_documents(
                splits_large,
                self.embeddings,
                persist_directory=large_dir,
                collection_name="large_chunks"
            )
        else:
            self.vectorstore_large.add_documents(splits_large)

        progress(1.0, desc="Hoàn thành!")
        return len(splits_small), ocr_pages, skipped_pages

    def get_all_files(self):
        if not self.vectorstore:
            return []

        all_data = self.vectorstore.get()
        if not all_data or 'metadatas' not in all_data:
            return []

        files = set()
        for meta in all_data['metadatas']:
            if meta and 'filename' in meta:
                files.add(meta['filename'])
        return list(files)

    def query(self, question: str):
        if not self.vectorstore:
            return "Chưa có tài liệu nào được thêm vào!", []

        # BƯỚC 1: Phân loại câu hỏi bằng LLM
        query_info = self.llm.classify_query(question)

        print(f"Query Analysis: {query_info}")  # Debug

        # BƯỚC 2: Quyết định strategy dựa trên phân loại
        query_type = query_info.get("type", "question")
        complexity = query_info.get("complexity", "medium")
        needs_detail = query_info.get("needs_detail", True)
        needs_all_files = query_info.get("needs_all_files", False)

        # Chọn vectorstore phù hợp
        if query_type == "exercise" or needs_detail:
            # Bài tập/chi tiết → dùng large chunks
            active_vectorstore = self.vectorstore_large if self.vectorstore_large else self.vectorstore
            k_value = 40
            chunks_per_file = 12
        elif query_type == "overview":
            # Tổng quan → dùng small chunks, nhiều file
            active_vectorstore = self.vectorstore
            k_value = 60
            chunks_per_file = 10
        else:
            # Câu hỏi thông thường → balanced
            active_vectorstore = self.vectorstore
            k_value = 30
            chunks_per_file = 8

        retriever = active_vectorstore.as_retriever(
            search_kwargs={"k": k_value}
        )

        all_docs = retriever.invoke(question)

        file_groups = {}
        for doc in all_docs:
            filename = doc.metadata.get('filename', 'Unknown')
            if filename not in file_groups:
                file_groups[filename] = []
            file_groups[filename].append(doc)

        # BƯỚC 3: Đảm bảo có sample từ tất cả file nếu cần
        if needs_all_files:
            all_files = self.get_all_files()
            for filename in all_files:
                if filename not in file_groups:
                    file_specific_docs = active_vectorstore.similarity_search(
                        filename,
                        k=5,
                        filter={"filename": filename}
                    )
                    if file_specific_docs:
                        file_groups[filename] = file_specific_docs

        balanced_docs = []
        for filename, docs in file_groups.items():
            balanced_docs.extend(docs[:chunks_per_file])

        # Bổ sung thêm nếu cần
        min_chunks = 25 if needs_detail else 15
        if len(balanced_docs) < min_chunks:
            for doc in all_docs:
                if doc not in balanced_docs:
                    balanced_docs.append(doc)
                if len(balanced_docs) >= min_chunks:
                    break

        context_parts = []
        for doc in balanced_docs:
            filename = doc.metadata.get('filename', 'Unknown')
            page = doc.metadata.get('page', 'N/A')
            context_parts.append(
                f"[File: {filename} - Trang {page}]\n{doc.page_content}"
            )

        context = "\n\n---\n\n".join(context_parts)

        # BƯỚC 4: Chọn prompt phù hợp với loại câu hỏi
        if query_type == "exercise":
            prompt = ChatPromptTemplate.from_template(
                "Bạn là giáo viên giỏi giải bài tập. Đọc KỸ TOÀN BỘ ngữ cảnh và trả lời CHI TIẾT.\n\n"
                "QUY TẮC:\n"
                "- Đọc TOÀN BỘ đề bài, không bỏ sót yêu cầu\n"
                "- Giải TỪNG CÂU/PHẦN một cách chi tiết\n"
                "- Nếu có bảng/danh sách → điền ĐẦY ĐỦ\n"
                "- Trích dẫn đề bài trước khi giải\n"
                "- Giải thích logic/lý do\n"
                "- Format rõ ràng, dễ đọc\n\n"
                "Ngữ cảnh:\n{context}\n\n"
                "Câu hỏi: {question}\n\n"
                "Trả lời chi tiết:"
            )
        elif query_type == "overview":
            file_list = ", ".join(file_groups.keys())
            prompt = ChatPromptTemplate.from_template(
                "Bạn là trợ lý thông minh, chuyên tổng hợp thông tin.\n\n"
                "Hiện có {num_files} file: {file_list}\n\n"
                "YÊU CẦU:\n"
                "- Tóm tắt nội dung chính của TỪNG file\n"
                "- So sánh/liên hệ nếu có\n"
                "- Trình bày rõ ràng, có cấu trúc\n\n"
                "Ngữ cảnh:\n{context}\n\n"
                "Câu hỏi: {question}\n\n"
                "Trả lời:"
            )
        else:
            # Câu hỏi thông thường
            prompt = ChatPromptTemplate.from_template(
                "Bạn là trợ lý thông minh. Trả lời dựa vào ngữ cảnh.\n\n"
                "LƯU Ý:\n"
                "- Trả lời chính xác, đầy đủ\n"
                "- Trích dẫn nguồn (file, trang)\n"
                "- Nếu không tìm thấy → nói rõ\n\n"
                "Ngữ cảnh:\n{context}\n\n"
                "Câu hỏi: {question}\n\n"
                "Trả lời:"
            )

        # Build chain
        if query_type == "overview":
            chain = (
                    {
                        "context": lambda x: context,
                        "question": RunnablePassthrough(),
                        "num_files": lambda x: len(file_groups),
                        "file_list": lambda x: ", ".join(file_groups.keys())
                    }
                    | prompt | self.llm | StrOutputParser()
            )
        else:
            chain = (
                    {"context": lambda x: context, "question": RunnablePassthrough()}
                    | prompt | self.llm | StrOutputParser()
            )

        answer = chain.invoke(question)

        mentioned_files = set()
        for doc in balanced_docs:
            filename = doc.metadata.get('filename', 'Unknown')
            if filename in answer:
                mentioned_files.add(filename)

        priority_docs = []
        other_docs = []

        for doc in balanced_docs:
            if doc.metadata.get('filename') in mentioned_files:
                priority_docs.append(doc)
            else:
                other_docs.append(doc)

        final_sources = priority_docs + other_docs

        # Thêm thông tin về strategy đã dùng
        strategy_info = f"\n\n*Strategy: {query_type} | Chunks: {k_value} | Detail: {needs_detail}*"

        return answer + strategy_info, final_sources