import os
import logging
from typing import List, Optional

import gradio as gr

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from llm import GroqLLM
from pdf_ocr_loader import OCRPDFLoader

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LangChainPDFRAG:
    def __init__(self, persist_directory="./chroma_db"):
        logger.info("Initializing LangChainPDFRAG...")
        self.llm = GroqLLM()
        
        # Upgraded embedding model - better for multilingual (Vietnamese + German + English)
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        logger.info("Using embedding model: all-mpnet-base-v2 (upgraded)")
        
        self.persist_directory = persist_directory
        
        # Store all documents for BM25
        self.all_documents: List[Document] = []
        self.bm25_retriever: Optional[BM25Retriever] = None

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

        self._init_vectorstores()
        logger.info("LangChainPDFRAG initialized successfully")

    def _init_vectorstores(self):
        """Initialize vectorstores from existing data or create empty"""
        if os.path.exists(self.persist_directory) and os.listdir(self.persist_directory):
            logger.info(f"Loading existing vectorstore from {self.persist_directory}")
            self.vectorstore = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings,
                collection_name="small_chunks"
            )
            
            # Load existing documents for BM25
            self._rebuild_bm25_from_vectorstore()

            # Large chunks vectorstore
            large_dir = self.persist_directory + "_large"
            if os.path.exists(large_dir) and os.listdir(large_dir):
                self.vectorstore_large = Chroma(
                    persist_directory=large_dir,
                    embedding_function=self.embeddings,
                    collection_name="large_chunks"
                )
            else:
                self.vectorstore_large = None
        else:
            logger.info("No existing vectorstore found, starting fresh")
            self.vectorstore = None
            self.vectorstore_large = None
    
    def _rebuild_bm25_from_vectorstore(self):
        """Rebuild BM25 index from existing vectorstore"""
        try:
            if self.vectorstore:
                data = self.vectorstore.get(include=["documents", "metadatas"])
                if data and data.get("documents"):
                    self.all_documents = []
                    for i, content in enumerate(data["documents"]):
                        metadata = data["metadatas"][i] if data.get("metadatas") else {}
                        self.all_documents.append(Document(page_content=content, metadata=metadata))
                    
                    if self.all_documents:
                        self.bm25_retriever = BM25Retriever.from_documents(self.all_documents)
                        self.bm25_retriever.k = 20
                        logger.info(f"Rebuilt BM25 index with {len(self.all_documents)} documents")
        except Exception as e:
            logger.warning(f"Failed to rebuild BM25: {e}")

    def _update_bm25(self, new_docs: List[Document]):
        """Update BM25 index with new documents"""
        self.all_documents.extend(new_docs)
        if self.all_documents:
            self.bm25_retriever = BM25Retriever.from_documents(self.all_documents)
            self.bm25_retriever.k = 20
            logger.info(f"Updated BM25 index, total: {len(self.all_documents)} documents")

    def add_pdf(self, pdf_path: str, enable_ocr: bool = True, progress=gr.Progress()):
        """Add a PDF to the vectorstore"""
        logger.info(f"Adding PDF: {pdf_path}, OCR enabled: {enable_ocr}")
        
        progress(0, desc="Đang đọc PDF...")
        loader = OCRPDFLoader(pdf_path, enable_ocr=enable_ocr)
        docs, ocr_pages, skipped_pages = loader.load()

        if not docs:
            logger.warning(f"No documents extracted from {pdf_path}")
            return 0, 0, skipped_pages

        progress(0.3, desc="Đang chia nhỏ văn bản (small chunks)...")
        splits_small = self.splitter_small.split_documents(docs)
        logger.info(f"Created {len(splits_small)} small chunks")

        progress(0.5, desc="Đang chia nhỏ văn bản (large chunks)...")
        splits_large = self.splitter_large.split_documents(docs)
        logger.info(f"Created {len(splits_large)} large chunks")

        progress(0.6, desc="Đang cập nhật BM25 index...")
        self._update_bm25(splits_small)

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
        logger.info(f"PDF added successfully: {len(splits_small)} chunks")
        return len(splits_small), ocr_pages, skipped_pages

    def get_all_files(self):
        """Get list of all files in the vectorstore"""
        if not self.vectorstore:
            return []

        try:
            all_data = self.vectorstore.get()
            if not all_data or 'metadatas' not in all_data:
                return []

            files = set()
            for meta in all_data['metadatas']:
                if meta and 'filename' in meta:
                    files.add(meta['filename'])
            return list(files)
        except Exception as e:
            logger.error(f"Error getting file list: {e}")
            return []

    def _detect_mentioned_files(self, question: str, all_files: list) -> list:
        """Detect which files are mentioned in the question"""
        mentioned = []
        question_lower = question.lower()
        
        for filename in all_files:
            # Check if filename (or part of it) is mentioned
            name_lower = filename.lower()
            name_without_ext = name_lower.rsplit('.', 1)[0]
            
            if name_lower in question_lower or name_without_ext in question_lower:
                mentioned.append(filename)
        
        return mentioned

    def _query_single_file(self, question: str, target_file: str):
        """
        EXCLUSIVE search in a single file only.
        No mixing with other files - prevents information pollution.
        """
        logger.info(f"Single file query: {target_file}")
        
        # Determine which vectorstore to use
        query_info = self.llm.classify_query(question)
        query_type = query_info.get("type", "question")
        needs_detail = query_info.get("needs_detail", True)
        
        if query_type == "exercise" or needs_detail:
            active_vectorstore = self.vectorstore_large if self.vectorstore_large else self.vectorstore
            k_value = 30
        else:
            active_vectorstore = self.vectorstore
            k_value = 20
        
        # Search ONLY in the target file
        try:
            semantic_docs = active_vectorstore.similarity_search(
                question,
                k=k_value,
                filter={"filename": target_file}
            )
            logger.info(f"Semantic search in {target_file}: {len(semantic_docs)} docs")
        except Exception as e:
            logger.error(f"Error in semantic search: {e}")
            semantic_docs = []
        
        # BM25 search filtered to target file
        bm25_docs = []
        if self.bm25_retriever:
            try:
                all_bm25 = self.bm25_retriever.invoke(question)
                bm25_docs = [d for d in all_bm25 if d.metadata.get('filename') == target_file][:k_value]
                logger.info(f"BM25 search in {target_file}: {len(bm25_docs)} docs")
            except Exception as e:
                logger.warning(f"BM25 error: {e}")
        
        # Combine and deduplicate
        all_docs = semantic_docs.copy()
        seen_contents = {d.page_content for d in all_docs}
        for doc in bm25_docs:
            if doc.page_content not in seen_contents:
                all_docs.append(doc)
                seen_contents.add(doc.page_content)
        
        if not all_docs:
            return f"Không tìm thấy thông tin trong file **{target_file}**.", []
        
        # Build context
        context_parts = []
        for doc in all_docs:
            page = doc.metadata.get('page', 'N/A')
            context_parts.append(f"[Trang {page}]\n{doc.page_content}")
        
        context = "\n\n---\n\n".join(context_parts)
        
        # Single-file focused prompt
        prompt = ChatPromptTemplate.from_template(
            f"Bạn là trợ lý thông minh. Trả lời dựa vào nội dung từ file **{target_file}**.\n\n"
            "LƯU Ý:\n"
            "- Chỉ sử dụng thông tin từ file này\n"
            "- Trả lời chính xác, đầy đủ\n"
            "- Trích dẫn trang nếu có\n"
            "- Nếu không tìm thấy → nói rõ\n\n"
            "Ngữ cảnh:\n{context}\n\n"
            "Câu hỏi: {question}\n\n"
            "Trả lời:"
        )
        
        chain = (
            {"context": lambda x: context, "question": RunnablePassthrough()}
            | prompt | self.llm | StrOutputParser()
        )
        
        try:
            answer = chain.invoke(question)
        except Exception as e:
            logger.error(f"Error in chain: {e}")
            return f"Lỗi khi xử lý: {str(e)}", []
        
        strategy_info = f"\n\n*File: {target_file} | Chunks: {len(all_docs)}*"
        return answer + strategy_info, all_docs

    def _query_multiple_files(self, question: str, target_files: list):
        """
        Search across multiple specific files only.
        Combines results from each file with balanced representation.
        """
        logger.info(f"Multi-file query: {target_files}")
        
        # Determine which vectorstore to use
        query_info = self.llm.classify_query(question)
        query_type = query_info.get("type", "question")
        needs_detail = query_info.get("needs_detail", True)
        
        if query_type == "exercise" or needs_detail:
            active_vectorstore = self.vectorstore_large if self.vectorstore_large else self.vectorstore
            k_per_file = 15
        else:
            active_vectorstore = self.vectorstore
            k_per_file = 10
        
        # Search in each target file
        all_docs = []
        for filename in target_files:
            try:
                # Semantic search
                semantic_docs = active_vectorstore.similarity_search(
                    question,
                    k=k_per_file,
                    filter={"filename": filename}
                )
                
                # BM25 search filtered to this file
                bm25_docs = []
                if self.bm25_retriever:
                    all_bm25 = self.bm25_retriever.invoke(question)
                    bm25_docs = [d for d in all_bm25 if d.metadata.get('filename') == filename][:k_per_file]
                
                # Combine and deduplicate
                file_docs = semantic_docs.copy()
                seen_contents = {d.page_content for d in file_docs}
                for doc in bm25_docs:
                    if doc.page_content not in seen_contents:
                        file_docs.append(doc)
                        seen_contents.add(doc.page_content)
                
                all_docs.extend(file_docs)
                logger.info(f"Found {len(file_docs)} chunks from {filename}")
            except Exception as e:
                logger.warning(f"Error searching file {filename}: {e}")
        
        if not all_docs:
            return f"Không tìm thấy thông tin trong các file đã chọn.", []
        
        # Build context
        context_parts = []
        for doc in all_docs:
            filename = doc.metadata.get('filename', 'Unknown')
            page = doc.metadata.get('page', 'N/A')
            context_parts.append(f"[File: {filename} - Trang {page}]\n{doc.page_content}")
        
        context = "\n\n---\n\n".join(context_parts)
        
        # Multi-file focused prompt
        file_list_str = ", ".join(target_files)
        prompt = ChatPromptTemplate.from_template(
            f"Bạn là trợ lý thông minh. Trả lời dựa vào nội dung từ các file: **{file_list_str}**.\n\n"
            "LƯU Ý:\n"
            "- Chỉ sử dụng thông tin từ các file đã chọn\n"
            "- Trả lời chính xác, đầy đủ\n"
            "- Trích dẫn file và trang nếu có\n"
            "- So sánh thông tin giữa các file nếu phù hợp\n"
            "- Nếu không tìm thấy → nói rõ\n\n"
            "Ngữ cảnh:\n{context}\n\n"
            "Câu hỏi: {question}\n\n"
            "Trả lời:"
        )
        
        chain = (
            {"context": lambda x: context, "question": RunnablePassthrough()}
            | prompt | self.llm | StrOutputParser()
        )
        
        try:
            answer = chain.invoke(question)
        except Exception as e:
            logger.error(f"Error in chain: {e}")
            return f"Lỗi khi xử lý: {str(e)}", []
        
        strategy_info = f"\n\n*Files: {len(target_files)} | Chunks: {len(all_docs)}*"
        return answer + strategy_info, all_docs

    def _hybrid_search(self, question: str, k: int = 30) -> List[Document]:
        """
        Custom Hybrid search combining:
        1. Semantic search (embeddings) - understands meaning
        2. BM25 (keyword) - finds exact matches
        
        Weights: 60% semantic, 40% keyword
        """
        if not self.vectorstore:
            return []
        
        # Get semantic results
        semantic_docs = self.vectorstore.similarity_search(question, k=k)
        logger.info(f"Semantic search returned {len(semantic_docs)} documents")
        
        # Get BM25 results if available
        bm25_docs = []
        if self.bm25_retriever:
            self.bm25_retriever.k = k
            bm25_docs = self.bm25_retriever.invoke(question)
            logger.info(f"BM25 search returned {len(bm25_docs)} documents")
        
        # Combine results with weighted scoring
        # Create a score dict: doc_content -> (doc, score)
        doc_scores = {}
        
        # Semantic results get scores based on position (higher = better)
        for i, doc in enumerate(semantic_docs):
            content = doc.page_content
            score = (len(semantic_docs) - i) / len(semantic_docs) * 0.6  # 60% weight
            doc_scores[content] = (doc, score)
        
        # BM25 results add to existing scores or create new entries
        for i, doc in enumerate(bm25_docs):
            content = doc.page_content
            bm25_score = (len(bm25_docs) - i) / len(bm25_docs) * 0.4  # 40% weight
            
            if content in doc_scores:
                existing_doc, existing_score = doc_scores[content]
                doc_scores[content] = (existing_doc, existing_score + bm25_score)
            else:
                doc_scores[content] = (doc, bm25_score)
        
        # Sort by combined score and return top k
        sorted_docs = sorted(doc_scores.values(), key=lambda x: x[1], reverse=True)
        results = [doc for doc, score in sorted_docs[:k]]
        
        logger.info(f"Hybrid search returned {len(results)} documents (combined)")
        return results

    def query(self, question: str, target_files: list = None):
        """
        Query the vectorstore with hybrid retrieval
        
        Args:
            question: The user's question
            target_files: Optional list of filenames - if specified, ONLY search in these files
        """
        if not self.vectorstore:
            return "Chưa có tài liệu nào được thêm vào!", []

        logger.info(f"Processing query: {question[:100]}...")
        
        # Get all available files
        all_files = self.get_all_files()
        
        # EXCLUSIVE MODE: If target_files is specified, only search in those files
        if target_files and len(target_files) > 0:
            if len(target_files) == 1:
                logger.info(f"EXCLUSIVE MODE: Searching only in file: {target_files[0]}")
                return self._query_single_file(question, target_files[0])
            else:
                logger.info(f"MULTI-FILE MODE: Searching in {len(target_files)} files: {target_files}")
                return self._query_multiple_files(question, target_files)
        
        # Detect which files are mentioned in the question
        mentioned_files = self._detect_mentioned_files(question, all_files)
        logger.info(f"Mentioned files: {mentioned_files}")

        # Phân loại câu hỏi bằng LLM
        query_info = self.llm.classify_query(question)
        logger.info(f"Query classification: {query_info}")

        # Quyết định strategy dựa trên phân loại
        query_type = query_info.get("type", "question")
        complexity = query_info.get("complexity", "medium")
        needs_detail = query_info.get("needs_detail", True)
        needs_all_files = query_info.get("needs_all_files", False)

        # Chọn vectorstore và parameters phù hợp
        if query_type == "exercise" or needs_detail:
            active_vectorstore = self.vectorstore_large if self.vectorstore_large else self.vectorstore
            k_value = 40
            chunks_per_file = 12
        elif query_type == "overview":
            active_vectorstore = self.vectorstore
            k_value = 60
            chunks_per_file = 10
        else:
            active_vectorstore = self.vectorstore
            k_value = 30
            chunks_per_file = 8

        # Use hybrid search for better results
        if mentioned_files:
            # Prioritize mentioned files with hybrid search
            logger.info(f"Prioritizing mentioned files with hybrid search: {mentioned_files}")
            file_groups = {}
            
            for filename in mentioned_files:
                try:
                    # Semantic search for mentioned file
                    semantic_docs = active_vectorstore.similarity_search(
                        question,
                        k=chunks_per_file * 2,
                        filter={"filename": filename}
                    )
                    
                    # BM25 search for mentioned file
                    bm25_docs = []
                    if self.bm25_retriever:
                        all_bm25 = self.bm25_retriever.invoke(question)
                        bm25_docs = [d for d in all_bm25 if d.metadata.get('filename') == filename][:chunks_per_file]
                    
                    # Combine and deduplicate
                    combined = semantic_docs.copy()
                    seen_contents = {d.page_content for d in combined}
                    for doc in bm25_docs:
                        if doc.page_content not in seen_contents:
                            combined.append(doc)
                            seen_contents.add(doc.page_content)
                    
                    if combined:
                        file_groups[filename] = combined
                        logger.info(f"Hybrid search: {len(combined)} chunks from {filename}")
                except Exception as e:
                    logger.warning(f"Error in hybrid search for {filename}: {e}")
            
            # Lower priority for other files
            for filename in [f for f in all_files if f not in mentioned_files]:
                try:
                    file_docs = active_vectorstore.similarity_search(
                        question,
                        k=chunks_per_file // 2,
                        filter={"filename": filename}
                    )
                    if file_docs:
                        file_groups[filename] = file_docs
                except Exception as e:
                    logger.warning(f"Error searching file {filename}: {e}")
        else:
            # No specific file - use full hybrid search
            all_docs = self._hybrid_search(question, k=k_value)

            file_groups = {}
            for doc in all_docs:
                filename = doc.metadata.get('filename', 'Unknown')
                if filename not in file_groups:
                    file_groups[filename] = []
                file_groups[filename].append(doc)
            
            # Ensure ALL files have minimum representation
            for filename in all_files:
                if filename not in file_groups or len(file_groups[filename]) < 3:
                    try:
                        file_specific_docs = active_vectorstore.similarity_search(
                            question,
                            k=chunks_per_file,
                            filter={"filename": filename}
                        )
                        if file_specific_docs:
                            if filename not in file_groups:
                                file_groups[filename] = []
                            existing_contents = {d.page_content for d in file_groups[filename]}
                            for doc in file_specific_docs:
                                if doc.page_content not in existing_contents:
                                    file_groups[filename].append(doc)
                            logger.info(f"Added chunks from underrepresented file: {filename}")
                    except Exception as e:
                        logger.warning(f"Error searching for file {filename}: {e}")

        balanced_docs = []
        for filename, docs in file_groups.items():
            balanced_docs.extend(docs[:chunks_per_file])

        # Bổ sung thêm nếu cần
        min_chunks = 25 if needs_detail else 15
        if len(balanced_docs) < min_chunks:
            all_hybrid = self._hybrid_search(question, k=k_value)
            for doc in all_hybrid:
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

        # Chọn prompt phù hợp với loại câu hỏi
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

        try:
            answer = chain.invoke(question)
        except Exception as e:
            logger.error(f"Error in query chain: {e}")
            return f"Lỗi khi xử lý câu hỏi: {str(e)}", []

        mentioned_files_in_answer = set()
        for doc in balanced_docs:
            filename = doc.metadata.get('filename', 'Unknown')
            if filename in answer:
                mentioned_files_in_answer.add(filename)

        priority_docs = []
        other_docs = []

        for doc in balanced_docs:
            if doc.metadata.get('filename') in mentioned_files_in_answer:
                priority_docs.append(doc)
            else:
                other_docs.append(doc)

        final_sources = priority_docs + other_docs

        # Thêm thông tin về strategy đã dùng
        strategy_info = f"\n\n*Strategy: {query_type} | Hybrid Search | Chunks: {len(balanced_docs)}*"

        logger.info(f"Query completed: strategy={query_type}, chunks={len(balanced_docs)}")
        return answer + strategy_info, final_sources