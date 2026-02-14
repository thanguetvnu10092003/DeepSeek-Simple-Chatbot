"""
Agentic RAG Module - LangGraph-based intelligent retrieval agent.

This module upgrades the traditional Hybrid RAG to an Agentic system using LangGraph.
The agent can:
- Route queries (simple vs complex)
- Decompose complex questions into sub-queries
- Grade retrieved documents for relevance
- Rewrite queries when retrieval is poor
- Check for hallucinations in generated answers
- Self-correct up to MAX_RETRIES times
"""

import logging
from typing import List, Optional, TypedDict, Annotated

from langchain_core.documents import Document
from langgraph.graph import StateGraph, END

logger = logging.getLogger(__name__)

MAX_RETRIES = 2


# ===========================
# Agent State
# ===========================

class AgentState(TypedDict):
    """State passed between nodes in the agent graph."""
    question: str                           # Original user question
    sub_questions: List[str]                # Decomposed sub-questions (for complex queries)
    current_sub_question: Optional[str]     # Currently processing sub-question
    target_files: Optional[List[str]]       # Files to search in
    documents: List[Document]               # Retrieved documents
    generation: str                         # Generated answer
    retry_count: int                        # Number of retries so far
    query_type: str                         # "simple" or "complex"
    reasoning_steps: List[str]              # Agent's thinking/reasoning log
    is_relevant: bool                       # Whether retrieved docs are relevant
    sub_answers: List[str]                  # Answers to sub-questions


# ===========================
# Agent Nodes
# ===========================

def route_query(state: AgentState, llm) -> AgentState:
    """
    Router Node: Analyze the query and decide the retrieval strategy.
    - simple: single retrieval
    - complex: decompose into sub-queries
    """
    question = state["question"]
    logger.info(f"[Router] Analyzing query: {question[:80]}...")

    route_result = llm.route_query(question)
    query_type = route_result.get("type", "simple")
    reasoning = route_result.get("reasoning", "No reasoning provided")

    state["query_type"] = query_type
    state["reasoning_steps"].append(f"[Router] Query type: {query_type} - {reasoning}")

    if query_type == "complex":
        sub_questions = llm.decompose_query(question)
        state["sub_questions"] = sub_questions
        state["reasoning_steps"].append(
            f"[Decompose] Split into {len(sub_questions)} sub-questions: {sub_questions}"
        )
    else:
        state["sub_questions"] = [question]

    logger.info(f"[Router] Type: {query_type}, Sub-questions: {len(state['sub_questions'])}")
    return state


def retrieve(state: AgentState, rag_system) -> AgentState:
    """
    Retriever Node: Perform hybrid search for current question(s).
    Uses the existing hybrid search infrastructure.
    """
    target_files = state.get("target_files")
    all_docs = []

    for sub_q in state["sub_questions"]:
        state["current_sub_question"] = sub_q
        logger.info(f"[Retriever] Searching for: {sub_q[:80]}...")

        if target_files and len(target_files) == 1:
            docs = _retrieve_single_file(rag_system, sub_q, target_files[0])
        elif target_files and len(target_files) > 1:
            docs = _retrieve_multiple_files(rag_system, sub_q, target_files)
        else:
            docs = _retrieve_all(rag_system, sub_q)

        all_docs.extend(docs)
        state["reasoning_steps"].append(
            f"[Retrieve] Found {len(docs)} chunks for: '{sub_q[:50]}...'"
        )

    # Deduplicate
    seen = set()
    unique_docs = []
    for doc in all_docs:
        if doc.page_content not in seen:
            seen.add(doc.page_content)
            unique_docs.append(doc)

    state["documents"] = unique_docs
    state["reasoning_steps"].append(
        f"[Retrieve] Total unique chunks: {len(unique_docs)}"
    )
    logger.info(f"[Retriever] Total unique documents: {len(unique_docs)}")
    return state


def _retrieve_single_file(rag_system, question: str, target_file: str) -> List[Document]:
    """Retrieve from a single file using hybrid search."""
    docs = []

    # Determine vectorstore
    query_info = rag_system.llm.classify_query(question)
    needs_detail = query_info.get("needs_detail", True)
    query_type = query_info.get("type", "question")
    
    if query_type == "exercise" or needs_detail:
        vs = rag_system.vectorstore_large if rag_system.vectorstore_large else rag_system.vectorstore
        k = 25
    else:
        vs = rag_system.vectorstore
        k = 15

    if vs:
        try:
            semantic = vs.similarity_search(question, k=k, filter={"filename": target_file})
            docs.extend(semantic)
        except Exception as e:
            logger.warning(f"Semantic search error: {e}")

    if rag_system.bm25_retriever:
        try:
            bm25_all = rag_system.bm25_retriever.invoke(question)
            bm25_filtered = [d for d in bm25_all if d.metadata.get("filename") == target_file][:k]
            docs.extend(bm25_filtered)
        except Exception as e:
            logger.warning(f"BM25 error: {e}")

    return docs


def _retrieve_multiple_files(rag_system, question: str, target_files: List[str]) -> List[Document]:
    """Retrieve from multiple specific files."""
    docs = []
    k_per_file = 12

    for filename in target_files:
        try:
            vs = rag_system.vectorstore_large if rag_system.vectorstore_large else rag_system.vectorstore
            if vs:
                semantic = vs.similarity_search(question, k=k_per_file, filter={"filename": filename})
                docs.extend(semantic)

            if rag_system.bm25_retriever:
                bm25_all = rag_system.bm25_retriever.invoke(question)
                bm25_filtered = [d for d in bm25_all if d.metadata.get("filename") == filename][:k_per_file]
                docs.extend(bm25_filtered)
        except Exception as e:
            logger.warning(f"Error retrieving from {filename}: {e}")

    return docs


def _retrieve_all(rag_system, question: str) -> List[Document]:
    """Retrieve across all files using hybrid search."""
    return rag_system._hybrid_search(question, k=30)


def grade_documents(state: AgentState, llm) -> AgentState:
    """
    Grader Node: Evaluate relevance of retrieved documents.
    Filters out irrelevant documents and decides if re-query is needed.
    """
    question = state["question"]
    documents = state["documents"]

    if not documents:
        state["is_relevant"] = False
        state["reasoning_steps"].append("[Grader] No documents found - marking as not relevant")
        return state

    logger.info(f"[Grader] Grading {len(documents)} documents...")

    grade_results = llm.grade_documents(question, documents)

    relevant_docs = []
    for doc, is_relevant in zip(documents, grade_results):
        if is_relevant:
            relevant_docs.append(doc)

    relevance_ratio = len(relevant_docs) / len(documents) if documents else 0

    state["documents"] = relevant_docs
    state["is_relevant"] = relevance_ratio >= 0.3  # At least 30% relevant
    state["reasoning_steps"].append(
        f"[Grader] {len(relevant_docs)}/{len(documents)} relevant ({relevance_ratio:.0%}). "
        f"Verdict: {'PASS' if state['is_relevant'] else 'FAIL - need rewrite'}"
    )

    logger.info(f"[Grader] Relevant: {len(relevant_docs)}/{len(documents)} ({relevance_ratio:.0%})")
    return state


def rewrite_query(state: AgentState, llm) -> AgentState:
    """
    Query Rewriter Node: Reformulate the query when retrieval is poor.
    """
    question = state["question"]
    state["retry_count"] += 1

    logger.info(f"[Rewriter] Rewriting query (attempt {state['retry_count']})...")

    # Build context from what we have so far
    context = ""
    if state["documents"]:
        context = "\n".join([d.page_content[:200] for d in state["documents"][:3]])

    new_question = llm.rewrite_query(question, context)
    state["sub_questions"] = [new_question]
    state["reasoning_steps"].append(
        f"[Rewrite] Attempt {state['retry_count']}: '{question[:40]}...' → '{new_question[:60]}...'"
    )

    logger.info(f"[Rewriter] New query: {new_question[:80]}")
    return state


def generate(state: AgentState, llm, rag_system) -> AgentState:
    """
    Generator Node: Generate the final answer from relevant documents.
    """
    question = state["question"]
    documents = state["documents"]
    target_files = state.get("target_files")

    if not documents:
        state["generation"] = "Không tìm thấy thông tin phù hợp trong tài liệu."
        state["reasoning_steps"].append("[Generator] No relevant documents - returning empty result")
        return state

    logger.info(f"[Generator] Generating answer from {len(documents)} documents...")

    # Build context
    context_parts = []
    for doc in documents:
        filename = doc.metadata.get("filename", "Unknown")
        page = doc.metadata.get("page", "N/A")
        context_parts.append(f"[File: {filename} - Trang {page}]\n{doc.page_content}")

    context = "\n\n---\n\n".join(context_parts)

    # Determine prompt based on query type
    query_info = llm.classify_query(question)
    query_type = query_info.get("type", "question")

    if target_files and len(target_files) == 1:
        system_msg = (
            f"Bạn là trợ lý thông minh. Trả lời dựa vào nội dung từ file **{target_files[0]}**.\n"
            "Chỉ sử dụng thông tin từ file này. Trích dẫn trang nếu có."
        )
    elif target_files and len(target_files) > 1:
        file_list_str = ", ".join(target_files)
        system_msg = (
            f"Bạn là trợ lý thông minh. Trả lời dựa vào nội dung từ các file: **{file_list_str}**.\n"
            "So sánh thông tin giữa các file nếu phù hợp. Trích dẫn file và trang."
        )
    elif query_type == "exercise":
        system_msg = (
            "Bạn là giáo viên giỏi giải bài tập. Đọc KỸ TOÀN BỘ ngữ cảnh.\n"
            "Giải TỪNG CÂU/PHẦN một cách chi tiết. Trích dẫn đề bài trước khi giải."
        )
    elif query_type == "overview":
        system_msg = (
            "Bạn là trợ lý thông minh, chuyên tổng hợp thông tin.\n"
            "Tóm tắt nội dung chính của TỪNG file. So sánh/liên hệ nếu có."
        )
    else:
        system_msg = (
            "Bạn là trợ lý thông minh. Trả lời dựa vào ngữ cảnh.\n"
            "Trả lời chính xác, đầy đủ. Trích dẫn nguồn (file, trang). "
            "Nếu không tìm thấy → nói rõ."
        )

    prompt_text = f"{system_msg}\n\nNgữ cảnh:\n{context}\n\nCâu hỏi: {question}\n\nTrả lời:"

    try:
        answer = llm._call(prompt_text)
    except Exception as e:
        logger.error(f"[Generator] Error: {e}")
        answer = f"Lỗi khi xử lý: {str(e)}"

    state["generation"] = answer
    state["reasoning_steps"].append(
        f"[Generator] Generated answer ({len(answer)} chars) from {len(documents)} chunks"
    )

    logger.info(f"[Generator] Answer generated ({len(answer)} chars)")
    return state


def hallucination_check(state: AgentState, llm) -> AgentState:
    """
    Self-Check Node: Verify the answer is grounded in the documents.
    """
    answer = state["generation"]
    documents = state["documents"]

    if not answer or not documents:
        state["reasoning_steps"].append("[SelfCheck] Skipped - no answer or no documents")
        return state

    logger.info("[SelfCheck] Checking for hallucinations...")

    is_grounded = llm.check_hallucination(answer, documents)

    if is_grounded:
        state["reasoning_steps"].append("[SelfCheck] PASS - Answer is grounded in documents")
    else:
        state["reasoning_steps"].append("[SelfCheck] WARNING - Potential hallucination detected")
        # Don't fail - just note it. The grader retry loop handles real issues.

    logger.info(f"[SelfCheck] Grounded: {is_grounded}")
    return state


# ===========================
# Conditional Edges
# ===========================

def should_rewrite(state: AgentState) -> str:
    """Decide whether to rewrite query or proceed to generation."""
    if not state["is_relevant"] and state["retry_count"] < MAX_RETRIES:
        logger.info(f"[Decision] Rewriting query (retry {state['retry_count'] + 1}/{MAX_RETRIES})")
        return "rewrite"
    elif not state["is_relevant"]:
        logger.info("[Decision] Max retries reached - generating with what we have")
        state["reasoning_steps"].append(
            f"[Decision] Max retries ({MAX_RETRIES}) reached. Generating with available documents."
        )
        return "generate"
    else:
        return "generate"


# ===========================
# Build LangGraph Workflow
# ===========================

def build_agent_graph(llm, rag_system):
    """
    Build the Agentic RAG workflow using LangGraph.

    Flow:
    route → retrieve → grade → (rewrite → retrieve → grade)* → generate → check → END
    """

    workflow = StateGraph(AgentState)

    # Add nodes with bound dependencies
    workflow.add_node("route", lambda state: route_query(state, llm))
    workflow.add_node("retrieve", lambda state: retrieve(state, rag_system))
    workflow.add_node("grade", lambda state: grade_documents(state, llm))
    workflow.add_node("rewrite", lambda state: rewrite_query(state, llm))
    workflow.add_node("generate", lambda state: generate(state, llm, rag_system))
    workflow.add_node("check", lambda state: hallucination_check(state, llm))

    # Set entry point
    workflow.set_entry_point("route")

    # Define edges
    workflow.add_edge("route", "retrieve")
    workflow.add_edge("retrieve", "grade")

    # Conditional: grade → rewrite or generate
    workflow.add_conditional_edges(
        "grade",
        should_rewrite,
        {
            "rewrite": "rewrite",
            "generate": "generate",
        }
    )

    # Rewrite loops back to retrieve
    workflow.add_edge("rewrite", "retrieve")

    # Generate → check → END
    workflow.add_edge("generate", "check")
    workflow.add_edge("check", END)

    return workflow.compile()


# ===========================
# Public API
# ===========================

def run_agentic_query(
    llm,
    rag_system,
    question: str,
    target_files: Optional[List[str]] = None,
) -> tuple:
    """
    Run an agentic RAG query.

    Args:
        llm: The GroqLLM instance
        rag_system: The LangChainPDFRAG instance
        question: User's question
        target_files: Optional list of files to search

    Returns:
        (answer: str, sources: List[Document], reasoning_steps: List[str])
    """
    logger.info(f"[AgenticRAG] Starting query: {question[:80]}...")

    # Build the graph
    graph = build_agent_graph(llm, rag_system)

    # Initialize state
    initial_state: AgentState = {
        "question": question,
        "sub_questions": [],
        "current_sub_question": None,
        "target_files": target_files,
        "documents": [],
        "generation": "",
        "retry_count": 0,
        "query_type": "simple",
        "reasoning_steps": [],
        "is_relevant": False,
        "sub_answers": [],
    }

    # Run the graph
    try:
        final_state = graph.invoke(initial_state)
    except Exception as e:
        logger.error(f"[AgenticRAG] Graph execution error: {e}")
        return (
            f"Lỗi khi xử lý (Agentic RAG): {str(e)}",
            [],
            [f"[Error] Agent failed: {str(e)}"],
        )

    answer = final_state.get("generation", "Không có câu trả lời.")
    sources = final_state.get("documents", [])
    reasoning = final_state.get("reasoning_steps", [])

    # Add strategy info
    strategy_info = (
        f"\n\n*Agentic RAG | Type: {final_state.get('query_type', 'N/A')} | "
        f"Retries: {final_state.get('retry_count', 0)}/{MAX_RETRIES} | "
        f"Chunks: {len(sources)}*"
    )
    answer += strategy_info

    logger.info(
        f"[AgenticRAG] Complete. Type: {final_state.get('query_type')}, "
        f"Retries: {final_state.get('retry_count')}, Chunks: {len(sources)}"
    )

    return answer, sources, reasoning
