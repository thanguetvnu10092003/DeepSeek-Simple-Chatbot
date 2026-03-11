import time
import functools
import json
import logging

from langchain_core.language_models.llms import LLM
from groq import Groq
from typing import List, Optional
import os

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def retry_with_backoff(max_retries: int = 3, base_delay: float = 1.0):
    """Decorator for retry logic with exponential backoff"""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    delay = base_delay * (2 ** attempt)
                    logger.warning(f"Attempt {attempt + 1}/{max_retries} failed: {e}. Retrying in {delay}s...")
                    time.sleep(delay)
            logger.error(f"All {max_retries} attempts failed")
            raise last_exception
        return wrapper
    return decorator


class QueryClassificationCache:
    """Simple LRU cache for query classifications"""
    def __init__(self, max_size: int = 100):
        self.cache = {}
        self.max_size = max_size
        self.access_order = []
    
    def get(self, key: str) -> Optional[dict]:
        if key in self.cache:
            # Move to end (most recently used)
            self.access_order.remove(key)
            self.access_order.append(key)
            logger.debug(f"Cache hit for query: {key[:50]}...")
            return self.cache[key]
        return None
    
    def set(self, key: str, value: dict):
        if len(self.cache) >= self.max_size:
            # Remove oldest
            oldest = self.access_order.pop(0)
            del self.cache[oldest]
        self.cache[key] = value
        self.access_order.append(key)


class GroqLLM(LLM):
    client: Groq = None
    model_name: str = "llama-3.3-70b-versatile"
    _classification_cache: QueryClassificationCache = None

    def __init__(self):
        super().__init__()
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("Chưa có GROQ_API_KEY trong file .env")
        self.client = Groq(api_key=api_key)
        self._classification_cache = QueryClassificationCache()

    @property
    def _llm_type(self):
        return "groq"

    @retry_with_backoff(max_retries=3, base_delay=1.0)
    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=2048,
        )
        return response.choices[0].message.content

    def _parse_json_response(self, result: str) -> dict:
        """Robust JSON parsing from LLM response"""
        # Remove markdown code block if present
        if result.startswith("```"):
            parts = result.split("```")
            if len(parts) >= 2:
                result = parts[1]
                if result.startswith("json"):
                    result = result[4:]
        
        result = result.strip()
        
        # Try to find JSON object in the response
        start_idx = result.find("{")
        end_idx = result.rfind("}") + 1
        
        if start_idx != -1 and end_idx > start_idx:
            result = result[start_idx:end_idx]
        
        return json.loads(result)

    def classify_query(self, question: str) -> dict:
        """Classify query using LLM to determine strategy"""
        # Check cache first
        cached = self._classification_cache.get(question)
        if cached:
            return cached
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{
                    "role": "user",
                    "content": f"""<persona>
You are an elite NLP Query Classifier.
</persona>

<constraints>
1. FORMATTING: You MUST return ONLY a valid JSON object.
2. NO HALLUCINATION: Do not include markdown codeblocks or any additional explanations.
</constraints>

<task_specific>
Analyze the following user question and determine its type, complexity, and requirements.

JSON EXPLANATION:
- "type": 
  * "overview": summarize/list across multiple files
  * "specific": asking about specific content in 1 file/topic
  * "exercise": solving exercises, needs full problem statement
  * "question": general question
- "needs_all_files": boolean, does it need information from ALL files?
- "complexity": "low", "medium", or "high"
- "needs_detail": boolean, needs a detailed or summary answer?

Example valid JSON output:
{{
    "type": "specific",
    "needs_all_files": false,
    "complexity": "medium",
    "needs_detail": true
}}
</task_specific>

<user_query>
{question}
</user_query>"""             }],
                temperature=0.1,
                max_tokens=150,
            )

            result = response.choices[0].message.content.strip()
            parsed = self._parse_json_response(result)
            
            # Cache the result
            self._classification_cache.set(question, parsed)
            
            return parsed
        except json.JSONDecodeError as e:
            logger.warning(f"JSON parse error: {e}")
            return self._default_classification()
        except Exception as e:
            logger.error(f"Classify error: {e}")
            return self._default_classification()
    
    def _default_classification(self) -> dict:
        """Fallback classification"""
        return {
            "type": "question",
            "needs_all_files": False,
            "complexity": "medium",
            "needs_detail": True
        }

    # ===========================
    # Agentic RAG Methods
    # ===========================

    def route_query(self, question: str) -> dict:
        """
        Route query to determine if it's simple or complex.
        Complex queries need decomposition into sub-questions.
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{
                    "role": "user",
                    "content": f"""<persona>
You are a smart Query Router.
</persona>

<constraints>
1. FORMATTING: You MUST return ONLY a valid JSON object.
2. NO HALLUCINATION: Do not explain your answer outside the JSON payload.
</constraints>

<task_specific>
Determine if the given query is SIMPLE or COMPLEX.

=== DEFINITIONS ===
A question is COMPLEX if it:
- Requires comparing information from multiple sources.
- Has multiple sub-questions or parts.
- Needs multi-step reasoning.
- Asks for analysis/synthesis across topics.

Expected JSON Format:
{{
    "type": "simple" | "complex",
    "reasoning": "<brief explanation of why it is simple or complex>"
}}
</task_specific>

<user_query>
{question}
</user_query>"""
                }],
                temperature=0.1,
                max_tokens=100,
            )
            result = response.choices[0].message.content.strip()
            return self._parse_json_response(result)
        except Exception as e:
            logger.warning(f"Route query error: {e}")
            return {"type": "simple", "reasoning": "fallback"}

    def decompose_query(self, question: str) -> list:
        """
        Decompose a complex question into simpler sub-questions.
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{
                    "role": "user",
                    "content": f"""<persona>
You are an expert Query Decomposer.
</persona>

<constraints>
1. FORMATTING: You MUST return ONLY a valid JSON object without markdown formatting.
2. NO PREAMBLES: Directly return the JSON.
</constraints>

<task_specific>
Your task is to break down a complex question into 2 to 4 simpler, independent sub-questions that can be answered sequentially.

Expected JSON Format:
{{
    "sub_questions": ["<sub_question_1>", "<sub_question_2>"]
}}
</task_specific>

<original_question>
{question}
</original_question>"""
                }],
                temperature=0.2,
                max_tokens=300,
            )
            result = response.choices[0].message.content.strip()
            parsed = self._parse_json_response(result)
            sub_qs = parsed.get("sub_questions", [question])
            # Ensure we always have at least the original question
            return sub_qs if sub_qs else [question]
        except Exception as e:
            logger.warning(f"Decompose query error: {e}")
            return [question]

    def grade_documents(self, question: str, documents: list) -> list:
        """
        Grade documents for relevance to the question.
        Returns a list of booleans (True = relevant).
        Grades in batches to save API calls.
        """
        if not documents:
            return []

        # Batch grading: grade up to 10 docs at a time
        batch_size = 10
        all_grades = []

        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            doc_summaries = []
            for j, doc in enumerate(batch):
                content_preview = doc.page_content[:300]
                doc_summaries.append(f"Doc {j+1}: {content_preview}")

            docs_text = "\n---\n".join(doc_summaries)

            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{
                        "role": "user",
                        "content": f"""<persona>
You are a strict Document Relevance Grader.
</persona>

<constraints>
1. FORMATTING: You MUST return ONLY a valid JSON object.
2. NO PREAMBLES: Directly return the JSON.
</constraints>

<task_specific>
Determine if each document contains information relevant to answering the question.
Provide an array of booleans mapping 1-to-1 to the provided documents (true = relevant, false = irrelevant).

Expected JSON Format:
{{
    "grades": [true, false, true]
}}
</task_specific>

<documents_to_grade>
{docs_text}
</documents_to_grade>

<user_query>
{question}
</user_query>"""
                    }],
                    temperature=0.1,
                    max_tokens=200,
                )
                result = response.choices[0].message.content.strip()
                parsed = self._parse_json_response(result)
                grades = parsed.get("grades", [True] * len(batch))

                # Ensure correct length
                while len(grades) < len(batch):
                    grades.append(True)
                all_grades.extend(grades[:len(batch)])
            except Exception as e:
                logger.warning(f"Grade documents error: {e}")
                all_grades.extend([True] * len(batch))

        return all_grades

    def rewrite_query(self, original_question: str, context: str = "") -> str:
        """
        Rewrite a query to improve retrieval results.
        """
        try:
            context_hint = ""
            if context:
                context_hint = f"\n\nPartial context found (may not be fully relevant):\n{context[:500]}"

            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{
                    "role": "user",
                    "content": f"""<persona>
You are an advanced Search Query Reformulator.
</persona>

<constraints>
1. FORMATTING: Return ONLY the newly rewritten query string.
2. NO PREAMBLES: Do not use quotes, JSON, or any conversational text.
</constraints>

<task_specific>
The following user query failed to retrieve good results from our vector database.
Your task is to rewrite it to improve semantic and keyword search performance. 
Try using different synonyms, isolating core technical terms, or being more specific.
{context_hint}
</task_specific>

<failed_query>
{original_question}
</failed_query>"""
                }],
                temperature=0.3,
                max_tokens=200,
            )
            rewritten = response.choices[0].message.content.strip()
            # Clean up - remove quotes if present
            rewritten = rewritten.strip('"').strip("'")
            return rewritten if rewritten else original_question
        except Exception as e:
            logger.warning(f"Rewrite query error: {e}")
            return original_question

    def check_hallucination(self, answer: str, documents: list) -> bool:
        """
        Check if the answer is grounded in the provided documents.
        Returns True if grounded (no hallucination), False otherwise.
        """
        if not documents or not answer:
            return True  # Can't check without docs

        # Build context from top documents
        doc_context = "\n---\n".join([
            doc.page_content[:400] for doc in documents[:5]
        ])

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{
                    "role": "user",
                    "content": f"""<persona>
You are an aggressive and strict Fact Checker (Hallucination Detector).
</persona>

<constraints>
1. FACTUALITY: The answer MUST NOT contain facts, claims, or data points that cannot be explicitly traced back to the documents.
2. FORMATTING: You MUST return ONLY a valid JSON object.
3. NO PREAMBLES: Directly return the JSON.
</constraints>

<task_specific>
Your ONLY job is to verify if the provided answer is **100% grounded** (supported) by the provided Source Documents.

Expected JSON Format:
{{
    "is_grounded": true | false,
    "reasoning": "<Point out specific unsupported claims if false, otherwise brief approval>"
}}
</task_specific>

<source_documents>
{doc_context}
</source_documents>

<answer_to_grade>
{answer[:1000]}
</answer_to_grade>"""
                }],
                temperature=0.1,
                max_tokens=150,
            )
            result = response.choices[0].message.content.strip()
            parsed = self._parse_json_response(result)
            return parsed.get("is_grounded", True)
        except Exception as e:
            logger.warning(f"Hallucination check error: {e}")
            return True  # Default to trusting the answer

    def generate_chat_title(self, first_message: str) -> str:
        """
        Tạo tiêu đề ngắn gọn cho cuộc trò chuyện dựa trên tin nhắn đầu tiên.
        Trả về tiêu đề tối đa 10 từ.
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{
                    "role": "user",
                    "content": f"""<persona>
You are a Chat Summarizer.
</persona>

<constraints>
1. FORMATTING: Return ONLY the raw title string.
2. NO PREAMBLES: Do not use quotation marks, markdown, or any conversational text.
</constraints>

<task_specific>
Generate a very short, concise title (maximum 8 words) for a conversation that begins with the message below.
</task_specific>

<message>
{first_message[:300]}
</message>"""
                }],
                temperature=0.3,
                max_tokens=30,
            )
            title = response.choices[0].message.content.strip().strip('"').strip("'")
            # Giới hạn độ dài
            words = title.split()
            if len(words) > 10:
                title = " ".join(words[:10])
            return title if title else "Cuộc trò chuyện mới"
        except Exception as e:
            logger.warning(f"Generate chat title error: {e}")
            # Fallback: dùng phần đầu tin nhắn
            truncated = first_message[:50].strip()
            if len(first_message) > 50:
                truncated += "..."
            return truncated