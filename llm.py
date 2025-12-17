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
        """Phân loại câu hỏi bằng LLM để quyết định strategy"""
        # Check cache first
        cached = self._classification_cache.get(question)
        if cached:
            return cached
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{
                    "role": "user",
                    "content": f"""Phân tích câu hỏi sau và trả lời ĐÚNG định dạng JSON:

Câu hỏi: "{question}"

Trả về JSON với các trường:
{{
    "type": "overview" | "specific" | "exercise" | "question",
    "needs_all_files": true/false,
    "complexity": "low" | "medium" | "high",
    "needs_detail": true/false
}}

Giải thích:
- type: 
  * "overview": tóm tắt/liệt kê nhiều file
  * "specific": hỏi về nội dung cụ thể 1 file/topic
  * "exercise": giải bài tập, cần đề bài đầy đủ
  * "question": câu hỏi thông thường
- needs_all_files: có cần thông tin từ TẤT CẢ file không?
- complexity: độ phức tạp câu hỏi
- needs_detail: cần câu trả lời chi tiết hay tóm tắt?
"""             }],
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