from langchain_core.language_models.llms import LLM
from groq import Groq
from typing import List, Optional
import os

class GroqLLM(LLM):
    client: Groq = None
    model_name: str = "llama-3.3-70b-versatile"

    def __init__(self):
        super().__init__()
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("Chưa có GROQ_API_KEY trong file .env")
        self.client = Groq(api_key=api_key)

    @property
    def _llm_type(self):
        return "groq"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=2048,
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Lỗi Groq API: {str(e)}"

    def classify_query(self, question: str) -> dict:
        """Phân loại câu hỏi bằng LLM để quyết định strategy"""
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
            # Loại bỏ markdown code block nếu có
            if result.startswith("```"):
                result = result.split("```")[1]
                if result.startswith("json"):
                    result = result[4:]
            result = result.strip()

            import json
            return json.loads(result)
        except Exception as e:
            print(f"Classify error: {e}")
            # Fallback về default
            return {
                "type": "question",
                "needs_all_files": False,
                "complexity": "medium",
                "needs_detail": True
            }