"""
Chat History Manager - Lưu trữ và quản lý lịch sử trò chuyện.

Sử dụng JSON files để lưu trữ persistent:
- ./chat_history/_index.json: danh sách conversations (id, title, updated_at)
- ./chat_history/{id}.json: nội dung từng conversation
"""

import json
import os
import uuid
import logging
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)

HISTORY_DIR = "./chat_history"
INDEX_FILE = os.path.join(HISTORY_DIR, "_index.json")


class ChatHistoryManager:
    """Quản lý lưu trữ lịch sử chat bằng JSON files."""

    def __init__(self, history_dir: str = HISTORY_DIR):
        self.history_dir = history_dir
        self.index_file = os.path.join(history_dir, "_index.json")
        os.makedirs(self.history_dir, exist_ok=True)

        if not os.path.exists(self.index_file):
            self._write_index([])

    # ------ Index helpers ------

    def _read_index(self) -> list:
        try:
            with open(self.index_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            return []

    def _write_index(self, index: list):
        with open(self.index_file, "w", encoding="utf-8") as f:
            json.dump(index, f, ensure_ascii=False, indent=2)

    def _conv_path(self, conv_id: str) -> str:
        return os.path.join(self.history_dir, f"{conv_id}.json")

    # ------ Public API ------

    def create_conversation(self) -> str:
        """Tạo cuộc trò chuyện mới, trả về conversation_id."""
        conv_id = uuid.uuid4().hex[:12]
        now = datetime.now().isoformat()

        conv_data = {
            "id": conv_id,
            "title": "New Chat",
            "created_at": now,
            "updated_at": now,
            "messages": [],
        }

        with open(self._conv_path(conv_id), "w", encoding="utf-8") as f:
            json.dump(conv_data, f, ensure_ascii=False, indent=2)

        index = self._read_index()
        index.insert(0, {
            "id": conv_id,
            "title": conv_data["title"],
            "updated_at": now,
        })
        self._write_index(index)

        logger.info(f"Created conversation: {conv_id}")
        return conv_id

    def save_conversation(self, conv_id: str, messages: list, title: Optional[str] = None):
        """Lưu messages và cập nhật timestamp. Có thể cập nhật title."""
        path = self._conv_path(conv_id)
        try:
            with open(path, "r", encoding="utf-8") as f:
                conv_data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            conv_data = {
                "id": conv_id,
                "title": title or "New Chat",
                "created_at": datetime.now().isoformat(),
            }

        now = datetime.now().isoformat()
        conv_data["messages"] = messages
        conv_data["updated_at"] = now
        if title:
            conv_data["title"] = title

        with open(path, "w", encoding="utf-8") as f:
            json.dump(conv_data, f, ensure_ascii=False, indent=2)

        # Update index
        index = self._read_index()
        for entry in index:
            if entry["id"] == conv_id:
                entry["updated_at"] = now
                if title:
                    entry["title"] = title
                break
        else:
            index.insert(0, {"id": conv_id, "title": conv_data.get("title", ""), "updated_at": now})

        # Sort by updated_at desc
        index.sort(key=lambda x: x.get("updated_at", ""), reverse=True)
        self._write_index(index)

    def load_conversation(self, conv_id: str) -> Optional[dict]:
        """Load nội dung cuộc trò chuyện."""
        path = self._conv_path(conv_id)
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            logger.warning(f"Could not load conversation: {conv_id}")
            return None

    def list_conversations(self) -> list:
        """Trả về danh sách conversations, sorted by updated_at desc."""
        return self._read_index()

    def delete_conversation(self, conv_id: str):
        """Xóa cuộc trò chuyện."""
        path = self._conv_path(conv_id)
        if os.path.exists(path):
            os.remove(path)

        index = self._read_index()
        index = [e for e in index if e["id"] != conv_id]
        self._write_index(index)
        logger.info(f"Deleted conversation: {conv_id}")

    def update_title(self, conv_id: str, title: str):
        """Cập nhật tiêu đề cuộc trò chuyện."""
        path = self._conv_path(conv_id)
        try:
            with open(path, "r", encoding="utf-8") as f:
                conv_data = json.load(f)
            conv_data["title"] = title
            with open(path, "w", encoding="utf-8") as f:
                json.dump(conv_data, f, ensure_ascii=False, indent=2)
        except (FileNotFoundError, json.JSONDecodeError):
            pass

        index = self._read_index()
        for entry in index:
            if entry["id"] == conv_id:
                entry["title"] = title
                break
        self._write_index(index)
        logger.info(f"Updated title for {conv_id}: {title}")

    def get_title(self, conv_id: str) -> str:
        """Lấy tiêu đề cuộc trò chuyện."""
        index = self._read_index()
        for entry in index:
            if entry["id"] == conv_id:
                return entry.get("title", "New Chat")
        return "New Chat"
