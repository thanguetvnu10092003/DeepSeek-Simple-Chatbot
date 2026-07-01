import io
import os
import base64
import logging
from pathlib import Path
from typing import List, Optional, Callable

import fitz  # pymupdf
from groq import Groq
from PIL import Image
from sentence_transformers import SentenceTransformer
import chromadb

logger = logging.getLogger(__name__)

CLIP_MODEL = 'jinaai/jina-clip-v2'
VLM_MODEL = 'meta-llama/llama-4-scout-17b-16e-instruct'
TOP_K = 3


class VisionRAG:
    def __init__(
        self,
        persist_directory: str = "./vision_chroma_db",
        pages_directory: str = "./vision_store/pages",
    ):
        logger.info("Initializing VisionRAG...")
        self.pages_dir = pages_directory
        os.makedirs(self.pages_dir, exist_ok=True)

        logger.info(f"Loading CLIP model: {CLIP_MODEL}")
        self.clip = SentenceTransformer(CLIP_MODEL, trust_remote_code=True)

        self._chroma = chromadb.PersistentClient(path=persist_directory)
        self._collection = self._chroma.get_or_create_collection(
            "vision_chunks",
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(f"VisionRAG ready. Collection size: {self._collection.count()}")

    # ── Index ─────────────────────────────────────────────────────────────────

    def add_pdf(self, pdf_path: str, progress: Callable = None) -> tuple:
        """Render each PDF page to PNG, encode with CLIP, store in ChromaDB.
        Returns (pages_indexed, pages_skipped)."""
        if progress is None:
            progress = lambda *a, **kw: None

        filename = Path(pdf_path).name
        page_dir = os.path.join(self.pages_dir, filename)
        os.makedirs(page_dir, exist_ok=True)

        indexed, skipped = 0, 0

        with fitz.open(pdf_path) as doc:
            total = len(doc)
            progress(0, desc="Rendering pages...")

            for i, page in enumerate(doc):
                progress(i / total, desc=f"Indexing page {i + 1}/{total}...")
                try:
                    pix = page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5))
                    img_path = os.path.join(page_dir, f"page_{i + 1}.png")
                    pix.save(img_path)

                    img = Image.open(img_path).convert('RGB')
                    if img.width > 1024:
                        ratio = 1024 / img.width
                        img = img.resize((1024, int(img.height * ratio)), Image.LANCZOS)
                        img.save(img_path)

                    embedding = self.clip.encode(img).tolist()

                    self._collection.upsert(
                        ids=[f"{filename}_page_{i + 1}"],
                        embeddings=[embedding],
                        metadatas=[{
                            "filename": filename,
                            "page": i + 1,
                            "image_path": img_path,
                        }],
                        documents=[f"{filename} page {i + 1}"],
                    )
                    indexed += 1
                except Exception as e:
                    logger.warning(f"Skipped page {i + 1} of {filename}: {e}")
                    skipped += 1

        progress(1.0, desc="Complete!")
        logger.info(f"VisionRAG indexed {indexed} pages, skipped {skipped} from {filename}")
        return indexed, skipped

    # ── File management ───────────────────────────────────────────────────────

    def get_all_files(self) -> List[str]:
        if self._collection.count() == 0:
            return []
        results = self._collection.get(include=["metadatas"])
        files = set()
        for meta in results["metadatas"]:
            if meta and "filename" in meta:
                files.add(meta["filename"])
        return list(files)

    def delete_file(self, filename: str) -> None:
        results = self._collection.get(
            where={"filename": filename},
            include=["metadatas"],
        )
        if results["ids"]:
            self._collection.delete(ids=results["ids"])
            logger.info(f"Deleted {len(results['ids'])} ChromaDB entries for {filename}")

        page_dir = os.path.join(self.pages_dir, filename)
        if os.path.exists(page_dir):
            for f in os.listdir(page_dir):
                os.remove(os.path.join(page_dir, f))
            os.rmdir(page_dir)
            logger.info(f"Deleted page images for {filename}")

    # ── Query ─────────────────────────────────────────────────────────────────

    def query(
        self,
        question: str,
        target_files: Optional[List[str]] = None,
        history: Optional[list] = None,
    ) -> tuple:
        """Encode question with CLIP, retrieve top-k pages, call Replicate VLM.
        Returns (answer_str, list_of_metadata_dicts)."""
        if self._collection.count() == 0:
            return "No documents in Vision Mode yet.", []

        text_embedding = self.clip.encode(question).tolist()

        where_filter = None
        if target_files and len(target_files) == 1:
            where_filter = {"filename": target_files[0]}
        elif target_files and len(target_files) > 1:
            where_filter = {"filename": {"$in": target_files}}

        if where_filter:
            filtered_count = len(self._collection.get(where=where_filter, include=[])["ids"])
            n = min(TOP_K, filtered_count)
        else:
            n = min(TOP_K, self._collection.count())

        if n == 0:
            return "No relevant pages found.", []

        query_kwargs = dict(
            query_embeddings=[text_embedding],
            n_results=n,
            include=["metadatas", "distances"],
        )
        if where_filter:
            query_kwargs["where"] = where_filter

        results = self._collection.query(**query_kwargs)

        metadatas = results["metadatas"][0] if results["metadatas"] else []
        if not metadatas:
            return "No relevant pages found.", []

        image_paths = [
            m["image_path"] for m in metadatas
            if os.path.exists(m["image_path"])
        ]
        if not image_paths:
            return "Page images not found on disk.", []

        combined_bytes = self._combine_images(image_paths)
        answer = self._call_vlm(combined_bytes, question, metadatas)

        strategy = f"\n\n*Vision RAG | CLIP retrieval | Pages: {len(metadatas)}*"
        return answer + strategy, metadatas

    # ── Static helpers ────────────────────────────────────────────────────────

    @staticmethod
    def _combine_images(image_paths: List[str], max_width: int = 1024) -> bytes:
        """Vertically stack page images, return PNG bytes."""
        images = []
        for p in image_paths:
            img = Image.open(p).convert('RGB')
            images.append(img)

        target_w = min(max_width, max(img.width for img in images))
        resized = []
        for img in images:
            if img.width != target_w:
                ratio = target_w / img.width
                img = img.resize((target_w, int(img.height * ratio)), Image.LANCZOS)
            resized.append(img)

        total_h = sum(img.height for img in resized)
        combined = Image.new('RGB', (target_w, total_h), (255, 255, 255))
        y = 0
        for img in resized:
            combined.paste(img, (0, y))
            y += img.height

        buf = io.BytesIO()
        combined.save(buf, format='PNG')
        return buf.getvalue()

    @staticmethod
    def _call_vlm(image_bytes: bytes, question: str, metadatas: list) -> str:
        """Send combined page image to Replicate VLM, return text answer."""
        pages = ", ".join(str(m['page']) for m in metadatas)
        prompt = (
            "You are an expert academic document analyst. "
            f"The image contains pages {pages} from an academic thesis. "
            "Analyze and evaluate all visual content you see — maps, charts, diagrams, graphs, satellite images, tables. "
            "For each visual element: describe what it shows, evaluate its quality and effectiveness, and explain what conclusions can be drawn from it. "
            "Be detailed and analytical, not just descriptive. "
            "Respond in the same language as the question.\n\n"
            f"Question: {question}"
        )
        img_b64 = base64.b64encode(image_bytes).decode('utf-8')
        client = Groq()
        completion = client.chat.completions.create(
            model=VLM_MODEL,
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{img_b64}"},
                    },
                    {"type": "text", "text": prompt},
                ],
            }],
            temperature=0.1,
            max_tokens=1024,
        )
        return completion.choices[0].message.content
