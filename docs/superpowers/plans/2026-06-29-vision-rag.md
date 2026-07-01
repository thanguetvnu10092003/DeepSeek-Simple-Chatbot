# Vision RAG Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a Vision RAG mode alongside existing Text RAG — PDFs indexed as CLIP image embeddings, answered by a Replicate VLM (Llama-3.2-11b-vision-instruct) that sees the actual page images.

**Architecture:** `VisionRAG` is a standalone class in `vision_rag.py` using raw `chromadb` (not LangChain) with CLIP embeddings from `sentence-transformers`. `server.py` routes to it when `vision_mode=True`. The frontend adds a "Vision Mode" toggle that switches file list, upload, and query routing independently from existing Text RAG.

**Tech Stack:** `sentence-transformers` (CLIP), `chromadb`, `pymupdf`, `Pillow`, `replicate`, FastAPI, vanilla JS

## Global Constraints

- No new packages — all dependencies already exist in `requirements.txt`
- CLIP model: `clip-ViT-B-32` via `sentence_transformers.SentenceTransformer`
- VLM model on Replicate: `meta/llama-3.2-11b-vision-instruct`
- Top-k pages retrieved per query: `TOP_K = 3` (constant in `vision_rag.py`)
- Max image width: 1024px — resize before storing and before sending to VLM
- ChromaDB persist path: `./vision_chroma_db`
- Page images stored at: `./vision_store/pages/<filename>/page_N.png`
- **Never modify:** `rag.py`, `agentic_rag.py`, `llm.py`, `pdf_ocr_loader.py`, `chat_history.py`

---

## File Map

| File | Action | Responsibility |
|---|---|---|
| `vision_rag.py` | Create | VisionRAG class: index, retrieve, query |
| `tests/__init__.py` | Create | Make tests/ a package |
| `tests/test_vision_rag.py` | Create | Unit tests for VisionRAG |
| `server.py` | Modify | Instantiate VisionRAG, route by `vision_mode` param |
| `static/index.html` | Modify | Add Vision Mode toggle HTML |
| `static/app.js` | Modify | State + events + API calls with `vision_mode` |
| `static/style.css` | Modify | Mode badge styling |

---

## Task 1: Create `vision_rag.py` — Core VisionRAG class

**Files:**
- Create: `vision_rag.py`
- Create: `tests/__init__.py`
- Create: `tests/test_vision_rag.py`

**Interfaces:**
- Produces:
  - `VisionRAG(persist_directory, pages_directory)` constructor
  - `add_pdf(pdf_path, progress) -> tuple[int, int]` — returns `(pages_indexed, pages_skipped)`
  - `get_all_files() -> List[str]`
  - `delete_file(filename: str) -> None`
  - `query(question, target_files=None, history=None) -> tuple[str, list]` — returns `(answer_str, list_of_metadata_dicts)`
  - `VisionRAG._combine_images(image_paths, max_width=1024) -> bytes` — static method
  - `VisionRAG._call_vlm(image_bytes, question, metadatas) -> str` — static method

- [ ] **Step 1: Create the tests package**

```
mkdir tests
```
Create `tests/__init__.py` (empty file).

- [ ] **Step 2: Write failing tests**

Create `tests/test_vision_rag.py`:

```python
import io
import os
import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from PIL import Image


# ── Fixture ───────────────────────────────────────────────────────────────────

@pytest.fixture
def tmp_rag(tmp_path):
    """VisionRAG with CLIP mocked to avoid model download."""
    with patch('vision_rag.SentenceTransformer') as mock_st:
        mock_model = MagicMock()
        mock_model.encode.return_value = np.zeros(512, dtype=np.float32)
        mock_st.return_value = mock_model

        from vision_rag import VisionRAG
        rag = VisionRAG(
            persist_directory=str(tmp_path / "chroma"),
            pages_directory=str(tmp_path / "pages"),
        )
        yield rag


def _save_png(path, w=100, h=100, color=(200, 200, 200)):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.new('RGB', (w, h), color=color).save(path)


# ── get_all_files ─────────────────────────────────────────────────────────────

def test_get_all_files_empty(tmp_rag):
    assert tmp_rag.get_all_files() == []


def test_get_all_files_returns_indexed_filename(tmp_rag):
    tmp_rag._collection.upsert(
        ids=["doc.pdf_page_1"],
        embeddings=[np.zeros(512).tolist()],
        metadatas=[{"filename": "doc.pdf", "page": 1, "image_path": "/fake/p.png"}],
        documents=["doc.pdf page 1"],
    )
    assert tmp_rag.get_all_files() == ["doc.pdf"]


# ── _combine_images ───────────────────────────────────────────────────────────

def test_combine_single_image_returns_png(tmp_path):
    from vision_rag import VisionRAG
    p = str(tmp_path / "p1.png")
    Image.new('RGB', (200, 300)).save(p)
    result = VisionRAG._combine_images([p])
    out = Image.open(io.BytesIO(result))
    assert out.size == (200, 300)


def test_combine_two_images_stacks_vertically(tmp_path):
    from vision_rag import VisionRAG
    p1, p2 = str(tmp_path / "p1.png"), str(tmp_path / "p2.png")
    Image.new('RGB', (200, 100)).save(p1)
    Image.new('RGB', (200, 150)).save(p2)
    result = VisionRAG._combine_images([p1, p2])
    out = Image.open(io.BytesIO(result))
    assert out.size == (200, 250)


def test_combine_resizes_wide_image(tmp_path):
    from vision_rag import VisionRAG
    p = str(tmp_path / "wide.png")
    Image.new('RGB', (2000, 400)).save(p)
    result = VisionRAG._combine_images([p], max_width=1024)
    out = Image.open(io.BytesIO(result))
    assert out.width == 1024


# ── delete_file ───────────────────────────────────────────────────────────────

def test_delete_file_removes_chroma_entries(tmp_rag, tmp_path):
    img_path = str(tmp_path / "pages" / "doc.pdf" / "page_1.png")
    _save_png(img_path)
    tmp_rag._collection.upsert(
        ids=["doc.pdf_page_1"],
        embeddings=[np.zeros(512).tolist()],
        metadatas=[{"filename": "doc.pdf", "page": 1, "image_path": img_path}],
        documents=["doc.pdf page 1"],
    )
    assert tmp_rag._collection.count() == 1
    tmp_rag.delete_file("doc.pdf")
    assert tmp_rag._collection.count() == 0


def test_delete_file_removes_png_files(tmp_rag, tmp_path):
    img_path = str(tmp_path / "pages" / "doc.pdf" / "page_1.png")
    _save_png(img_path)
    tmp_rag._collection.upsert(
        ids=["doc.pdf_page_1"],
        embeddings=[np.zeros(512).tolist()],
        metadatas=[{"filename": "doc.pdf", "page": 1, "image_path": img_path}],
        documents=["doc.pdf page 1"],
    )
    tmp_rag.delete_file("doc.pdf")
    assert not os.path.exists(img_path)


# ── query ─────────────────────────────────────────────────────────────────────

def test_query_returns_message_when_empty(tmp_rag):
    answer, sources = tmp_rag.query("what is this?")
    assert "No documents" in answer
    assert sources == []


def test_query_calls_vlm_and_returns_answer(tmp_rag, tmp_path):
    img_path = str(tmp_path / "pages" / "test.pdf" / "page_1.png")
    _save_png(img_path)
    tmp_rag._collection.upsert(
        ids=["test.pdf_page_1"],
        embeddings=[np.zeros(512).tolist()],
        metadatas=[{"filename": "test.pdf", "page": 1, "image_path": img_path}],
        documents=["test.pdf page 1"],
    )
    with patch('vision_rag.replicate.run', return_value=["Answer from VLM"]):
        answer, sources = tmp_rag.query("describe the document")
    assert "Answer from VLM" in answer
    assert len(sources) == 1
    assert sources[0]["filename"] == "test.pdf"


def test_query_filters_by_target_file(tmp_rag, tmp_path):
    for fname, page_num in [("a.pdf", 1), ("b.pdf", 1)]:
        img_path = str(tmp_path / "pages" / fname / f"page_{page_num}.png")
        _save_png(img_path)
        tmp_rag._collection.upsert(
            ids=[f"{fname}_page_{page_num}"],
            embeddings=[np.zeros(512).tolist()],
            metadatas=[{"filename": fname, "page": page_num, "image_path": img_path}],
            documents=[f"{fname} page {page_num}"],
        )
    with patch('vision_rag.replicate.run', return_value=["ok"]):
        answer, sources = tmp_rag.query("test", target_files=["a.pdf"])
    assert all(s["filename"] == "a.pdf" for s in sources)
```

- [ ] **Step 3: Run tests — confirm they all fail**

```
pytest tests/test_vision_rag.py -v
```
Expected: `ModuleNotFoundError: No module named 'vision_rag'`

- [ ] **Step 4: Create `vision_rag.py`**

```python
import io
import os
import base64
import logging
from pathlib import Path
from typing import List, Optional, Callable

import fitz  # pymupdf
import replicate
from PIL import Image
from sentence_transformers import SentenceTransformer
import chromadb

logger = logging.getLogger(__name__)

CLIP_MODEL = 'clip-ViT-B-32'
VLM_MODEL = 'meta/llama-3.2-11b-vision-instruct'
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
        self.clip = SentenceTransformer(CLIP_MODEL)

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

        doc = fitz.open(pdf_path)
        indexed, skipped = 0, 0
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

        doc.close()
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

        query_kwargs = dict(
            query_embeddings=[text_embedding],
            n_results=min(TOP_K, self._collection.count()),
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
            if img.width > max_width:
                ratio = max_width / img.width
                img = img.resize((max_width, int(img.height * ratio)), Image.LANCZOS)
            images.append(img)

        target_w = images[0].width
        total_h = sum(img.height for img in images)
        combined = Image.new('RGB', (target_w, total_h), (255, 255, 255))
        y = 0
        for img in images:
            combined.paste(img, (0, y))
            y += img.height

        buf = io.BytesIO()
        combined.save(buf, format='PNG')
        return buf.getvalue()

    @staticmethod
    def _call_vlm(image_bytes: bytes, question: str, metadatas: list) -> str:
        """Send combined page image to Replicate VLM, return text answer."""
        page_refs = ", ".join(
            f"page {m['page']} of {m['filename']}" for m in metadatas
        )
        prompt = (
            "You are a document analysis assistant. "
            f"The image shows the following document pages: {page_refs}. "
            "Answer based ONLY on what is visible in the image. "
            "Cite the page number for each claim.\n\n"
            f"Question: {question}"
        )
        img_b64 = base64.b64encode(image_bytes).decode('utf-8')
        output = replicate.run(
            VLM_MODEL,
            input={
                "prompt": prompt,
                "image": f"data:image/png;base64,{img_b64}",
            },
        )
        return "".join(output)
```

- [ ] **Step 5: Run tests — confirm they all pass**

```
pytest tests/test_vision_rag.py -v
```
Expected: all 10 tests PASS

- [ ] **Step 6: Commit**

```bash
git add vision_rag.py tests/__init__.py tests/test_vision_rag.py
git commit -m "feat(vision-rag): add VisionRAG core class with CLIP + Replicate VLM"
```

---

## Task 2: Modify `server.py` — Vision mode routing

**Files:**
- Modify: `server.py`

**Interfaces:**
- Consumes: `VisionRAG` from `vision_rag` (Task 1)
- `GET /api/files?vision_mode=false` — returns files from `vision_system` or `rag_system`
- `DELETE /api/files/{filename}?vision_mode=false`
- `POST /api/upload` form field `vision_mode: bool`
- `POST /api/chat` body field `vision_mode: bool`

- [ ] **Step 1: Add import and instantiation**

In `server.py`, find:
```python
from rag import LangChainPDFRAG
from chat_history import ChatHistoryManager
```

Replace with:
```python
from rag import LangChainPDFRAG
from vision_rag import VisionRAG
from chat_history import ChatHistoryManager
```

Find:
```python
# Initialize systems
rag_system = LangChainPDFRAG()
history_manager = ChatHistoryManager()
os.makedirs(UPLOAD_DIR, exist_ok=True)
```

Replace with:
```python
# Initialize systems
rag_system = LangChainPDFRAG()
vision_system = VisionRAG()
history_manager = ChatHistoryManager()
os.makedirs(UPLOAD_DIR, exist_ok=True)
```

- [ ] **Step 2: Update `ChatRequest` model**

Find:
```python
class ChatRequest(BaseModel):
    message: str
    conversation_id: Optional[str] = None
    selected_files: Optional[List[str]] = None
    use_agentic: bool = True
```

Replace with:
```python
class ChatRequest(BaseModel):
    message: str
    conversation_id: Optional[str] = None
    selected_files: Optional[List[str]] = None
    use_agentic: bool = True
    vision_mode: bool = False
```

- [ ] **Step 3: Update `GET /api/files`**

Find:
```python
@app.get("/api/files")
async def list_files():
    """Danh sách files đã upload vào vectorstore và thông tin tồn tại vật lý"""
    files = rag_system.get_all_files()
    file_info = []
    for f in files:
        has_file = os.path.exists(os.path.join(UPLOAD_DIR, f))
        file_info.append({"name": f, "hasPreview": has_file})
    return {"files": file_info}
```

Replace with:
```python
@app.get("/api/files")
async def list_files(vision_mode: bool = False):
    """Danh sách files đã upload vào vectorstore và thông tin tồn tại vật lý"""
    if vision_mode:
        files = vision_system.get_all_files()
        file_info = [{"name": f, "hasPreview": False} for f in files]
    else:
        files = rag_system.get_all_files()
        file_info = []
        for f in files:
            has_file = os.path.exists(os.path.join(UPLOAD_DIR, f))
            file_info.append({"name": f, "hasPreview": has_file})
    return {"files": file_info}
```

- [ ] **Step 4: Update `DELETE /api/files/{filename}`**

Find:
```python
@app.delete("/api/files/{filename}")
async def delete_file(filename: str):
    """Xóa file khỏi vectorstore và thư mục uploads"""
    # 1. Xóa khỏi ChromaDB & BM25
    rag_system.delete_file(filename)
    
    # 2. Xóa file vật lý
    file_path = os.path.join(UPLOAD_DIR, filename)
    if os.path.exists(file_path):
        try:
            os.remove(file_path)
            logger.info(f"Deleted physical file: {file_path}")
        except Exception as e:
            logger.error(f"Cannot delete physical file {filename}: {e}")
            
    return {"status": "ok", "message": f"Deleted {filename}"}
```

Replace with:
```python
@app.delete("/api/files/{filename}")
async def delete_file(filename: str, vision_mode: bool = False):
    """Xóa file khỏi vectorstore và thư mục uploads"""
    if vision_mode:
        vision_system.delete_file(filename)
    else:
        rag_system.delete_file(filename)
        file_path = os.path.join(UPLOAD_DIR, filename)
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
                logger.info(f"Deleted physical file: {file_path}")
            except Exception as e:
                logger.error(f"Cannot delete physical file {filename}: {e}")

    return {"status": "ok", "message": f"Deleted {filename}"}
```

- [ ] **Step 5: Update `POST /api/upload`**

Find:
```python
@app.post("/api/upload")
async def upload_files(
    files: List[UploadFile] = File(...),
    enable_ocr: bool = Form(False)
):
    """Upload và xử lý files"""
    results = []
    existing_db_files = rag_system.get_all_files()
```

Replace with:
```python
@app.post("/api/upload")
async def upload_files(
    files: List[UploadFile] = File(...),
    enable_ocr: bool = Form(False),
    vision_mode: bool = Form(False),
):
    """Upload và xử lý files"""
    results = []
    existing_db_files = vision_system.get_all_files() if vision_mode else rag_system.get_all_files()
```

Then find (inside the upload loop, after saving the file and checking for pdf):
```python
            if file_ext == '.pdf':
                num_chunks, ocr_pages, skipped_pages = rag_system.add_pdf(temp_path, enable_ocr)
                elapsed = time.time() - start_time

                if num_chunks == 0:
                    results.append({
                        "name": file_name, "status": "warning",
                        "message": f"OCR needed ({skipped_pages} scanned pages)"
                    })
                else:
                    results.append({
                        "name": file_name, "status": "ok",
                        "chunks": num_chunks, "ocr_pages": ocr_pages,
                        "time": f"{elapsed:.1f}s"
                    })
```

Replace with:
```python
            if file_ext == '.pdf':
                if vision_mode:
                    pages_indexed, pages_skipped = vision_system.add_pdf(temp_path)
                    elapsed = time.time() - start_time
                    results.append({
                        "name": file_name, "status": "ok",
                        "chunks": pages_indexed, "ocr_pages": 0,
                        "time": f"{elapsed:.1f}s"
                    })
                else:
                    num_chunks, ocr_pages, skipped_pages = rag_system.add_pdf(temp_path, enable_ocr)
                    elapsed = time.time() - start_time

                    if num_chunks == 0:
                        results.append({
                            "name": file_name, "status": "warning",
                            "message": f"OCR needed ({skipped_pages} scanned pages)"
                        })
                    else:
                        results.append({
                            "name": file_name, "status": "ok",
                            "chunks": num_chunks, "ocr_pages": ocr_pages,
                            "time": f"{elapsed:.1f}s"
                        })
```

Also update the final file list at the bottom of `upload_files`:

Find:
```python
    all_files_raw = rag_system.get_all_files()
    all_files = []
    for f in all_files_raw:
        has_file = os.path.exists(os.path.join(UPLOAD_DIR, f))
        all_files.append({"name": f, "hasPreview": has_file})
    
    return {"results": results, "all_files": all_files}
```

Replace with:
```python
    if vision_mode:
        all_files = [{"name": f, "hasPreview": False} for f in vision_system.get_all_files()]
    else:
        all_files = []
        for f in rag_system.get_all_files():
            has_file = os.path.exists(os.path.join(UPLOAD_DIR, f))
            all_files.append({"name": f, "hasPreview": has_file})

    return {"results": results, "all_files": all_files}
```

- [ ] **Step 6: Update `POST /api/chat`**

Find (inside the `generate()` closure):
```python
            if request.use_agentic:
                answer, sources, reasoning_steps = rag_system.query_agentic(
                    message, target_files=target_files, history=messages[:-1]
                )
            else:
                answer, sources = rag_system.query(
                    message, target_files=target_files, history=messages[:-1]
                )

            elapsed = time.time() - start_time

            # Build response with sources
            response = answer + "\n\n"

            if sources:
                response += "---\n**References:**\n"
                file_sources = {}
                for doc in sources[:15]:
                    meta = doc.metadata
                    fname = meta.get('filename', 'Unknown')
                    page = meta.get('page', 'N/A')
```

Replace with:
```python
            if request.vision_mode:
                answer, sources = vision_system.query(
                    message, target_files=target_files, history=messages[:-1]
                )
                reasoning_steps = []
            elif request.use_agentic:
                answer, sources, reasoning_steps = rag_system.query_agentic(
                    message, target_files=target_files, history=messages[:-1]
                )
            else:
                answer, sources = rag_system.query(
                    message, target_files=target_files, history=messages[:-1]
                )
                reasoning_steps = []

            elapsed = time.time() - start_time

            # Build response with sources
            response = answer + "\n\n"

            if sources:
                response += "---\n**References:**\n"
                file_sources = {}
                for doc in sources[:15]:
                    # Vision mode returns plain dicts; text mode returns Document objects
                    meta = doc if isinstance(doc, dict) else doc.metadata
                    fname = meta.get('filename', 'Unknown')
                    page = meta.get('page', 'N/A')
```

- [ ] **Step 7: Update mode label**

Find:
```python
            mode_label = "Agentic RAG" if request.use_agentic else "Traditional RAG"
```

Replace with:
```python
            if request.vision_mode:
                mode_label = "Vision RAG"
            elif request.use_agentic:
                mode_label = "Agentic RAG"
            else:
                mode_label = "Traditional RAG"
```

- [ ] **Step 8: Smoke-test the server starts without errors**

```
python -c "import server; print('server imports OK')"
```
Expected: `server imports OK`

- [ ] **Step 9: Commit**

```bash
git add server.py
git commit -m "feat(server): route to VisionRAG when vision_mode=True"
```

---

## Task 3: Frontend — Vision Mode toggle

**Files:**
- Modify: `static/index.html`
- Modify: `static/app.js`
- Modify: `static/style.css`

**Interfaces:**
- Consumes: all API endpoints now accept `vision_mode` param (Task 2)
- Produces: sidebar toggle that switches mode; `vision_mode` state sent on every API call

- [ ] **Step 1: Add Vision Mode toggle to `index.html`**

Find:
```html
                <div class="upload-options">
                    <label class="toggle-label">
                        <input type="checkbox" id="ocr-toggle">
                        <span class="toggle-slider"></span>
                        <span>Paid OCR (~$0.001/page)</span>
                    </label>
                </div>
```

Replace with:
```html
                <div class="upload-options">
                    <label class="toggle-label">
                        <input type="checkbox" id="vision-mode-toggle">
                        <span class="toggle-slider"></span>
                        <span>Vision Mode</span>
                    </label>
                    <label class="toggle-label" id="ocr-toggle-label">
                        <input type="checkbox" id="ocr-toggle">
                        <span class="toggle-slider"></span>
                        <span>Paid OCR (~$0.001/page)</span>
                    </label>
                </div>
```

Also add a mode badge after the header title. Find:
```html
                <div class="header-title" id="header-title">New Chat</div>
```

Replace with:
```html
                <div class="header-title" id="header-title">New Chat</div>
                <span class="mode-badge text" id="mode-badge">Text</span>
```

- [ ] **Step 2: Add `visionMode` state and DOM binding to `app.js`**

Find:
```js
    // ===== State =====
    let currentConvId = null;
    let isProcessing = false;
    let selectedUploadFiles = [];
    let currentAbortController = null;
```

Replace with:
```js
    // ===== State =====
    let currentConvId = null;
    let isProcessing = false;
    let selectedUploadFiles = [];
    let currentAbortController = null;
    let visionMode = false;
```

Find (in the DOM elements block, after the last existing constant):
```js
    const hljsTheme = $('#hljs-theme');
```

Replace with:
```js
    const hljsTheme = $('#hljs-theme');
    const visionModeToggle = $('#vision-mode-toggle');
    const ocrToggleLabel = $('#ocr-toggle-label');
    const modeBadge = $('#mode-badge');
```

- [ ] **Step 3: Add Vision Mode toggle event listener to `bindEvents()`**

Find (in `bindEvents()`, after the theme toggle block):
```js
        // New chat
        btnNewChat.addEventListener('click', newConversation);
```

Insert before that line:
```js
        // Vision Mode toggle
        visionModeToggle.addEventListener('change', () => {
            visionMode = visionModeToggle.checked;
            // OCR toggle is irrelevant in vision mode
            ocrToggleLabel.style.opacity = visionMode ? '0.4' : '1';
            ocrToggleLabel.style.pointerEvents = visionMode ? 'none' : '';
            // Update badge
            modeBadge.textContent = visionMode ? 'Vision' : 'Text';
            modeBadge.className = `mode-badge ${visionMode ? 'vision' : 'text'}`;
            // Reload file list for the active mode
            loadFiles();
        });

        // New chat
```

- [ ] **Step 4: Update `loadFiles()` to pass `vision_mode`**

Find:
```js
    async function loadFiles() {
        try {
            const res = await fetch('/api/files');
```

Replace with:
```js
    async function loadFiles() {
        try {
            const res = await fetch(`/api/files?vision_mode=${visionMode}`);
```

- [ ] **Step 5: Update `uploadFiles()` to pass `vision_mode`**

Find:
```js
        const formData = new FormData();
        selectedUploadFiles.forEach(f => formData.append('files', f));
        formData.append('enable_ocr', ocrToggle.checked);
```

Replace with:
```js
        const formData = new FormData();
        selectedUploadFiles.forEach(f => formData.append('files', f));
        formData.append('enable_ocr', ocrToggle.checked);
        formData.append('vision_mode', visionMode);
```

- [ ] **Step 6: Update `sendMessage()` to pass `vision_mode`**

Find:
```js
                body: JSON.stringify({
                    message: message,
                    conversation_id: currentConvId,
                    selected_files: selectedFiles.length > 0 ? selectedFiles : null,
                    use_agentic: agenticToggle.checked
                }),
```

Replace with:
```js
                body: JSON.stringify({
                    message: message,
                    conversation_id: currentConvId,
                    selected_files: selectedFiles.length > 0 ? selectedFiles : null,
                    use_agentic: agenticToggle.checked,
                    vision_mode: visionMode
                }),
```

- [ ] **Step 7: Update delete handler to pass `vision_mode`**

Find:
```js
                                const response = await fetch(`/api/files/${encodeURIComponent(btn.dataset.file)}`, { method: 'DELETE' });
```

Replace with:
```js
                                const response = await fetch(`/api/files/${encodeURIComponent(btn.dataset.file)}?vision_mode=${visionMode}`, { method: 'DELETE' });
```

- [ ] **Step 8: Add mode badge styles to `style.css`**

Append to the end of `static/style.css`:

```css
/* ── Vision mode badge ──────────────────────────────────────────── */
.mode-badge {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 10px;
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.5px;
    text-transform: uppercase;
    margin-left: 8px;
    vertical-align: middle;
    flex-shrink: 0;
}

.mode-badge.vision {
    background: var(--accent-color, #7c6ef4);
    color: #fff;
}

.mode-badge.text {
    background: var(--border-color, #333);
    color: var(--text-muted, #888);
}
```

- [ ] **Step 9: Manual smoke test**

Start the server:
```
python server.py
```

Open `http://localhost:8000` and verify:
1. "Vision Mode" toggle appears in sidebar under Upload section
2. "Text" badge visible in chat header
3. Toggle ON → badge changes to "Vision", file list reloads (empty), OCR toggle grays out
4. Toggle OFF → badge reverts to "Text", file list shows text-indexed files
5. Upload a PDF in Vision Mode → completes without error, file appears in list
6. Ask a question in Vision Mode → answer arrives citing page numbers

- [ ] **Step 10: Commit**

```bash
git add static/index.html static/app.js static/style.css
git commit -m "feat(ui): add Vision Mode toggle with mode badge and API routing"
```
