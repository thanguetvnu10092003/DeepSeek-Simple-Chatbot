# Vision RAG — Design Spec

**Date:** 2026-06-29  
**Status:** Approved  

---

## Overview

Add a **Vision RAG mode** alongside the existing Text RAG pipeline. Instead of converting PDF pages to text via OCR and then embedding text tokens, Vision RAG renders each page as an image, encodes it with a CLIP vision encoder, and uses a Replicate VLM to generate answers from the raw page images.

The two modes are fully independent and run in parallel. The user switches between them via a sidebar toggle.

---

## Goals

- Preserve existing Text RAG without any changes or regression risk
- Enable retrieval from document pages using visual embeddings (CLIP)
- Enable answer generation by passing page images directly to a VLM (no OCR needed)
- Expose a "Vision Mode" toggle in the UI sidebar

---

## Architecture

### Index Phase (PDF Upload)

```
PDF file
  → pymupdf: render each page → PIL Image (PNG)
  → resize to max 1024px (memory + API safety)
  → save PNG to ./vision_store/pages/<filename>/page_N.png
  → CLIP image encoder (SentenceTransformer, local CPU) → 512-dim embedding
  → upsert embedding + metadata into ChromaDB ("vision_chroma_db")
```

### Query Phase (User Question)

```
text question
  → CLIP text encoder (same SentenceTransformer model, local CPU)
  → similarity_search ChromaDB → top-k entries (image_path, page, filename)
  → load top-3 to top-5 page PNGs from disk
  → upload images + question to Replicate VLM
      model: meta/llama-3.2-11b-vision-instruct
  → answer text returned to user
```

---

## Components

### New file: `vision_rag.py`

Class `VisionRAG` with the following public interface:

```python
class VisionRAG:
    def __init__(self,
                 persist_directory="./vision_chroma_db",
                 pages_directory="./vision_store/pages")

    def add_pdf(self, pdf_path: str, progress: Callable = None) -> tuple[int, int]
        # Returns (pages_indexed, pages_skipped)

    def get_all_files(self) -> List[str]

    def delete_file(self, filename: str) -> None
        # Removes ChromaDB entries + PNG files from disk

    def query(self, question: str,
              target_files: list = None,
              history: list = None) -> tuple[str, list]
        # Returns (answer, sources)
        # sources: list of dicts with {filename, page, image_path}
```

**Internal details:**
- CLIP model: `clip-ViT-B-32` via `SentenceTransformer` (already in requirements, ~350MB download on first run)
- ChromaDB collection name: `"vision_chunks"`
- Top-k retrieval: `k=5` pages sent to VLM, configurable
- Each ChromaDB entry metadata: `{filename, page, image_path}`

### Modified file: `server.py`

- Instantiate `VisionRAG` at startup alongside `LangChainPDFRAG`
- All existing endpoints (`/upload`, `/query`, `/files`, `/delete`) receive an additional `vision_mode: bool = False` parameter
- Route to `vision_system` or `rag_system` based on this flag
- No changes to existing text RAG logic

### Modified file: Frontend JS

- Add a toggle switch "Vision Mode" in the sidebar near the file list
- When toggled: reload file list, show badge in chat header ("Vision" or "Text")
- `vision_mode` flag sent as part of every upload/query/files API request

### No changes to:
- `rag.py`
- `agentic_rag.py`
- `llm.py`
- `pdf_ocr_loader.py`
- `chat_history.py`

---

## Storage Layout

```
./vision_store/
  pages/
    document1.pdf/
      page_1.png
      page_2.png
    document2.pdf/
      page_1.png

./vision_chroma_db/         ← ChromaDB with CLIP visual embeddings
./vision_chroma_db_large/   ← not used (vision has no small/large split)
```

---

## Replicate VLM

**Model:** `meta/llama-3.2-11b-vision-instruct`  
**Input:** up to 5 PNG images (base64 or URL) + question text  
**Output:** text answer  
**Reuses:** existing `REPLICATE_API_TOKEN` from `.env`

Prompt sent to VLM:
```
You are a document analysis assistant. 
The following pages are from a PDF document.
Answer the question based ONLY on what you see in the images.
Cite the page number for each claim.

Question: {question}
```

---

## Error Handling

| Situation | Handling |
|---|---|
| Replicate API error during query | Return error message string, do not crash |
| CLIP weights not yet downloaded | Auto-download on first `VisionRAG()` init, log progress |
| Page render fails (corrupted page) | Skip page, log warning, continue |
| No documents indexed in vision mode | Return `"No documents in Vision Mode yet."` |
| Image too large for Replicate | Resize to max 1024px before sending |
| Vision ChromaDB missing on startup | Create fresh, log info |

---

## Dependencies

No new packages required beyond what is already in `requirements.txt`:
- `sentence-transformers>=3.0.0` — already present, supports CLIP models
- `Pillow>=10.0.0` — already present, used for image resize
- `pymupdf>=1.24.0` — already present, used for page rendering
- `chromadb>=0.5.0` — already present, used for vision embeddings
- `replicate>=1.0.0` — already present, used for VLM calls

**Only addition:** CLIP model weights download (~350MB) on first `VisionRAG()` initialization.

---

## Out of Scope

- Agentic Vision RAG (no LangGraph integration for vision mode in this iteration)
- ColPali / patch-level embeddings
- Cross-mode search (querying both Text and Vision simultaneously)
- Vision mode for the existing Agentic RAG tab
