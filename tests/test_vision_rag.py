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
