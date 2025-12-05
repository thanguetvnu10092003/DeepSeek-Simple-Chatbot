import fitz
import os

from typing import List
from langchain_core.documents import Document
from pathlib import Path
import replicate


class OCRPDFLoader:
    def __init__(self, file_path: str, text_threshold: int = 50, enable_ocr: bool = True):
        self.file_path = file_path
        self.text_threshold = text_threshold
        self.enable_ocr = enable_ocr
        self.ocr_used = False

    def load(self) -> tuple[List[Document], int, int]:
        doc = fitz.open(self.file_path)
        documents = []
        ocr_pages = 0
        skipped_pages = 0

        for i, page in enumerate(doc):
            text = page.get_text()

            if len(text.strip()) < self.text_threshold:
                if self.enable_ocr:
                    text = self._ocr_page(page, i)
                    ocr_pages += 1
                else:
                    skipped_pages += 1
                    continue

            if text.strip():
                documents.append(Document(
                    page_content=text.strip(),
                    metadata={
                        "source": self.file_path,
                        "page": i + 1,
                        "filename": Path(self.file_path).name
                    }
                ))

        doc.close()
        self.ocr_used = ocr_pages > 0
        return documents, ocr_pages, skipped_pages

    def _ocr_page(self, page, page_index, temp_dir="./temp_ocr"):
        os.makedirs(temp_dir, exist_ok=True)
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
        img_path = f"{temp_dir}/page_{page_index}.png"
        pix.save(img_path)

        with open(img_path, "rb") as f:
            result = replicate.run(
                "lucataco/deepseek-ocr:cb3b474fbfc56b1664c8c7841550bccecbe7b74c30e45ce938ffca1180b4dff5",
                input={"image": f}
            )
        os.remove(img_path)
        return result