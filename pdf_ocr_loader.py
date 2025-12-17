import fitz
import os
import time
import logging

from typing import List, Tuple
from langchain_core.documents import Document
from pathlib import Path
import replicate

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def retry_with_backoff(max_retries: int = 3, base_delay: float = 2.0):
    """Retry decorator with exponential backoff for API calls"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    delay = base_delay * (2 ** attempt)
                    logger.warning(f"OCR attempt {attempt + 1}/{max_retries} failed: {e}. Retrying in {delay}s...")
                    time.sleep(delay)
            logger.error(f"All {max_retries} OCR attempts failed")
            raise last_exception
        return wrapper
    return decorator


class OCRPDFLoader:
    def __init__(self, file_path: str, text_threshold: int = 50, enable_ocr: bool = True):
        self.file_path = file_path
        self.text_threshold = text_threshold
        self.enable_ocr = enable_ocr
        self.ocr_used = False
        self.temp_dir = "./temp_ocr"

    def load(self) -> Tuple[List[Document], int, int]:
        """Load PDF and return documents, OCR page count, and skipped page count"""
        logger.info(f"Loading PDF: {self.file_path}")
        doc = fitz.open(self.file_path)
        documents = []
        ocr_pages = 0
        skipped_pages = 0

        try:
            for i, page in enumerate(doc):
                text = page.get_text()

                if len(text.strip()) < self.text_threshold:
                    if self.enable_ocr:
                        logger.info(f"Page {i + 1} needs OCR (text length: {len(text.strip())})")
                        text = self._ocr_page(page, i)
                        ocr_pages += 1
                    else:
                        logger.debug(f"Page {i + 1} skipped (OCR disabled)")
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

            self.ocr_used = ocr_pages > 0
            logger.info(f"PDF loaded: {len(documents)} pages, {ocr_pages} OCR, {skipped_pages} skipped")
            
        finally:
            doc.close()
            self._cleanup_temp_dir()
        
        return documents, ocr_pages, skipped_pages

    @retry_with_backoff(max_retries=3, base_delay=2.0)
    def _ocr_page(self, page, page_index: int) -> str:
        """OCR a single page with retry logic"""
        os.makedirs(self.temp_dir, exist_ok=True)
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
        img_path = f"{self.temp_dir}/page_{page_index}.png"
        pix.save(img_path)

        try:
            with open(img_path, "rb") as f:
                result = replicate.run(
                    "lucataco/deepseek-ocr:cb3b474fbfc56b1664c8c7841550bccecbe7b74c30e45ce938ffca1180b4dff5",
                    input={"image": f}
                )
            return result
        finally:
            # Always clean up the temp image
            if os.path.exists(img_path):
                os.remove(img_path)
    
    def _cleanup_temp_dir(self):
        """Clean up temporary OCR directory"""
        try:
            if os.path.exists(self.temp_dir):
                for file in os.listdir(self.temp_dir):
                    file_path = os.path.join(self.temp_dir, file)
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                os.rmdir(self.temp_dir)
                logger.debug("Temp OCR directory cleaned up")
        except Exception as e:
            logger.warning(f"Failed to cleanup temp directory: {e}")