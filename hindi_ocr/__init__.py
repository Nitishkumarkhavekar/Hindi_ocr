"""Hindi-OCR: standalone Devanagari document OCR producing hindi-ocr/1.0 page documents and Gemma extraction records."""
from .pipeline import HindiOCR, __version__   # noqa: F401
from .schema import gemma_record, validate_gemma_record, page_document, to_text, to_tsv   # noqa: F401
