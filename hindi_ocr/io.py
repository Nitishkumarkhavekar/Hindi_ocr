"""Inputs: image paths, image / PDF bytes, numpy arrays, directories and globs; PDFs are rendered page by page with pypdfium2."""
import glob
import os
from pathlib import Path
import cv2
import numpy as np

IMAGE_EXT = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
PDF_EXT = {".pdf"}


def is_pdf_bytes(x):
    return isinstance(x, (bytes, bytearray, memoryview)) and bytes(x[:5]) == b"%PDF-"


def to_bgr(x):
    """Path / bytes / array / PIL image -> uint8 BGR array."""
    if isinstance(x, np.ndarray):
        if x.ndim == 2:
            return cv2.cvtColor(x, cv2.COLOR_GRAY2BGR)
        if x.shape[2] == 4:
            return cv2.cvtColor(x, cv2.COLOR_BGRA2BGR)
        return np.ascontiguousarray(x)
    if isinstance(x, (bytes, bytearray, memoryview)):
        im = cv2.imdecode(np.frombuffer(bytes(x), np.uint8), cv2.IMREAD_COLOR)
        if im is None:
            raise ValueError("could not decode image bytes")
        return im
    if isinstance(x, (str, os.PathLike)):
        im = cv2.imdecode(np.fromfile(str(x), np.uint8), cv2.IMREAD_COLOR)   # fromfile: non-ASCII paths on Windows
        if im is None:
            raise ValueError(f"could not read image {x}")
        return im
    if hasattr(x, "convert"):   # PIL
        return cv2.cvtColor(np.asarray(x.convert("RGB")), cv2.COLOR_RGB2BGR)
    raise TypeError(f"unsupported input {type(x)}")


def expand_inputs(inputs):
    """file / directory / glob / list -> sorted list of image and PDF paths."""
    items = [inputs] if isinstance(inputs, (str, os.PathLike)) else list(inputs)
    out = []
    for item in items:
        p = Path(item)
        if p.is_dir():
            out += sorted(str(f) for f in p.iterdir() if f.suffix.lower() in IMAGE_EXT | PDF_EXT)
        elif p.is_file():
            out.append(str(p))
        else:
            out += sorted(f for f in glob.glob(str(item)) if Path(f).suffix.lower() in IMAGE_EXT | PDF_EXT)
    return out


def pdf_page_count(path):
    import pypdfium2 as pdfium
    doc = pdfium.PdfDocument(bytes(path) if isinstance(path, (bytes, bytearray)) else str(path))
    n = len(doc)
    doc.close()
    return n


def pdf_pages(path, dpi=200):
    """Yield BGR page images of a PDF (path or bytes)."""
    import pypdfium2 as pdfium
    import gc
    doc = pdfium.PdfDocument(bytes(path) if isinstance(path, (bytes, bytearray)) else str(path))
    try:
        for i in range(len(doc)):
            page = doc[i]
            bitmap = page.render(scale=dpi / 72.0)
            pil = bitmap.to_pil().convert("RGB")
            img = cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)
            for obj in (bitmap, page):                       # release pdfium objects deterministically, not via GC
                try:
                    obj.close()
                except Exception:
                    pass
            yield img
    finally:
        try:
            doc.close()
        except RuntimeError:                                  # pypdfium2 child-set mutated by a finalizer during close: retry once
            gc.collect()
            try:
                doc.close()
            except Exception:
                pass


def iter_sources(paths, dpi=200):
    """Yield (label, image-or-path) per page: 'name.pdf#3' for PDF pages, 'scan.png' for images."""
    for p in paths:
        if Path(p).suffix.lower() in PDF_EXT:
            for k, im in enumerate(pdf_pages(p, dpi=dpi)):
                yield f"{Path(p).name}#{k + 1}", im
        else:
            yield Path(p).name, p
