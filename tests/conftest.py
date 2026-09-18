"""Shared fixtures: a synthetic Hindi/English page rendered with a Devanagari font through PyMuPDF's Story layout, which
shapes complex scripts with HarfBuzz (PIL without libraqm draws matras unshaped, which is not what printed Hindi looks
like). No deed data lives in the repo."""
import os
import numpy as np
import pytest

FONT_DIRS = [os.environ.get("HINDI_OCR_TEST_FONT_DIR", ""), r"C:\Windows\Fonts", "/usr/share/fonts/truetype/noto", "/usr/share/fonts/truetype/lohit-devanagari"]
FONT_FILES = ["Nirmala.ttc", "mangal.ttf", "NotoSansDevanagari-Regular.ttf", "Lohit-Devanagari.ttf"]

# (text, key token the recognizer must reproduce — tokens the bootstrap PP-OCRv5 model reads reliably; the retrained
#  pack should get every word, tighten these then)
SAMPLE_LINES = [
    ("विक्रय विलेख", "विलेख"),
    ("प्रलेख संख्या: 1738 दिनांक: 20-04-2020", "1738"),
    ("तहसील वजीराबाद गांव हुड्डा सेक्टर 31", "तहसील"),
    ("कुल राशि 24900000 रुपये", "24900000"),
    ("Sale Deed Registration Office Gurugram", "Registration"),
    ("भूमि का विवरण: निवासीय 420 वर्ग मीटर", "विवरण"),
]


def find_font():
    for d in FONT_DIRS:
        for f in FONT_FILES:
            if d and os.path.isfile(os.path.join(d, f)):
                return d, f
    return None, None


@pytest.fixture(scope="session")
def synthetic_page():
    """-> (BGR image, SAMPLE_LINES); skips when PyMuPDF or a Devanagari font is missing."""
    fitz = pytest.importorskip("fitz")
    font_dir, font_file = find_font()
    if not font_file:
        pytest.skip("no Devanagari font found; set HINDI_OCR_TEST_FONT_DIR")
    css = f"@font-face {{font-family: dev; src: url({font_file});}} p {{font-family: dev; font-size: 26px; margin: 22px 0;}}"
    html = "".join(f"<p>{t}</p>" for t, _ in SAMPLE_LINES)
    story = fitz.Story(html=html, user_css=css, archive=fitz.Archive(font_dir))
    import io
    buf = io.BytesIO()
    writer = fitz.DocumentWriter(buf)
    device = writer.begin_page(fitz.Rect(0, 0, 800, 520))
    story.place(fitz.Rect(40, 20, 760, 500))
    story.draw(device)
    writer.end_page()
    writer.close()
    rendered = fitz.open("pdf", buf.getvalue())
    pix = rendered[0].get_pixmap(dpi=200)
    rendered.close()
    img = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, pix.n)[:, :, :3]
    return img[:, :, ::-1].copy(), SAMPLE_LINES
