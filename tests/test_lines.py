"""Line helpers and row grouping (no models): long-line splitting at gaps, digit-script and danda normalisation,
row-joined Gemma entries."""
import numpy as np
from hindi_ocr.lines import split_at_gaps, digit_script_consistent, page_digit_script_consistent, danda_consistent
from hindi_ocr.reading_order import order_rows, order_lines
from hindi_ocr import schema


def strip(words, h=40, gap=20, glyph=30):
    """White strip with black blocks for words separated by `gap` empty columns."""
    w = sum(glyph * n for n in words) + gap * (len(words) - 1)
    img = np.full((h, w, 3), 255, np.uint8)
    x = 0
    for n in words:
        img[8:h - 8, x:x + glyph * n] = 0
        x += glyph * n + gap
    return img


def test_split_at_gaps_only_wide_lines_and_at_empty_columns():
    narrow = strip([2, 2])
    assert len(split_at_gaps(narrow, max_ratio=25)) == 1
    wide = strip([10, 10, 10, 10, 10])                     # 1500 px wide, 40 high -> ratio 37.5
    pieces = split_at_gaps(wide, max_ratio=25)
    assert len(pieces) >= 2 and pieces[0][0] == 0 and sum(c.shape[1] for _, c in pieces) == wide.shape[1]
    for x, c in pieces:                                    # every chunk boundary sits in a gap: chunk edges are white columns
        assert c.shape[1] / c.shape[0] <= 25 + 1
        assert c[:, 0].min() == 255 or x == 0
    solid = np.zeros((40, 1500, 3), np.uint8)              # no gap at all -> no split
    assert len(split_at_gaps(solid, max_ratio=25)) == 1


def test_digit_script_consistency_token_line_page():
    assert digit_script_consistent("रजिस्ट्रेशन फीस की राशि 5000० रुपये") == "रजिस्ट्रेशन फीस की राशि 50000 रुपये"
    assert digit_script_consistent("राशि- 18६57600 रुपये") == "राशि- 18657600 रुपये"
    assert digit_script_consistent("कुल १५ रुपये") == "कुल १५ रुपये"                          # single-script line untouched
    assert digit_script_consistent("दिनांक 08-08-2022 समय ११:16") == "दिनांक 08-08-2022 समय 11:16"
    assert digit_script_consistent("Sector-14 Gurugram") == "Sector-14 Gurugram"           # not a digit-only token
    page = ["राशि 24900000 रुपये", "स्टाम्प का मूल्य- १०१ रुपये", "EChallan:93277576", "सेवा शुल्क- 200"]
    assert page_digit_script_consistent(page)[1] == "स्टाम्प का मूल्य- 101 रुपये"
    dev_page = ["राशि २४९००००० रुपये", "मूल्य १०१ रुपये", "दिनांक ०८-०८-२०२२", "शुल्क 200"]  # a Devanagari-numeral page converts the stray Latin token
    assert page_digit_script_consistent(dev_page)[3] == "शुल्क २००"
    assert page_digit_script_consistent(["1234 और ५६७८"]) == ["1234 और ५६७८"]              # no dominant script -> untouched


def test_danda_only_in_devanagari_context():
    assert danda_consistent("किया गया | दोनों पक्षों") == "किया गया । दोनों पक्षों"
    assert danda_consistent("प्रस्तुत किया गया |") == "प्रस्तुत किया गया ।"
    assert danda_consistent("a | b") == "a | b"
    assert danda_consistent("Rs. 100 | Paid") == "Rs. 100 | Paid"


def box(x0, y0, x1, y1):
    return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], np.float32)


def test_rows_group_side_by_side_cells():
    b = [box(0, 100, 300, 130), box(400, 102, 700, 132), box(0, 160, 700, 190), box(0, 40, 300, 70)]
    assert order_rows(b) == [[3], [0, 1], [2]]
    assert order_lines(b) == [3, 0, 1, 2]


def line(text, row):
    return {"id": "l", "row": row, "text": text, "confidence": 0.99, "min_conf": 0.9, "bbox": [0, 0, 1, 1], "polygon": [], "script": "hi", "flags": []}


def test_gemma_entries_join_rows():
    page = schema.page_document(0, 1, 1, "d.pdf#1", [line("मकान", 0), line("1738.08 Sq.Feet", 0), line("भूमि का विवरण", 1), line("", 1), line("निवासीय", 2), line("420 Sq. Meters", 2)], {})
    assert schema.page_entries(page, join_rows=True) == ["मकान | 1738.08 Sq.Feet", "भूमि का विवरण", "निवासीय | 420 Sq. Meters"]
    assert schema.page_entries(page, join_rows=False) == ["मकान", "1738.08 Sq.Feet", "भूमि का विवरण", "निवासीय", "420 Sq. Meters"]
    rec = schema.gemma_record([page])
    assert rec["ocr_text"]["Page 1"][0] == "मकान | 1738.08 Sq.Feet" and schema.validate_gemma_record(rec)
