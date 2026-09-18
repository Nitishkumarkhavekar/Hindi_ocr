"""Schema tests (no models): the Gemma record structure, the page document, reading order, flags. Fast, CPU only."""
import json
import numpy as np
from hindi_ocr import schema
from hindi_ocr.reading_order import order_lines


def line(text, conf=0.99, y=0, x=0):
    return {"id": "l", "text": text, "confidence": conf, "min_conf": conf, "bbox": [x, y, x + 100, y + 20], "polygon": [], "script": schema.script_of(text), "flags": []}


def test_gemma_record_structure_matches_downstream():
    p1 = schema.page_document(0, 100, 100, "d.pdf#1", [line("विक्रय विलेख"), line("Sale Deed")], {})
    p2 = schema.page_document(1, 100, 100, "d.pdf#2", [line("कुल राशि 100")], {})
    rec = schema.gemma_record([p1, p2])
    assert list(rec) == ["prompt", "ocr_text", "output"]
    assert rec["prompt"] == schema.GEMMA_PROMPT and rec["prompt"][0].startswith("Extract structured data")
    assert rec["ocr_text"] == {"Page 1": ["विक्रय विलेख", "Sale Deed"], "Page 2": ["कुल राशि 100"]}
    assert set(rec["output"]) == {"buyer_details", "seller_details", "confirming_party_details", "property_details", "document_details"}
    assert rec["output"]["buyer_details"][0]["aadhaar_number"] is None and rec["output"]["confirming_party_details"] == []
    assert rec["output"]["property_details"]["sale_consideration"] is None and rec["output"]["document_details"]["transaction_date"] is None
    assert schema.validate_gemma_record(rec)
    assert schema.validate_gemma_record(json.loads(json.dumps(rec, ensure_ascii=False)))   # survives serialisation


def test_gemma_record_with_verified_output_and_empty_lines_dropped():
    p = schema.page_document(0, 100, 100, "d.pdf#1", [line("x"), line("")], {})
    out = json.loads(json.dumps(schema.GEMMA_OUTPUT_TEMPLATE))
    out["buyer_details"][0]["name"] = "राम कुमार"
    rec = schema.gemma_record([p], output=out)
    assert rec["ocr_text"]["Page 1"] == ["x"] and rec["output"]["buyer_details"][0]["name"] == "राम कुमार"
    assert schema.validate_gemma_record(rec)


def test_validate_rejects_wrong_shapes():
    import pytest
    good = schema.gemma_record([schema.page_document(0, 1, 1, "a", [line("x")], {})])
    for bad in [{**good, "extra": 1}, {**good, "ocr_text": {"1": ["x"]}}, {**good, "ocr_text": {"Page 1": "x"}},
                {**good, "output": {**good["output"], "buyer_details": [{"name": None}]}}]:
        with pytest.raises(ValueError):
            schema.validate_gemma_record(bad)


def test_script_and_flags():
    assert schema.script_of("विक्रय") == "hi" and schema.script_of("Deed") == "en" and schema.script_of("डीड Sale") == "mixed" and schema.script_of("1234") == "num"
    th = {"low_conf": 0.8, "check": 0.95}
    assert schema.line_flags(0.5, 0.5, "x", th) == ["low_conf"]
    assert schema.line_flags(0.9, 0.9, "x", th) == ["check"]
    assert schema.line_flags(0.99, 0.3, "डीड Sale", th) == ["weak_char", "script_mix"]
    assert schema.line_flags(0.99, 0.9, "x", th) == []


def box(x0, y0, x1, y1, skew=0.0):
    return np.array([[x0, y0 + skew], [x1, y0], [x1, y1 - skew], [x0, y1]], np.float32)


def test_reading_order_rows_and_skew():
    b = [box(0, 100, 500, 130), box(0, 40, 500, 70), box(600, 40, 900, 70), box(0, 160, 500, 190)]
    assert order_lines(b) == [1, 2, 0, 3]                                  # row 1: two side-by-side boxes, then rows 2, 3
    skewed = [box(0, 60 + 40 * k, 900, 90 + 40 * k, skew=25) for k in range(6)]   # consecutive tilted lines whose bboxes overlap
    assert order_lines(skewed) == list(range(6))
    assert order_lines(list(reversed(skewed))) == list(reversed(range(6)))
    assert order_lines([]) == []


def test_text_and_tsv():
    p1 = schema.page_document(0, 1, 1, "d.pdf#1", [line("a")], {})
    p2 = schema.page_document(1, 1, 1, "d.pdf#2", [line("b")], {})
    assert schema.to_text([p1, p2]) == "a\n\n----- page 2/2 -----\n\nb\n"
    t = schema.to_tsv([p1]).splitlines()
    assert t[0].startswith("page\tline") and t[1].startswith("d.pdf#1\tl\t")
