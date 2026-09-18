"""End-to-end on a synthetic Hindi page: models load, every rendered line is detected, its key token is recognised, and the
Gemma record validates. Runs on whatever provider is available (HINDI_OCR_CPU=1 forces CPU)."""
import json
import os
import pytest
import numpy as np
from hindi_ocr import HindiOCR, validate_gemma_record


@pytest.fixture(scope="module")
def ocr():
    return HindiOCR()


def test_ready_reports_models_and_providers(ocr):
    r = ocr.ready()
    assert r["providers"]["det"]["provider"] in ("cuda", "cpu") and r["providers"]["rec"]["provider"] in ("cuda", "cpu")
    assert r["model"]["rec_backend"] == "paddle_devanagari" and len(r["model"]["rec_model_md5"]) == 12 and r["model"]["rec_classes"] == 570


def test_synthetic_page_recognised(ocr, synthetic_page):
    img, expected = synthetic_page
    doc = ocr.page(img, index=0, file="synthetic.png")
    assert doc["schema"] == "hindi-ocr/1.0" and doc["page"]["width"] == img.shape[1] and doc["page"]["file"] == "synthetic.png"
    texts = [l["text"] for l in doc["lines"]]
    assert len(texts) >= len(expected), texts
    joined = "\n".join(texts)
    missing = [tok for _, tok in expected if tok not in joined]
    assert not missing, f"not recognised: {missing}\n{joined}"
    assert texts.index(next(t for t in texts if "विलेख" in t)) < texts.index(next(t for t in texts if "विवरण" in t))   # reading order
    for l in doc["lines"]:
        assert 0 <= l["min_conf"] <= l["confidence"] <= 1 and len(l["bbox"]) == 4 and len(l["polygon"]) == 4
        assert l["script"] in ("hi", "en", "num", "mixed", "other")


def test_gemma_record_from_page_images(ocr, synthetic_page):
    img, expected = synthetic_page
    rec = ocr.gemma([img, img])
    assert validate_gemma_record(rec)
    assert list(rec["ocr_text"]) == ["Page 1", "Page 2"] and any("विलेख" in t for t in rec["ocr_text"]["Page 1"])
    assert rec["output"]["seller_details"][0]["pan_card_number"] is None
    json.dumps(rec, ensure_ascii=False)


def test_cli_writes_gemma_file(tmp_path, synthetic_page):
    import cv2
    from hindi_ocr.cli import main
    img, _ = synthetic_page
    src = tmp_path / "deed1.png"
    cv2.imwrite(str(src), img)
    assert main(["--input", str(src), "--out-dir", str(tmp_path / "out"), "--format", "all", "--quiet"]) == 0
    files = sorted(p.name for p in (tmp_path / "out").iterdir())
    assert files == ["deed1.json", "deed1.tsv", "deed1.txt", "deed1_ocr.json"]
    rec = json.loads((tmp_path / "out" / "deed1_ocr.json").read_text(encoding="utf-8"))
    assert validate_gemma_record(rec) and list(rec["ocr_text"]) == ["Page 1"]
    full = json.loads((tmp_path / "out" / "deed1.json").read_text(encoding="utf-8"))
    assert full["schema"] == "hindi-ocr/1.0-document" and full["page_count"] == 1 and full["pages"][0]["lines"]
