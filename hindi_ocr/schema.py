"""Output schemas.

1. Page document (`hindi-ocr/1.0`) — everything the pipeline knows about one page:
   {"schema", "model": {det_md5, rec backend/md5/charset_md5}, "page": {index, width, height, file},
    "lines": [{"id", "text", "confidence", "min_conf", "bbox": [x0,y0,x1,y1], "polygon": [[x,y]x4], "script", "flags"}],
    "text": reading-order plain text}

2. Gemma extraction record (`<deed>_ocr.json`) — the exact structure the downstream Gemma 4B pipeline consumes and
   the verified training files use:
   {"prompt": [prompt lines], "ocr_text": {"Page 1": [line, line, ...], "Page 2": [...]}, "output": {...}}
   `output` is the extraction target: for inference it is the empty template (every field null), for training it is
   the verified answer. Entries are the page's text lines in reading order; cells that share a row (table label + value)
   are joined with ' | ' (gemma.join_rows in the yaml). Nothing is rewritten."""
import json
import unicodedata
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCHEMA = "hindi-ocr/1.0"
GEMMA_PROMPT = (HERE / "gemma_prompt.txt").read_text(encoding="utf-8").split("\n")
GEMMA_OUTPUT_TEMPLATE = json.loads((HERE / "gemma_output_template.json").read_text(encoding="utf-8"))


def script_of(s):
    dev = sum(0x0900 <= ord(c) <= 0x097F for c in s)
    lat = sum(c.isascii() and c.isalpha() for c in s)
    if dev and lat:
        return "mixed"
    if dev:
        return "hi"
    if lat:
        return "en"
    return "num" if any(c.isdigit() for c in s) else "other"


def line_flags(conf, min_conf, text, thresholds):
    flags = []
    if conf < thresholds.get("low_conf", 0.8):
        flags.append("low_conf")
    elif conf < thresholds.get("check", 0.95):
        flags.append("check")
    if min_conf < 0.5:
        flags.append("weak_char")
    if script_of(text) == "mixed":
        flags.append("script_mix")
    return flags


def page_document(index, width, height, file, lines, model):
    text = "\n".join(l["text"] for l in lines)
    return {"schema": SCHEMA, "model": model, "page": {"index": index, "width": width, "height": height, "file": file},
            "lines": lines, "text": text}


ROW_SEP = " | "


def page_entries(page, join_rows=True):
    """ocr_text entries for one page: each text line, or — with join_rows — cells that share a row (table label + value,
    side-by-side fields) joined with ' | ' so the pair stays together for the extractor."""
    if not join_rows:
        return [l["text"] for l in page["lines"] if l["text"]]
    out, cur, cur_row = [], [], None
    for k, l in enumerate(page["lines"]):
        if not l["text"]:
            continue
        r = l.get("row", f"line{k}")                       # documents without row info: one entry per line
        if cur and r != cur_row:
            out.append(ROW_SEP.join(cur))
            cur = []
        cur.append(l["text"])
        cur_row = r
    if cur:
        out.append(ROW_SEP.join(cur))
    return out


def gemma_record(pages, output=None, prompt=None, join_rows=True):
    """pages: page documents of one deed in order -> the Gemma extraction record."""
    ocr_text = {f"Page {k}": page_entries(p, join_rows) for k, p in enumerate(pages, 1)}
    return {"prompt": list(prompt or GEMMA_PROMPT), "ocr_text": ocr_text,
            "output": json.loads(json.dumps(GEMMA_OUTPUT_TEMPLATE)) if output is None else output}


def validate_gemma_record(rec):
    """Raise ValueError if `rec` does not have the downstream structure."""
    if not isinstance(rec, dict) or set(rec) != {"prompt", "ocr_text", "output"}:
        raise ValueError(f"top-level keys must be prompt/ocr_text/output, got {sorted(rec) if isinstance(rec, dict) else type(rec)}")
    if not isinstance(rec["prompt"], list) or not all(isinstance(x, str) for x in rec["prompt"]):
        raise ValueError("prompt must be a list of strings")
    for k, v in rec["ocr_text"].items():
        if not (k.startswith("Page ") and k[5:].isdigit()):
            raise ValueError(f"ocr_text key {k!r} must be 'Page N'")
        if not isinstance(v, list) or not all(isinstance(x, str) for x in v):
            raise ValueError(f"ocr_text[{k!r}] must be a list of strings")
    out = rec["output"]
    if set(out) != set(GEMMA_OUTPUT_TEMPLATE):
        raise ValueError(f"output keys must be {sorted(GEMMA_OUTPUT_TEMPLATE)}")
    for section in ("buyer_details", "seller_details", "confirming_party_details"):
        for party in out[section]:
            if set(party) != set(GEMMA_OUTPUT_TEMPLATE["buyer_details"][0]):
                raise ValueError(f"{section} entry has wrong fields")
    for section in ("property_details", "document_details"):
        if set(out[section]) != set(GEMMA_OUTPUT_TEMPLATE[section]):
            raise ValueError(f"{section} has wrong fields")
    return True


def to_text(pages):
    parts = []
    for k, p in enumerate(pages, 1):
        parts.append((f"----- page {k}/{len(pages)} -----\n\n" if k > 1 else "") + p["text"])
    return "\n\n".join(parts) + "\n"


def to_tsv(pages):
    rows = ["page\tline\tbbox\ttext\tconfidence\tmin_conf\tflags"]
    for p in pages:
        for l in p["lines"]:
            rows.append("\t".join([p["page"]["file"], l["id"], ",".join(str(int(round(v))) for v in l["bbox"]), l["text"].replace("\t", " "),
                                   f"{l['confidence']:.4f}", f"{l['min_conf']:.4f}", " ".join(l["flags"])]))
    return "\n".join(rows) + "\n"


def nfc(s):
    return unicodedata.normalize("NFC", s)
