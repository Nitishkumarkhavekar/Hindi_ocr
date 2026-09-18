"""Evaluate page-level OCR against a human-verified reference.

Reference file (JSON): [{"pdf": path, "page": 1-based, "lines": [verified printed lines in reading order],
                         "exclude": [[x0,y0,x1,y1], ...], "ref_dpi": 200}]   — boxes (page pixels at ref_dpi) that hold handwriting,
signatures, struck-through or truncated text; OCR lines whose centre falls inside one are ignored.

Metrics per page and overall:
  CER          Levenshtein distance between the reading-order page texts / reference characters (order-sensitive)
  word acc     1 - word-level Levenshtein / reference words (order-sensitive)
  bag-of-words recall   reference words found anywhere on the page (order-free)
  line exact   reference lines reproduced exactly (order-free)

    hindi-ocr-eval --reference ref.json [--config alt.yaml] [--json out.json]"""
import argparse
import json
import sys
import time
import unicodedata
from collections import Counter


def levenshtein(a, b):
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def norm(s):
    return " ".join(unicodedata.normalize("NFC", s).split())


def inside(bbox, boxes, margin=4):
    cx, cy = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
    return any(x0 - margin <= cx <= x1 + margin and y0 - margin <= cy <= y1 + margin for x0, y0, x1, y1 in boxes)


def score_page(ref_lines, ocr_lines):
    ref = [norm(l) for l in ref_lines]
    hyp = [norm(l) for l in ocr_lines if norm(l)]
    rt, ht = "\n".join(ref), "\n".join(hyp)
    cer = levenshtein(rt, ht) / max(len(rt), 1)
    rw, hw = rt.split(), ht.split()
    wer = levenshtein(rw, hw) / max(len(rw), 1)
    bag = sum((Counter(rw) & Counter(hw)).values()) / max(len(rw), 1)
    exact = sum(1 for l in ref if l in set(hyp)) / max(len(ref), 1)
    return {"cer": cer, "word_acc": 1 - wer, "bag_recall": bag, "line_exact": exact, "ref_chars": len(rt), "ref_words": len(rw), "ref_lines": len(ref)}


def evaluate(ocr, reference, verbose=True, dpi=None):
    from .io import pdf_pages
    results = []
    t0 = time.time()
    for item in reference:
        use_dpi = dpi or ocr.cfg["pdf"].get("dpi", 200)
        pages = pdf_pages(item["pdf"], dpi=use_dpi)
        img = next(im for k, im in enumerate(pages, 1) if k == item["page"])
        doc = ocr.page(img, index=item["page"] - 1, file=f"{item['pdf']}#{item['page']}")
        k = use_dpi / item.get("ref_dpi", 200)                      # exclusion boxes were drawn at the reference dpi
        excl = [[v * k for v in box] for box in item.get("exclude", [])]
        kept = [l["text"] for l in doc["lines"] if not inside(l["bbox"], excl, margin=4 * k)]
        s = score_page(item["lines"], kept)
        s.update({"pdf": item["pdf"], "page": item["page"], "ocr_lines": kept})
        results.append(s)
        if verbose:
            print(f"{item['pdf'].split('/')[-1][:24]:24} p{item['page']}: CER {s['cer']*100:5.2f}%  word acc {s['word_acc']*100:5.1f}%  "
                  f"bag {s['bag_recall']*100:5.1f}%  line exact {s['line_exact']*100:5.1f}%  ({s['ref_lines']} lines)", file=sys.stderr)
    C = sum(r["ref_chars"] for r in results)
    W = sum(r["ref_words"] for r in results)
    L = sum(r["ref_lines"] for r in results)
    overall = {"cer": sum(r["cer"] * r["ref_chars"] for r in results) / C, "word_acc": sum(r["word_acc"] * r["ref_words"] for r in results) / W,
               "bag_recall": sum(r["bag_recall"] * r["ref_words"] for r in results) / W, "line_exact": sum(r["line_exact"] * r["ref_lines"] for r in results) / L,
               "pages": len(results), "seconds": round(time.time() - t0, 1)}
    if verbose:
        print(f"{'OVERALL':24}    CER {overall['cer']*100:5.2f}%  word acc {overall['word_acc']*100:5.1f}%  bag {overall['bag_recall']*100:5.1f}%  "
              f"line exact {overall['line_exact']*100:5.1f}%  ({L} lines, {overall['seconds']} s)", file=sys.stderr)
    return overall, results


def main(argv=None):
    ap = argparse.ArgumentParser(prog="hindi-ocr-eval")
    ap.add_argument("--reference", required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--provider", default=None)
    ap.add_argument("--json", default=None, help="write per-page results here")
    ap.add_argument("--set", action="append", default=[], help="config override key=value, e.g. detector.unclip_ratio=1.8")
    a = ap.parse_args(argv)
    from .pipeline import HindiOCR
    overrides = {}
    for kv in a.set:
        k, _, v = kv.partition("=")
        overrides[k] = json.loads(v) if v and v[0] in "0123456789-.[{tfn\"" else v
    ocr = HindiOCR(config=a.config, providers=a.provider, **overrides)
    ref = json.load(open(a.reference, encoding="utf-8"))
    overall, results = evaluate(ocr, ref)
    if a.json:
        json.dump({"overall": overall, "pages": results}, open(a.json, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
