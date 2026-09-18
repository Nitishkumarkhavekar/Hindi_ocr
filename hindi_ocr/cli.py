"""`hindi-ocr` command line.

  hindi-ocr --input "D:/Deeds/Hind deeds/combined" --out-dir out/               # every PDF -> <deed>_ocr.json (Gemma record)
  hindi-ocr --input deed.pdf --out-dir out/ --format gemma,json,txt,tsv           # + full page documents, text, TSV
  hindi-ocr --input deed.pdf --out-dir out/ --provider cpu --limit 3

Formats: gemma (<deed>_ocr.json: prompt / ocr_text per page / empty output — the downstream Gemma 4B input),
json (<deed>.json: hindi-ocr/1.0 page documents with boxes, confidences and flags), txt, tsv.
Exit codes: 0 ok · 1 no input · 2 models failed to load · 3 some documents failed."""
import argparse
import json
import sys
import time
import traceback
from pathlib import Path

FORMATS = ["gemma", "json", "txt", "tsv"]


def build_parser():
    ap = argparse.ArgumentParser(prog="hindi-ocr", description="Hindi-OCR: Devanagari document OCR -> Gemma extraction records")
    ap.add_argument("--input", "-i", nargs="+", required=True, help="PDF / image file(s), directory or glob")
    ap.add_argument("--out-dir", "-o", default="hindi_ocr_out")
    ap.add_argument("--format", "-f", default="gemma", help=f"comma list of {FORMATS} or 'all'")
    ap.add_argument("--provider", choices=["auto", "cuda", "cpu"], default=None)
    ap.add_argument("--pdf-dpi", type=int, default=None)
    ap.add_argument("--config", default=None, help="alternative hindi_ocr.yaml")
    ap.add_argument("--limit", type=int, default=0, help="only the first N input files")
    ap.add_argument("--skip-existing", action="store_true", help="skip inputs whose first requested output already exists (resume a batch)")
    ap.add_argument("--quiet", "-q", action="store_true")
    return ap


def main(argv=None):
    a = build_parser().parse_args(argv)
    from .pipeline import HindiOCR
    from .io import expand_inputs
    from . import schema
    fmts = FORMATS if a.format == "all" else [f.strip() for f in a.format.split(",") if f.strip()]
    bad = [f for f in fmts if f not in FORMATS]
    if bad:
        print(f"unknown format(s) {bad}; choose from {FORMATS}", file=sys.stderr)
        return 1
    paths = expand_inputs(a.input)
    paths = paths[:a.limit] if a.limit else paths
    if not paths:
        print(f"no inputs match {a.input}", file=sys.stderr)
        return 1
    say = (lambda *x: None) if a.quiet else (lambda *x: print(*x, file=sys.stderr, flush=True))
    try:
        ocr = HindiOCR(config=a.config, providers=a.provider)
    except Exception as e:
        print(f"failed to load models: {type(e).__name__}: {e}", file=sys.stderr)
        traceback.print_exc()
        return 2
    r = ocr.ready()
    say(f"{r['name']} {r['version']} | det={r['providers']['det']['provider']} rec={r['providers']['rec']['provider']} "
        f"({r['model']['rec_backend']}) | load {r['load_s']}s | {len(paths)} input file(s)")
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    n_pages = n_docs = failed = 0
    suffix = {"gemma": "_ocr.json", "json": ".json", "txt": ".txt", "tsv": ".tsv"}[fmts[0]]
    skipped = 0
    for path in paths:
        stem = Path(path).stem
        if a.skip_existing and (out / (stem + suffix)).exists():
            skipped += 1
            continue
        pages = None
        for attempt in (1, 2):                                # one retry: transient failures (GPU arena, pdfium finalizer race)
            try:
                pages = ocr.pages(path, pdf_dpi=a.pdf_dpi)
                break
            except Exception as e:
                print(f"{Path(path).name}: attempt {attempt} failed: {type(e).__name__}: {e}", file=sys.stderr)
                if attempt == 2:
                    traceback.print_exc()
        if pages is None:
            failed += 1
            continue
        n_pages += len(pages)
        n_docs += 1
        if "gemma" in fmts:
            rec = schema.gemma_record(pages, join_rows=ocr.cfg["gemma"].get("join_rows", True))
            schema.validate_gemma_record(rec)
            (out / f"{stem}_ocr.json").write_text(json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8")
        if "json" in fmts:
            (out / f"{stem}.json").write_text(json.dumps({"schema": "hindi-ocr/1.0-document", "file": Path(path).name, "page_count": len(pages),
                                                          "model": ocr.model, "pages": pages}, ensure_ascii=False, indent=1), encoding="utf-8")
        if "txt" in fmts:
            (out / f"{stem}.txt").write_text(schema.to_text(pages), encoding="utf-8")
        if "tsv" in fmts:
            (out / f"{stem}.tsv").write_text(schema.to_tsv(pages), encoding="utf-8")
        lines = sum(len(p["lines"]) for p in pages)
        conf = [l["confidence"] for p in pages for l in p["lines"]]
        say(f"{Path(path).name}: {len(pages)} pages, {lines} lines, mean conf {sum(conf) / max(len(conf), 1):.3f}, "
            f"low_conf {sum('low_conf' in l['flags'] for p in pages for l in p['lines'])}")
    wall = time.time() - t0
    st = ocr.last_stats
    say(f"done: {n_pages} pages / {n_docs} document(s) in {wall:.1f} s = {n_pages / max(wall, 1e-9):.2f} pages/s | "
        f"ms/page {st.get('ms_per_page', {})} | failed {failed} | skipped {skipped} | wrote {out}")
    return 3 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
