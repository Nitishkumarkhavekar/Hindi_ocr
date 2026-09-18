# Hindi-OCR

Standalone OCR for printed **Hindi (Devanagari) + English** documents — sale deeds first — producing the record the
downstream **Gemma 4B extraction pipeline** consumes. It shares nothing with `arjuna-ocr-kn-en` (the Kannada/English
package): separate models, code, config and outputs.

```
hindi_ocr/
  hindi_ocr.yaml            config (models, provider, thresholds, dpi)
  models/det/inference.onnx PP-OCRv5 mobile text detector (DB), script-agnostic            4.8 MB
  models/rec/inference.onnx devanagari_PP-OCRv5_mobile_rec — the bootstrap recognizer     7.9 MB
  models/rec/charset.json   568 characters (CTC classes: blank, charset, space = 570)
  models/rec_en/            en_PP-OCRv5_mobile_rec — secondary recognizer for pure-Latin lines                 7.9 MB
  hindi_ocr/                package: config · providers · io · detector · recognizer · lines · reading_order · schema · pipeline · cli · evaluate
  tests/                    schema tests (no models) + end-to-end on a synthetic shaped-Devanagari page (no deed data)
```

Both models are official PaddlePaddle ONNX exports (Apache-2.0) and run on ONNX Runtime — CUDA when available, CPU otherwise.

## Install / run

```powershell
cd "D:\Kanen OCR\hindi_ocr"
pip install -e .                                  # onnxruntime-gpu, opencv, pyclipper, shapely, pyyaml, pypdfium2
hindi-ocr --input "D:\Deeds\Hind deeds\combined" --out-dir "D:\Deeds\Hind deeds\OCR_Output" --format gemma,json,txt
python -m pytest tests                            # 15 tests, ~5 s
hindi-ocr-eval --reference "D:/Deeds/Hind deeds/eval/reference.json"   # CER / word accuracy against the verified pages
```

```python
from hindi_ocr import HindiOCR
ocr = HindiOCR()                       # ocr.ready() -> providers, model hashes, fallbacks
rec = ocr.gemma("deed.pdf")            # {"prompt": [...], "ocr_text": {"Page 1": [...]}, "output": {...}}
pages = ocr.pages("deed.pdf")          # hindi-ocr/1.0 page documents: lines with bbox, polygon, confidence, flags
doc = ocr.page("scan.png")
```

## Outputs (`--format`)

| format | file | content |
|---|---|---|
| `gemma` (default) | `<deed>_ocr.json` | **the Gemma record**: `prompt` (the extraction prompt, line list), `ocr_text` `{"Page N": [line, …]}` in reading order, `output` = the empty extraction template (all fields `null`). Same structure as the verified training files (`*_ocr.json`). |
| `json` | `<deed>.json` | `hindi-ocr/1.0-document`: every page with `lines[]` (`text`, `confidence`, `min_conf`, `bbox`, `polygon`, `script`, `flags`), `model` hashes |
| `txt` | `<deed>.txt` | plain text, `----- page k/n -----` separators |
| `tsv` | `<deed>.tsv` | one row per line with bbox and confidence |

Flags per line: `low_conf` (< 0.80), `check` (0.80–0.95), `weak_char` (weakest character < 0.5), `script_mix`.
Nothing is filtered or rewritten; `recognizer.drop_score` in the yaml can drop lines below a confidence if wanted.

## Swapping the recognizer (Route A)

The recognizer is a registered backend behind one interface (`hindi_ocr/recognizer.py`):

| backend | model | status |
|---|---|---|
| `paddle_devanagari` | PP-OCRv5 mobile Devanagari, 48 px, 570 classes | **current** — bootstrap; also used to pseudo-label the deeds for training |
| `arjuna_hindi` | retrained SVTRv2-B KN+HI+EN pack, 32 px | slot ready — drop the exported ONNX + `charset.json` into `models/rec/`, set `recognizer.backend: arjuna_hindi` |

Nothing else changes: detector, reading order, schema, CLI and tests stay as they are, and every output carries the
recognizer's backend name, model md5 and charset md5 in `model`, so results are traceable to the model that made them.

## Pipeline

PDF → pages at 200 dpi (pypdfium2) → DB detector (longest side ≤ 1920) → perspective line crops → recognizer
(batched by aspect ratio, greedy CTC decode, NFC) → reading order (rows by centre height + side-by-side test, robust to
skewed pages) → page document → Gemma record. Speed on an RTX 3050 laptop: ~1.5 pages/s.

## Accuracy work (v0.2): error analysis and what fixed what

Measured on a human-verified reference (`hindi-ocr-eval --reference ref.json`; 128 Hindi-page lines from three deeds of
different scan quality, plus 96 English-page lines; handwriting, signatures and struck-through text excluded). The reference
lives outside the repo — it contains names from the deeds.

| pipeline | Hindi pages: CER / word acc / line exact | English pages: CER / word acc / line exact |
|---|---|---|
| v0.1 (stock PP-OCRv5 pipeline) | 8.77 % / 72.6 % / 46.1 % | 1.51 % / 88.2 % / 69.8 % |
| **v0.2 (this)** | **6.25 % / 81.8 % / 48.4 %** | **1.47 % / 95.1 % / 90.6 %** |

Recurring errors found (aligned diff against the reference, ~200 character errors):

| error | count | cause | fix |
|---|--:|---|---|
| dropped word spaces (`निवासीADV`, `RAJESHKUMAR`) | 33 | CTC model skips the space class at narrow gaps | **space recovery from geometry**: each emitted character's CTC time-step gives its x position; an empty column run ≥ 0.22 × line height between two characters is a word boundary (`recognizer.space_gap`) |
| loose crops | (whole-line) | model trained on tight PaddleOCR crops; every +5 % vertical padding cost ~1.5 CER points | `crop_pad 0.06 → 0.0`, `unclip_ratio 1.6 → 1.5` |
| very long lines (45+ chars) read worst | — | CTC accuracy degrades with aspect ratio | lines wider than 25 × height are split at word gaps and read in pieces (`recognizer.chunk_ratio`) |
| Devanagari digits inside Latin numbers (`5000०`, `18६57600`, a lone `१०१`) | 7 | glyph confusion | one digit system per token / line / page: the minority script follows the majority (`postprocess.digit_script_consistent`) |
| danda emitted as ASCII `\|` | 4 | charset | `\|` next to Devanagari → `।` (`postprocess.danda_consistent`) |
| English names / IDs read by the Devanagari model | ~36 Latin chars | one model for two scripts | pure-Latin lines get a second read from `en_PP-OCRv5_mobile_rec`; the more confident read wins (`recognizer.latin`) |
| table label and value split into separate entries | (structure) | line-level output | cells sharing a row are one `ocr_text` entry joined with ` \| ` (`gemma.join_rows`) |

Tried and rejected (no gain or worse on the reference): rendering at 150/250/300 dpi, larger crop padding, horizontal
padding, contrast stretch / CLAHE on crops, cubic / Lanczos upsampling, a confidence-gated second pass with a roomier
crop, and re-reading Latin *pieces* of mixed-script lines (2× slower, no measurable gain — `mixed_chunk_ratio: 0`).

What remains is genuinely the recognizer: dropped reph / `्र` conjuncts (वर्णित→वणित, ग्रामीण→गामीण), `क्ष` mangled,
anusvara dropped, `ु`↔`ू`, `/` swallowed between Devanagari words. Those need the retrained pack (Route A) — nothing in the
pipeline can invent the missing strokes without hard-coding words.

## Known limits of the bootstrap recognizer

Stock PP-OCRv5 Devanagari was trained on ~3.6k lines. On deeds it reads most printed Hindi correctly but makes
matra/conjunct slips (e.g. विक्रय → विक्रिय, ड्यूटी → इयूटी) and occasionally emits Devanagari digits for Latin ones.
Handwriting, stamps and signatures come out low-confidence (flagged, not dropped). These are the errors the retrained
pack is meant to remove.
