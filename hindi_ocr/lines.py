"""Line-level helpers between detection and recognition — all generic, nothing document-specific.

split_at_gaps(crop, max_ratio, ...)  Very wide lines (width/height above max_ratio) are split at whitespace columns into
                                     chunks the CTC recognizer handles well; the pieces are read separately and joined.
danda_consistent(text)               '|' adjacent to Devanagari -> danda '।' (the model has no other use for '|' there).
digit_script_consistent(text)        A token made only of digits and digit-like punctuation that mixes Latin and Devanagari
                                     digits is unified to the script of its majority ('5000०' -> '50000'). Only digit-only
                                     tokens are touched: a number is never allowed to be half in each script.
"""
import re
import cv2
import numpy as np

DEV_DIGITS = "०१२३४५६७८९"
LAT_DIGITS = "0123456789"
_DEV2LAT = str.maketrans(DEV_DIGITS, LAT_DIGITS)
_LAT2DEV = str.maketrans(LAT_DIGITS, DEV_DIGITS)
_NUM_TOKEN = re.compile(r"^[0-9०-९][0-9०-९.,:/\-]*$")


def split_at_gaps(crop, max_ratio=25.0, min_gap_frac=0.25, ink_thresh=0.02):
    """-> list of (x_offset, chunk) covering the crop left to right. A single [(0, crop)] when no split is needed or
    no usable gap exists. Gaps are runs of near-empty columns at least min_gap_frac * height wide."""
    h, w = crop.shape[:2]
    if w <= max_ratio * h:
        return [(0, crop)]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    ink = binary.mean(axis=0) / 255.0                       # fraction of ink per column
    empty = ink < ink_thresh
    min_gap = max(2, int(min_gap_frac * h))
    gaps = []                                               # (start, end) of empty runs
    start = None
    for x, e in enumerate(np.append(empty, False)):
        if e and start is None:
            start = x
        elif not e and start is not None:
            if x - start >= min_gap:
                gaps.append((start, x))
            start = None
    if not gaps:
        return [(0, crop)]
    target = max_ratio * h * 0.8                            # aim a little under the limit
    cuts, last = [], 0
    while w - last > max_ratio * h:
        candidates = [g for g in gaps if last + 0.3 * target < (g[0] + g[1]) / 2 <= last + target]
        if not candidates:
            break
        g = candidates[-1]                                  # the right-most gap inside the window keeps chunks big
        cut = (g[0] + g[1]) // 2
        cuts.append(cut)
        last = cut
    if not cuts:
        return [(0, crop)]
    edges = [0] + cuts + [w]
    return [(edges[i], crop[:, edges[i]:edges[i + 1]]) for i in range(len(edges) - 1) if edges[i + 1] - edges[i] > 2]


def digit_script_consistent(text):
    """Digit-only tokens: a token mixing Latin and Devanagari digits takes its majority script; then, if the line still
    holds digit tokens of both scripts, the minority script follows the majority (one line, one digit system)."""
    toks = text.split(" ")
    for k, tok in enumerate(toks):
        if _NUM_TOKEN.match(tok):
            dev = sum(c in DEV_DIGITS for c in tok)
            lat = sum(c in LAT_DIGITS for c in tok)
            if dev and lat:
                toks[k] = tok.translate(_DEV2LAT if lat >= dev else _LAT2DEV)
    dev = sum(sum(c in DEV_DIGITS for c in t) for t in toks if _NUM_TOKEN.match(t))
    lat = sum(sum(c in LAT_DIGITS for c in t) for t in toks if _NUM_TOKEN.match(t))
    if dev and lat:
        table = _DEV2LAT if lat >= dev else _LAT2DEV
        toks = [t.translate(table) if _NUM_TOKEN.match(t) else t for t in toks]
    return " ".join(toks)


_PIPE_AFTER_DEV = re.compile(r"(?<=[\u0900-\u097F\s])\|")
_PIPE_BEFORE_DEV = re.compile(r"\|(?=[\s\u0900-\u097F])")


def danda_consistent(text):
    """A vertical bar next to Devanagari text is the danda (U+0964) mis-emitted as ASCII '|'; only Devanagari context is touched."""
    if not any(0x0900 <= ord(c) <= 0x097F for c in text):
        return text
    text = _PIPE_AFTER_DEV.sub("\u0964", text)
    return _PIPE_BEFORE_DEV.sub("\u0964", text)


def page_digit_script_consistent(texts, dominance=0.8):
    """One page, one digit system: when >= `dominance` of a page's digits are in one script, digit-only tokens in the
    other script are converted (a lone '१०१' on a page of Latin numerals is a recognition slip, not a script switch)."""
    dev = sum(c in DEV_DIGITS for t in texts for c in t)
    lat = sum(c in LAT_DIGITS for t in texts for c in t)
    if not dev or not lat:
        return texts
    if lat / (dev + lat) >= dominance:
        table = _DEV2LAT
    elif dev / (dev + lat) >= dominance:
        table = _LAT2DEV
    else:
        return texts
    return [" ".join(t.translate(table) if _NUM_TOKEN.match(t) else t for t in text.split(" ")) for text in texts]
