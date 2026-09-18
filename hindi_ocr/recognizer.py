"""Text-line recognizers behind one interface, so the model can be swapped without touching the pipeline:

    rec = build_recognizer(cfg, session)     # by cfg["recognizer"]["backend"]
    rec(crops) -> [(text, confidence, min_char_conf), ...]
    rec.ident  -> {"backend", "model_md5", "charset_md5", "classes"}  (stamped into every output)

Backends:
  paddle_devanagari  PP-OCRv5 mobile Devanagari (48 px, CTC; blank=0, charset, space=last). The bootstrap model.
  arjuna_hindi       placeholder for the retrained SVTRv2 KN+HI+EN pack (32 px, CTC) — register when the ONNX exists.
"""
import hashlib
import json
import unicodedata
import cv2
import numpy as np

BACKENDS = {}


def register(name):
    def deco(cls):
        BACKENDS[name] = cls
        return cls
    return deco


def md5_file(path, n=12):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:n]


class CTCRecognizer:
    """Shared batching + greedy CTC decode. Subclasses set height, normalisation and the class layout."""
    height = 48
    blank = 0

    def __init__(self, session, cfg, charset, model_path=None, backend=None):
        self.sess = session
        r = cfg["recognizer"]
        self.bs = int(r.get("batch", 32))
        self.max_width = -(-int(r.get("max_width", 3200)) // 128) * 128
        self.enhance = r.get("enhance") or "none"          # none | stretch | clahe | gray : crop preprocessing before resize
        self.space_gap = float(r.get("space_gap", 0) or 0)  # recover dropped spaces: gap (fraction of height) between chars that means a word boundary
        self.space_gap_latin = float(r.get("space_gap_latin", self.space_gap) or self.space_gap)  # stricter gap when both neighbours are Latin/digits (no headline joins them)
        self.interp = {"linear": cv2.INTER_LINEAR, "cubic": cv2.INTER_CUBIC, "lanczos": cv2.INTER_LANCZOS4}[r.get("upsample", "linear")]
        self.chars = list(charset)
        self.input_name = session.get_inputs()[0].name
        self.run_opts = None   # set by the pipeline (providers.run_options)
        self.ident = {"backend": backend or r.get("backend"), "model_md5": md5_file(model_path or cfg.path("models", "rec")),
                      "charset_md5": hashlib.md5("".join(self.chars).encode("utf-8")).hexdigest()[:12], "classes": self.num_classes}

    # ---- to override ------------------------------------------------------------------------------------------------
    @property
    def num_classes(self):
        return len(self.chars) + 2   # blank + charset + space

    def label(self, idx):
        """class index -> character (None for blank)."""
        if idx == self.blank:
            return None
        if idx == len(self.chars) + 1:
            return " "
        return self.chars[idx - 1]

    def normalise(self, crop):
        x = crop.astype(np.float32) / 255.0
        return (x - 0.5) / 0.5

    # ---- shared ----------------------------------------------------------------------------------------------------
    def _enhance(self, crop):
        if self.enhance == "none":
            return crop
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        if self.enhance == "stretch":                       # per-crop contrast stretch between the 2nd and 98th percentile
            lo, hi = np.percentile(gray, (2, 98))
            gray = np.clip((gray.astype(np.float32) - lo) * (255.0 / max(hi - lo, 1)), 0, 255).astype(np.uint8)
        elif self.enhance == "clahe":
            gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(2, 8)).apply(gray)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    def _prepare(self, crop):
        """-> (tensor CHW, ink profile per column of the resized crop or None)."""
        crop = self._enhance(crop)
        h, w = crop.shape[:2]
        nw = min(self.max_width, max(8, int(round(w * self.height / h))))
        img = cv2.resize(crop, (nw, self.height), interpolation=self.interp if nw > w else cv2.INTER_AREA)
        profile = None
        if self.space_gap:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
            profile = binary.mean(axis=0) / 255.0
        return self.normalise(img).transpose(2, 0, 1), profile

    def _recover_spaces(self, chars, steps, T, W, profile):
        """Insert a space between consecutive characters when the columns between their CTC positions contain an empty
        run at least space_gap * height wide. Devanagari words hang from a continuous headline, so a real gap is a word
        boundary; Latin inter-letter gaps are far below the threshold."""
        if profile is None or len(chars) < 2:
            return chars
        empty = profile < 0.02
        out = [chars[0]]
        for k in range(1, len(chars)):
            if chars[k] != " " and chars[k - 1] != " ":
                latin = all(c.isascii() and c.isalnum() for c in (chars[k - 1], chars[k]))
                min_gap = max(2, int(round((self.space_gap_latin if latin else self.space_gap) * self.height)))
                x0 = int((steps[k - 1] + 0.5) * W / T)
                x1 = int((steps[k] + 0.5) * W / T)
                if x1 - x0 > min_gap:
                    seg = empty[max(x0, 0):min(x1, len(empty))]
                    run = best = 0
                    for e in seg:
                        run = run + 1 if e else 0
                        best = max(best, run)
                    if best >= min_gap:
                        out.append(" ")
            out.append(chars[k])
        return out

    def _decode(self, probs, width=None, profile=None):
        ids = probs.argmax(-1)
        out, prev, confs, steps = [], -1, [], []
        for t, i in enumerate(ids):
            if i != prev and i != self.blank:
                ch = self.label(int(i))
                if ch is not None:
                    out.append(ch)
                    confs.append(float(probs[t, i]))
                    steps.append(t)
            prev = i
        if self.space_gap and width:
            out = self._recover_spaces(out, steps, len(ids), width, profile)
        text = unicodedata.normalize("NFC", "".join(out)).strip()
        if not confs:
            return "", 0.0, 0.0
        return text, float(np.mean(confs)), float(np.min(confs))

    def __call__(self, crops, bs=None):
        bs = bs or self.bs
        if not crops:
            return []
        order = sorted(range(len(crops)), key=lambda i: crops[i].shape[1] / max(crops[i].shape[0], 1))   # batch similar widths
        results = [None] * len(crops)
        for s in range(0, len(order), bs):
            idx = order[s:s + bs]
            prepared = [self._prepare(crops[i]) for i in idx]
            tensors = [t for t, _ in prepared]
            W = min(self.max_width, -(-max(t.shape[2] for t in tensors) // 128) * 128)   # width buckets of 128 px, fixed batch size:
            batch = np.zeros((bs, 3, self.height, W), np.float32)                          # few distinct shapes -> no arena fragmentation
            for k, t in enumerate(tensors):
                batch[k, :, :, :t.shape[2]] = t
            probs = self.sess.run(None, {self.input_name: batch}, self.run_opts)[0]
            for k, i in enumerate(idx):
                results[i] = self._decode(probs[k], W, prepared[k][1])
        return results


@register("paddle_devanagari")
class PaddleDevanagari(CTCRecognizer):
    """PP-OCRv5 mobile Devanagari: input [b,3,48,W] BGR normalised to [-1,1]; output [b,T,570] softmax."""
    height = 48


@register("arjuna_hindi")
class ArjunaHindi(CTCRecognizer):
    """Slot for the retrained SVTRv2-B KN+HI+EN pack (32 px input, CTC, no trailing space class). Point
    models.rec / models.charset at the exported ONNX + charset.json and set recognizer.backend: arjuna_hindi."""
    height = 32

    @property
    def num_classes(self):
        return len(self.chars) + 1

    def label(self, idx):
        return None if idx == self.blank else self.chars[idx - 1]


@register("paddle_latin")
class PaddleLatin(CTCRecognizer):
    """PP-OCRv5 mobile English / Latin recognizer (48 px, CTC, blank + charset + space) - the optional secondary model
    that re-reads Latin-script and digit segments (recognizer.latin in the yaml)."""
    height = 48


def build_recognizer(cfg, session, name=None, model_path=None, charset_path=None):
    name = name or cfg["recognizer"].get("backend", "paddle_devanagari")
    if name not in BACKENDS:
        raise ValueError(f"unknown recognizer backend {name!r}; registered: {list(BACKENDS)}")
    with open(charset_path or cfg.path("models", "charset"), encoding="utf-8") as f:
        charset = json.load(f)
    rec = BACKENDS[name](session, cfg, charset, model_path=model_path, backend=name)
    n_out = session.get_outputs()[0].shape[-1]
    if isinstance(n_out, int) and n_out != rec.num_classes:
        raise ValueError(f"recognizer {name}: model has {n_out} classes but charset implies {rec.num_classes}")
    return rec
