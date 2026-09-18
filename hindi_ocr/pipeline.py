"""HindiOCR: detector -> line crops -> recognizer -> reading order -> page document / Gemma record.

    ocr = HindiOCR()                                # loads hindi_ocr.yaml, both ONNX models
    doc = ocr.page("scan.png")                      # one hindi-ocr/1.0 page document
    pages = ocr.pages("deed.pdf")                   # list of page documents
    deeds = ocr.documents(["a.pdf", "b.pdf"])       # [(name, [page documents])]
    rec = ocr.gemma("deed.pdf")                     # {"prompt", "ocr_text": {"Page N": [...]}, "output"}
    ocr.ready()                                     # providers per model, model hashes, fallbacks
"""
import time
from pathlib import Path
import numpy as np
from .config import Config
from .providers import make_session, run_options
from .detector import Detector, crop_line
from .recognizer import build_recognizer, md5_file
from .reading_order import order_rows
from .io import to_bgr, expand_inputs, iter_sources, is_pdf_bytes, pdf_pages
from .lines import split_at_gaps, digit_script_consistent, danda_consistent, page_digit_script_consistent
from . import schema

__version__ = "0.2.0"


class HindiOCR:
    name = "Hindi-OCR"
    version = __version__

    def __init__(self, config=None, providers=None, **overrides):
        t0 = time.time()
        self.cfg = Config(config, **overrides)
        policy = providers or self.cfg["provider"].get("policy", "auto")
        mem = self.cfg["provider"].get("cuda_mem_limit_gb", 2)
        det_sess, self.det_info = make_session(self.cfg.path("models", "det"), policy, mem)
        rec_sess, self.rec_info = make_session(self.cfg.path("models", "rec"), policy, mem)
        self.det = Detector(det_sess, self.cfg)
        self.rec = build_recognizer(self.cfg, rec_sess)
        self.det.run_opts, self.rec.run_opts = run_options(self.det_info), run_options(self.rec_info)
        lat = self.cfg["recognizer"].get("latin") or {}
        self.latin = None
        if lat.get("model"):                                       # optional secondary recognizer for Latin script / digits
            lat_sess, self.latin_info = make_session(self.cfg.root / lat["model"], policy, mem)
            self.latin = build_recognizer(self.cfg, lat_sess, name=lat.get("backend", "paddle_latin"), model_path=self.cfg.root / lat["model"], charset_path=self.cfg.root / lat["charset"])
            self.latin.run_opts = run_options(self.latin_info)
            self.latin_chunk_ratio = float(lat.get("mixed_chunk_ratio", 8))
            self.latin_margin = float(lat.get("margin", 0.0))
        self.model = {"det": "pp-ocrv5-mobile-det", "det_md5": md5_file(self.cfg.path("models", "det")), **{f"rec_{k}": v for k, v in self.rec.ident.items()},
                      "pipeline": __version__}
        if self.latin:
            self.model.update({"latin_backend": self.latin.ident["backend"], "latin_model_md5": self.latin.ident["model_md5"]})
        self.thresholds = self.cfg["flags"]
        r = self.cfg["recognizer"]
        self.drop = float(r.get("drop_score", 0.0))
        self.crop_pad = float(r.get("crop_pad", 0.06))
        self.crop_pad_x = float(r.get("crop_pad_x", 0.0))
        self.chunk_ratio = float(r.get("chunk_ratio", 0) or 0)            # 0 = never split long lines
        sp = r.get("second_pass") or {}
        self.second_pass_conf = float(sp.get("conf", 0)) if sp else 0.0   # 0 = off
        self.second_pass_pad = float(sp.get("pad", 0.15)) if sp else 0.15
        self.digit_script = bool(self.cfg["postprocess"].get("digit_script_consistent", False))
        self.danda = bool(self.cfg["postprocess"].get("danda_consistent", False))
        self.load_s = round(time.time() - t0, 2)
        self.last_stats = {}

    def ready(self):
        prov = {"det": self.det_info, "rec": self.rec_info}
        if self.latin:
            prov["latin"] = self.latin_info
        return {"name": self.name, "version": self.version, "load_s": self.load_s, "model": self.model, "providers": prov}

    # ---- one page ---------------------------------------------------------------------------------------------------
    def page(self, image, index=0, file=None):
        img = to_bgr(image)
        h, w = img.shape[:2]
        t0 = time.perf_counter()
        boxes = self.det(img)
        t1 = time.perf_counter()
        crops = [crop_line(img, b, pad=self.crop_pad, pad_x=self.crop_pad_x) for b, _ in boxes]
        texts = self._recognize(crops)
        if self.latin:
            texts = self._route_latin(crops, texts)
        if self.second_pass_conf > 0:                                       # re-read the weakest lines with a roomier crop; keep the better read
            weak = [i for i, (t, c, _) in enumerate(texts) if c < self.second_pass_conf]
            if weak:
                alt = self._recognize([crop_line(img, boxes[i][0], pad=self.second_pass_pad, pad_x=self.crop_pad_x + 0.02) for i in weak])
                for i, r in zip(weak, alt):
                    if r[1] > texts[i][1] + 0.02:
                        texts[i] = r
        if self.digit_script:
            texts = [(digit_script_consistent(t), c, m) for t, c, m in texts]
            fixed = page_digit_script_consistent([t for t, _, _ in texts])
            texts = [(t2, c, m) for t2, (_, c, m) in zip(fixed, texts)]
        if self.danda:
            texts = [(danda_consistent(t), c, m) for t, c, m in texts]
        t2 = time.perf_counter()
        rows = order_rows([b for b, _ in boxes])
        lines = []
        k = 0
        for r, row in enumerate(rows):
            for i in row:
                box, det_score = boxes[i]
                text, conf, min_conf = texts[i]
                if not text or conf < self.drop:
                    continue
                lines.append(self._line(k, r, box, det_score, text, conf, min_conf))
                k += 1
        label = file or (Path(image).name if isinstance(image, (str, Path)) else None)
        doc = schema.page_document(index, w, h, label, lines, self.model)
        doc["timing_ms"] = {"det": round((t1 - t0) * 1000, 1), "rec": round((t2 - t1) * 1000, 1), "lines": len(lines)}
        return doc

    def _line(self, k, row, box, det_score, text, conf, min_conf):
        return {"id": f"l{k}", "row": row, "text": text, "confidence": round(conf, 4), "min_conf": round(min_conf, 4),
                "bbox": [round(float(box[:, 0].min()), 1), round(float(box[:, 1].min()), 1), round(float(box[:, 0].max()), 1), round(float(box[:, 1].max()), 1)],
                "polygon": [[round(float(x), 1), round(float(y), 1)] for x, y in box], "det_score": round(float(det_score), 3),
                "script": schema.script_of(text), "flags": schema.line_flags(conf, min_conf, text, self.thresholds)}

    def _recognize(self, crops):
        """rec over crops, splitting very wide ones at word gaps when chunk_ratio is set; pieces are re-joined in order."""
        if not self.chunk_ratio:
            return self.rec(crops)
        pieces, owner = [], []
        for i, c in enumerate(crops):
            for _, chunk in split_at_gaps(c, self.chunk_ratio):
                pieces.append(chunk)
                owner.append(i)
        res = self.rec(pieces)
        out = [None] * len(crops)
        for i in range(len(crops)):
            parts = [res[k] for k in range(len(res)) if owner[k] == i]
            text = " ".join(t for t, _, _ in parts if t).strip()
            n = [len(t) for t, _, _ in parts if t]
            conf = sum(c * len(t) for t, c, _ in parts if t) / max(sum(n), 1) if n else 0.0
            out[i] = (text, conf, min([m for t, _, m in parts if t] or [0.0]))
        return out

    @staticmethod
    def _script_mix(text):
        """-> (devanagari letters, latin letters + digits) counts."""
        dev = sum(0x0900 <= ord(c) <= 0x097F for c in text)
        lat = sum(c.isascii() and c.isalnum() for c in text)
        return dev, lat

    def _better(self, a, b):
        """Pick between two (text, conf, min_conf) reads; b (the Latin model) wins on confidence + margin."""
        if not b[0]:
            return a
        if not a[0] or b[1] >= a[1] + self.latin_margin:
            return b
        return a

    def _route_latin(self, crops, texts):
        """Lines the Devanagari model reads as Latin/digits get a second read from the Latin model; mixed-script lines are
        split at word gaps and each Latin-dominant piece is re-read. The higher-confidence read wins per piece."""
        out = list(texts)
        pure, mixed = [], []
        for i, (t, c, m) in enumerate(texts):
            dev, lat = self._script_mix(t)
            if lat and not dev:
                pure.append(i)
            elif lat >= 2 and dev:
                mixed.append(i)
        if pure:
            alt = self.latin([crops[i] for i in pure])
            for i, r in zip(pure, alt):
                out[i] = self._better(texts[i], r)
        for i in (mixed if self.latin_chunk_ratio > 0 else []):
            pieces = split_at_gaps(crops[i], self.latin_chunk_ratio)
            if len(pieces) < 2:
                continue
            chunks = [c for _, c in pieces]
            dev_reads = self.rec(chunks)
            latin_idx = [k for k, (t, _, _) in enumerate(dev_reads) if t and self._script_mix(t)[1] > self._script_mix(t)[0]]
            if latin_idx:
                lat_reads = self.latin([chunks[k] for k in latin_idx])
                for k, r in zip(latin_idx, lat_reads):
                    dev_reads[k] = self._better(dev_reads[k], r)
            parts = [r for r in dev_reads if r[0]]
            if not parts:
                continue
            n = sum(len(t) for t, _, _ in parts)
            joined = (" ".join(t for t, _, _ in parts), sum(c * len(t) for t, c, _ in parts) / n, min(m for _, _, m in parts))
            if joined[1] >= texts[i][1] - 0.05:                     # keep the chunked read unless it is clearly less confident
                out[i] = joined
        return out

    # ---- many pages -------------------------------------------------------------------------------------------------
    def _sources(self, inputs, dpi):
        if isinstance(inputs, (str, Path)) or (isinstance(inputs, (list, tuple)) and inputs and all(isinstance(x, (str, Path)) for x in inputs)):
            return iter_sources(expand_inputs(inputs), dpi=dpi)
        items = list(inputs) if isinstance(inputs, (list, tuple)) else [inputs]

        def gen():
            for k, x in enumerate(items):
                if is_pdf_bytes(x):
                    for i, im in enumerate(pdf_pages(x, dpi=dpi)):
                        yield f"upload{k + 1}.pdf#{i + 1}", im
                else:
                    yield getattr(x, "name", None) or f"upload{k + 1}", x
        return gen()

    def stream(self, inputs, pdf_dpi=None):
        """Yield page documents in input order."""
        dpi = pdf_dpi or self.cfg["pdf"].get("dpi", 200)
        t0 = time.time()
        n = 0
        det_ms = rec_ms = 0.0
        for label, src in self._sources(inputs, dpi):
            doc = self.page(src, index=n, file=label)
            n += 1
            det_ms += doc["timing_ms"]["det"]
            rec_ms += doc["timing_ms"]["rec"]
            wall = time.time() - t0
            self.last_stats = {"pages": n, "wall_s": round(wall, 2), "pages_per_s": round(n / max(wall, 1e-9), 2),
                               "ms_per_page": {"det": round(det_ms / n, 1), "rec": round(rec_ms / n, 1)}}
            yield doc

    def pages(self, inputs, pdf_dpi=None):
        return list(self.stream(inputs, pdf_dpi))

    def documents(self, inputs, pdf_dpi=None):
        """Group page documents by source file: [(name, [pages])] in input order."""
        groups, order = {}, []
        for doc in self.stream(inputs, pdf_dpi):
            base = (doc["page"]["file"] or f"page_{doc['page']['index']}").partition("#")[0]
            if base not in groups:
                groups[base] = []
                order.append(base)
            groups[base].append(doc)
        return [(b, groups[b]) for b in order]

    def gemma(self, source, pdf_dpi=None, output=None):
        """One deed (PDF / image / list of page images) -> the Gemma extraction record."""
        pages = self.pages(source, pdf_dpi)
        return schema.gemma_record(pages, output=output, join_rows=self.cfg["gemma"].get("join_rows", True))
