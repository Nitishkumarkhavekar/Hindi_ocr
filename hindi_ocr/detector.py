"""Text-line detection: PP-OCRv5 mobile DB detector (ONNX). Script-agnostic — it finds lines, the recognizer reads them.

detect(image) -> list of (polygon 4x2 float32 in page pixels, score), unordered. Post-processing follows PaddleOCR's
DBPostProcess (threshold -> contours -> min-area rectangle -> unclip by area/perimeter ratio)."""
import cv2
import numpy as np
import pyclipper
from shapely.geometry import Polygon

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


class Detector:
    def __init__(self, session, cfg):
        self.sess = session
        d = cfg["detector"]
        self.limit = int(d.get("limit_side_len", 1920))
        self.thresh = float(d.get("thresh", 0.3))
        self.box_thresh = float(d.get("box_thresh", 0.6))
        self.unclip_ratio = float(d.get("unclip_ratio", 1.5))
        self.max_candidates = int(d.get("max_candidates", 1000))
        self.min_size = float(d.get("min_box_size", 5))
        self.input_name = session.get_inputs()[0].name
        self.run_opts = None   # set by the pipeline (providers.run_options)

    # ---- preprocessing --------------------------------------------------------------------------------------------
    WIDTH_CLASSES = (0.5, 0.625, 0.75, 0.875, 1.0)   # short side padded to one of these fractions of the long side

    def _resize(self, img):
        """Scale so the long side is <= limit, then pad (white) onto one of a few fixed canvases so the GPU sees a handful
        of input shapes instead of one per page — dynamic shapes fragment the ONNX Runtime CUDA arena until it fails."""
        h, w = img.shape[:2]
        scale = min(1.0, self.limit / max(h, w))
        nh, nw = max(32, int(round(h * scale / 32)) * 32), max(32, int(round(w * scale / 32)) * 32)
        resized = cv2.resize(img, (nw, nh))
        long_side = max(nh, nw)
        canvas_long = self.limit if long_side > self.limit * 0.5 else max(32, int(round(long_side / 32)) * 32)
        short = min(nh, nw)
        canvas_short = next((int(round(f * canvas_long / 32)) * 32 for f in self.WIDTH_CLASSES if f * canvas_long >= short), canvas_long)
        ch, cw = (canvas_long, canvas_short) if nh >= nw else (canvas_short, canvas_long)
        canvas = np.full((ch, cw, 3), 255, np.uint8)
        canvas[:nh, :nw] = resized
        return canvas, nw / w, nh / h

    def _tensor(self, img):
        x = (img.astype(np.float32) / 255.0 - MEAN) / STD
        return np.ascontiguousarray(x.transpose(2, 0, 1)[None])

    # ---- post-processing (DBPostProcess) -----------------------------------------------------------------------------
    @staticmethod
    def _box_score(prob, box):
        h, w = prob.shape
        xs = np.clip([np.floor(box[:, 0].min()), np.ceil(box[:, 0].max())], 0, w - 1).astype(int)
        ys = np.clip([np.floor(box[:, 1].min()), np.ceil(box[:, 1].max())], 0, h - 1).astype(int)
        mask = np.zeros((ys[1] - ys[0] + 1, xs[1] - xs[0] + 1), np.uint8)
        cv2.fillPoly(mask, [(box - [xs[0], ys[0]]).round().astype(np.int32)], 1)
        return float(cv2.mean(prob[ys[0]:ys[1] + 1, xs[0]:xs[1] + 1], mask)[0])

    def _unclip(self, box):
        poly = Polygon(box)
        if poly.length == 0:
            return None
        distance = poly.area * self.unclip_ratio / poly.length
        pc = pyclipper.PyclipperOffset()
        pc.AddPath(box.round().astype(int).tolist(), pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        out = pc.Execute(distance)
        return np.array(out[0], np.float32) if out else None

    @staticmethod
    def _min_rect(contour):
        rect = cv2.minAreaRect(contour)
        pts = sorted(cv2.boxPoints(rect).tolist(), key=lambda p: p[0])
        left, right = sorted(pts[:2], key=lambda p: p[1]), sorted(pts[2:], key=lambda p: p[1])
        return np.array([left[0], right[0], right[1], left[1]], np.float32), min(rect[1])

    def _boxes(self, prob, sx, sy, w0, h0):
        bitmap = (prob > self.thresh).astype(np.uint8)
        contours, _ = cv2.findContours(bitmap, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        out = []
        for c in contours[:self.max_candidates]:
            box, side = self._min_rect(c)
            if side < self.min_size:
                continue
            score = self._box_score(prob, box)
            if score < self.box_thresh:
                continue
            ex = self._unclip(box)
            if ex is None:
                continue
            box, side = self._min_rect(ex.reshape(-1, 1, 2).round().astype(np.int32))
            if side < self.min_size + 2:
                continue
            box[:, 0] = np.clip(box[:, 0] / sx, 0, w0 - 1)
            box[:, 1] = np.clip(box[:, 1] / sy, 0, h0 - 1)
            out.append((box, score))
        return out

    def __call__(self, image):
        h0, w0 = image.shape[:2]
        resized, sx, sy = self._resize(image)
        prob = self.sess.run(None, {self.input_name: self._tensor(resized)}, self.run_opts)[0][0, 0]
        return self._boxes(prob, sx, sy, w0, h0)


def crop_line(image, box, pad=0.06, pad_x=0.0):
    """Perspective-crop a quadrilateral line box (4x2) into an upright strip; pad / pad_x add context around it as a
    fraction of the line height (Devanagari carries marks above the headline and below the baseline)."""
    w = int(max(np.linalg.norm(box[1] - box[0]), np.linalg.norm(box[2] - box[3])))
    h = int(max(np.linalg.norm(box[3] - box[0]), np.linalg.norm(box[2] - box[1])))
    w, h = max(w, 4), max(h, 4)
    ph, pw = int(round(h * pad)), int(round(h * pad_x))
    dst = np.array([[pw, ph], [w - 1 + pw, ph], [w - 1 + pw, h - 1 + ph], [pw, h - 1 + ph]], np.float32)
    M = cv2.getPerspectiveTransform(box.astype(np.float32), dst)
    crop = cv2.warpPerspective(image, M, (w + 2 * pw, h + 2 * ph), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    if crop.shape[0] >= crop.shape[1] * 1.5:   # vertical strip -> rotate upright
        crop = np.rot90(crop)
    return crop
