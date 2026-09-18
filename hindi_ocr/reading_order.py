"""Reading order for detected lines: top-to-bottom, left-to-right within a row. A row is a set of boxes whose centres
sit at the same height *and* that lie side by side; consecutive lines on a skewed page have overlapping bounding boxes
but overlapping x-ranges, so they are kept as separate rows. Adequate for deed pages (single column with occasional
side-by-side fields)."""
import numpy as np


def _geometry(box):
    """4x2 polygon -> (cx, cy, x0, x1, text height): height from the polygon's short sides, not the axis-aligned bbox."""
    h = 0.5 * (np.linalg.norm(box[3] - box[0]) + np.linalg.norm(box[2] - box[1]))
    return float(box[:, 0].mean()), float(box[:, 1].mean()), float(box[:, 0].min()), float(box[:, 0].max()), max(float(h), 1.0)


def order_lines(boxes):
    """boxes: list of 4x2 polygons -> list of indices in reading order."""
    return [i for row in order_rows(boxes) for i in row]


def order_rows(boxes):
    """boxes: list of 4x2 polygons -> rows top-to-bottom, each a list of indices left-to-right (table cells / side-by-side
    fields share a row)."""
    if not boxes:
        return []
    g = [_geometry(np.asarray(b, np.float32)) for b in boxes]
    idx = sorted(range(len(boxes)), key=lambda i: g[i][1])
    rows = []   # each row: [indices], with running centre y and height
    for i in idx:
        cx, cy, x0, x1, h = g[i]
        placed = False
        for row in rows[-3:]:                      # only the last few rows can still be at this height
            same_height = abs(cy - row["cy"]) < 0.5 * min(h, row["h"])
            x_overlap = min(x1, row["x1"]) - max(x0, row["x0"])
            side_by_side = x_overlap < 0.2 * min(x1 - x0, row["x1"] - row["x0"])
            if same_height and side_by_side:
                row["items"].append(i)
                n = len(row["items"])
                row["cy"] = (row["cy"] * (n - 1) + cy) / n
                row["h"] = min(row["h"], h)
                row["x0"], row["x1"] = min(row["x0"], x0), max(row["x1"], x1)
                placed = True
                break
        if not placed:
            rows.append({"items": [i], "cy": cy, "h": h, "x0": x0, "x1": x1})
    rows.sort(key=lambda r: r["cy"])
    return [sorted(r["items"], key=lambda i: g[i][2]) for r in rows]
