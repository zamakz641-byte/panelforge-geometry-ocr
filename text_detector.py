from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Sequence, Tuple
import hashlib
import os
import tempfile
import urllib.request

import cv2
import numpy as np

from image_io import imread_unicode


MODEL_SHA256 = "d2a7720d45a54257208b1e13e36a8479894cb74155a5efe29462512d42f49da9"
MODEL_URLS = (
    "https://huggingface.co/SWHL/RapidOCR/resolve/main/PP-OCRv4/ch_PP-OCRv4_det_infer.onnx?download=true",
    "https://www.modelscope.cn/models/RapidAI/RapidOCR/resolve/v3.9.2/onnx/PP-OCRv4/det/ch_PP-OCRv4_det_mobile.onnx",
)


@dataclass
class TextDetectorConfig:
    analysis_width: int = 768
    tile_size: int = 768
    tile_overlap: int = 96
    bitmap_threshold: float = 0.28
    box_threshold: float = 0.52
    min_side: int = 4
    min_area: int = 12
    expand_ratio: float = 1.34
    max_candidates: int = 1200


def _safe_progress(cb: Callable[[Dict[str, Any]], None] | None, **payload):
    if cb:
        try:
            cb(payload)
        except Exception:
            pass


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_text_model(app_dir: Path, progress=None) -> Path:
    """Download the small PP-OCRv4 text detector on first OCR use.

    The model is kept beside the app so subsequent runs are fully offline.
    """
    model_dir = app_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "ch_PP-OCRv4_det_mobile.onnx"

    if model_path.exists() and _sha256(model_path) == MODEL_SHA256:
        return model_path
    if model_path.exists():
        try:
            model_path.unlink()
        except OSError:
            pass

    _safe_progress(progress, stage="ocr-model", progress=0.35,
                   message="Premier usage OCR : téléchargement du détecteur de texte (~4,8 Mo)…")

    last_error: Exception | None = None
    for url in MODEL_URLS:
        tmp_path: Path | None = None
        try:
            fd, tmp_name = tempfile.mkstemp(prefix="panelforge_ocr_", suffix=".onnx", dir=str(model_dir))
            os.close(fd)
            tmp_path = Path(tmp_name)
            req = urllib.request.Request(url, headers={"User-Agent": "PanelForge/1.3"})
            with urllib.request.urlopen(req, timeout=60) as src, open(tmp_path, "wb") as dst:
                while True:
                    chunk = src.read(1024 * 1024)
                    if not chunk:
                        break
                    dst.write(chunk)
            if _sha256(tmp_path) != MODEL_SHA256:
                raise RuntimeError("Le modèle OCR téléchargé est invalide (SHA-256 incorrect).")
            tmp_path.replace(model_path)
            return model_path
        except Exception as exc:
            last_error = exc
            if tmp_path and tmp_path.exists():
                try:
                    tmp_path.unlink()
                except OSError:
                    pass

    raise RuntimeError(
        "Impossible de télécharger le modèle de détection OCR. "
        "Vérifie la connexion Internet puis relance, ou décoche l'option OCR. "
        f"Détail: {last_error}"
    )


def _resize_for_ocr(image: np.ndarray, width: int) -> Tuple[np.ndarray, float]:
    h, w = image.shape[:2]
    if w <= 0 or h <= 0:
        raise ValueError("Image OCR invalide")
    scale = width / float(w)
    out_h = max(1, int(round(h * scale)))
    interp = cv2.INTER_CUBIC if scale > 1.0 else cv2.INTER_AREA
    return cv2.resize(image, (width, out_h), interpolation=interp), scale


def _preprocess(tile: np.ndarray) -> np.ndarray:
    # PaddleOCR detection preprocessing: RGB, ImageNet-style mean/std.
    rgb = cv2.cvtColor(tile, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)
    rgb = (rgb - mean) / std
    return np.transpose(rgb, (2, 0, 1))[None, ...].astype(np.float32, copy=False)


def _box_score(prob: np.ndarray, contour: np.ndarray) -> float:
    x, y, w, h = cv2.boundingRect(contour)
    if w <= 0 or h <= 0:
        return 0.0
    x0 = max(0, x); y0 = max(0, y)
    x1 = min(prob.shape[1], x + w); y1 = min(prob.shape[0], y + h)
    if x1 <= x0 or y1 <= y0:
        return 0.0
    local = contour.copy().astype(np.int32)
    local[:, 0, 0] -= x0
    local[:, 0, 1] -= y0
    mask = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
    cv2.fillPoly(mask, [local], 1)
    vals = prob[y0:y1, x0:x1][mask.astype(bool)]
    return float(vals.mean()) if vals.size else 0.0


def _decode_probability_map(prob: np.ndarray, cfg: TextDetectorConfig,
                            tile_w: int, tile_h: int) -> List[Dict[str, Any]]:
    if prob.ndim != 2:
        prob = np.squeeze(prob)
    if prob.ndim != 2:
        raise RuntimeError(f"Sortie OCR inattendue: {prob.shape}")

    bitmap = (prob >= cfg.bitmap_threshold).astype(np.uint8) * 255
    bitmap = cv2.morphologyEx(bitmap, cv2.MORPH_CLOSE, np.ones((2, 2), np.uint8))
    contours, _ = cv2.findContours(bitmap, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) > cfg.max_candidates:
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:cfg.max_candidates]

    sy = tile_h / float(prob.shape[0])
    sx = tile_w / float(prob.shape[1])
    boxes: List[Dict[str, Any]] = []
    for cnt in contours:
        if cv2.contourArea(cnt) < cfg.min_area:
            continue
        score = _box_score(prob, cnt)
        if score < cfg.box_threshold:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        if w < cfg.min_side or h < cfg.min_side:
            continue

        x1 = x * sx; y1 = y * sy
        x2 = (x + w) * sx - 1; y2 = (y + h) * sy - 1
        bw = max(1.0, x2 - x1 + 1.0); bh = max(1.0, y2 - y1 + 1.0)
        pad_x = max(2.0, bw * (cfg.expand_ratio - 1.0) * 0.5)
        pad_y = max(2.0, bh * (cfg.expand_ratio - 1.0) * 0.5)
        x1 = max(0.0, x1 - pad_x); y1 = max(0.0, y1 - pad_y)
        x2 = min(tile_w - 1.0, x2 + pad_x); y2 = min(tile_h - 1.0, y2 + pad_y)
        boxes.append({
            "x1": int(round(x1)), "y1": int(round(y1)),
            "x2": int(round(x2)), "y2": int(round(y2)),
            "score": score,
        })
    return boxes


def _iou_or_containment(a: Dict[str, Any], b: Dict[str, Any]) -> Tuple[float, float]:
    x1 = max(a["x1"], b["x1"]); y1 = max(a["y1"], b["y1"])
    x2 = min(a["x2"], b["x2"]); y2 = min(a["y2"], b["y2"])
    if x2 < x1 or y2 < y1:
        return 0.0, 0.0
    inter = (x2 - x1 + 1) * (y2 - y1 + 1)
    aa = max(1, (a["x2"] - a["x1"] + 1) * (a["y2"] - a["y1"] + 1))
    ab = max(1, (b["x2"] - b["x1"] + 1) * (b["y2"] - b["y1"] + 1))
    return inter / float(aa + ab - inter), inter / float(min(aa, ab))


def _dedupe_boxes(boxes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for b in sorted(boxes, key=lambda x: x.get("score", 0.0), reverse=True):
        duplicate = False
        for k in out:
            iou, contain = _iou_or_containment(b, k)
            if iou >= 0.45 or contain >= 0.82:
                duplicate = True
                break
        if not duplicate:
            out.append(b)
    return sorted(out, key=lambda x: (x["y1"], x["x1"]))


class PPOCRTextDetector:
    def __init__(self, model_path: Path, cfg: TextDetectorConfig | None = None):
        self.cfg = cfg or TextDetectorConfig()
        self.net = cv2.dnn.readNetFromONNX(str(model_path))
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    def detect(self, image_bgr: np.ndarray) -> List[Dict[str, Any]]:
        cfg = self.cfg
        scaled, scale = _resize_for_ocr(image_bgr, cfg.analysis_width)
        sh, sw = scaled.shape[:2]
        tile_size = max(320, int(cfg.tile_size))
        overlap = max(0, min(tile_size // 3, int(cfg.tile_overlap)))
        step = max(64, tile_size - overlap)
        found: List[Dict[str, Any]] = []

        y0 = 0
        tile_index = 0
        while y0 < sh:
            tile_index += 1
            y1 = min(sh, y0 + tile_size)
            chunk = scaled[y0:y1, :]
            canvas = np.full((tile_size, tile_size, 3), 255, dtype=np.uint8)
            copy_h = min(tile_size, chunk.shape[0])
            copy_w = min(tile_size, chunk.shape[1])
            canvas[:copy_h, :copy_w] = chunk[:copy_h, :copy_w]

            self.net.setInput(_preprocess(canvas))
            pred = self.net.forward()
            if pred.ndim == 4:
                prob = pred[0, 0]
            elif pred.ndim == 3:
                prob = pred[0]
            else:
                prob = pred
            boxes = _decode_probability_map(prob, cfg, tile_size, tile_size)

            keep_from = 0 if y0 == 0 else overlap // 2
            keep_to = copy_h if y1 >= sh else max(keep_from + 1, copy_h - overlap // 2)
            for b in boxes:
                cy = (b["y1"] + b["y2"]) * 0.5
                if cy < keep_from or cy >= keep_to or b["x1"] >= sw or b["y1"] >= copy_h:
                    continue
                bx1 = max(0, min(sw - 1, b["x1"]))
                bx2 = max(0, min(sw - 1, b["x2"]))
                by1 = max(0, min(sh - 1, y0 + b["y1"]))
                by2 = max(0, min(sh - 1, y0 + min(copy_h - 1, b["y2"])))
                if bx2 < bx1 or by2 < by1:
                    continue
                found.append({
                    "x1": int(round(bx1 / scale)),
                    "y1": int(round(by1 / scale)),
                    "x2": int(round(bx2 / scale)),
                    "y2": int(round(by2 / scale)),
                    "score": float(b["score"]),
                })

            if y1 >= sh:
                break
            y0 += step

        return _dedupe_boxes(found)


def detect_text_regions(meta: Sequence[Any], app_dir: Path, ocr_width: int = 768,
                        progress=None) -> List[Dict[str, Any]]:
    model_path = ensure_text_model(app_dir, progress=progress)
    cfg = TextDetectorConfig(analysis_width=max(512, min(1152, int(ocr_width))))
    cfg.tile_size = cfg.analysis_width
    cfg.tile_overlap = max(64, int(round(cfg.tile_size * 0.125)))
    detector = PPOCRTextDetector(model_path, cfg)

    all_boxes: List[Dict[str, Any]] = []
    n = max(1, len(meta))
    for i, m in enumerate(meta, 1):
        img = imread_unicode(m.path, cv2.IMREAD_COLOR)
        if img is None:
            continue
        boxes = detector.detect(img)
        sx = m.analysis_w / float(m.orig_w)
        sy = m.analysis_h / float(m.orig_h)
        for b in boxes:
            ax1 = int(round(b["x1"] * sx)); ax2 = int(round(b["x2"] * sx))
            ay1 = m.analysis_y0 + int(round(b["y1"] * sy))
            ay2 = m.analysis_y0 + int(round(b["y2"] * sy))
            all_boxes.append({
                "x1": max(0, min(m.analysis_w - 1, ax1)),
                "y1": max(m.analysis_y0, min(m.analysis_y1, ay1)),
                "x2": max(0, min(m.analysis_w - 1, ax2)),
                "y2": max(m.analysis_y0, min(m.analysis_y1, ay2)),
                "score": float(b.get("score", 0.0)),
                "source_file": m.display_name,
            })
        _safe_progress(progress, stage="ocr", progress=0.36 + 0.20 * (i / n),
                       message=f"Détection texte OCR {i}/{len(meta)} • {len(all_boxes)} zones")
    return _dedupe_boxes(all_boxes)


def _intersection(a: Dict[str, Any], b: Dict[str, Any]) -> Tuple[int, int, int]:
    x1 = max(a["x1"], b["x1"]); y1 = max(a["y1"], b["y1"])
    x2 = min(a["x2"], b["x2"]); y2 = min(a["y2"], b["y2"])
    if x2 < x1 or y2 < y1:
        return 0, 0, 0
    return (x2 - x1 + 1) * (y2 - y1 + 1), x2 - x1 + 1, y2 - y1 + 1


def _vertical_gap(a: Dict[str, Any], b: Dict[str, Any]) -> int:
    if a["y2"] < b["y1"]:
        return b["y1"] - a["y2"] - 1
    if b["y2"] < a["y1"]:
        return a["y1"] - b["y2"] - 1
    return 0


def _box_intersection_area(a: Dict[str, Any], b: Dict[str, Any]) -> int:
    x1 = max(int(a["x1"]), int(b["x1"])); y1 = max(int(a["y1"]), int(b["y1"]))
    x2 = min(int(a["x2"]), int(b["x2"])); y2 = min(int(a["y2"]), int(b["y2"]))
    if x2 < x1 or y2 < y1:
        return 0
    return (x2 - x1 + 1) * (y2 - y1 + 1)


def filter_text_only_panels(canvas_bgr: np.ndarray, panels: Sequence[Dict[str, Any]],
                            text_boxes: Sequence[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Remove geometry candidates that are really only standalone typography.

    Geometry is intentionally generous, because it must not miss low-detail manga panels.
    Once OCR exists we can do a safer second pass: mask OCR text inside each geometry box,
    then measure how much *visual evidence remains outside the text*. A title such as
    ``HOWEVER...`` leaves almost nothing; an illustrated panel still has color, edges or
    non-white structure after its speech text is masked.

    This is deliberately conservative. If OCR is uncertain, the geometry candidate stays.
    """
    if not panels or not text_boxes:
        kept = [dict(p) for p in panels]
        for i, p in enumerate(kept, 1):
            p["panel_id"] = i
        return kept, {"removed": 0, "removed_boxes": []}

    H, W = canvas_bgr.shape[:2]
    kept: List[Dict[str, Any]] = []
    removed: List[Dict[str, Any]] = []
    mask_pad = max(2, int(round(W * 0.015)))

    for panel in panels:
        p = dict(panel)
        x1 = max(0, min(W - 1, int(p["x1"]))); x2 = max(0, min(W - 1, int(p["x2"])))
        y1 = max(0, min(H - 1, int(p["y1"]))); y2 = max(0, min(H - 1, int(p["y2"])))
        if x2 < x1 or y2 < y1:
            kept.append(p)
            continue

        roi = canvas_bgr[y1:y2 + 1, x1:x2 + 1]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        sat = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)[:, :, 1]
        edges = cv2.Canny(gray, 50, 120) > 0
        text_mask = np.zeros(gray.shape, dtype=np.uint8)
        overlaps = 0

        for tb in text_boxes:
            if _box_intersection_area(p, tb) <= 0:
                continue
            overlaps += 1
            tx1 = max(x1, int(tb["x1"])) - x1
            ty1 = max(y1, int(tb["y1"])) - y1
            tx2 = min(x2, int(tb["x2"])) - x1
            ty2 = min(y2, int(tb["y2"])) - y1
            tx1 = max(0, tx1 - mask_pad); ty1 = max(0, ty1 - mask_pad)
            tx2 = min(text_mask.shape[1] - 1, tx2 + mask_pad)
            ty2 = min(text_mask.shape[0] - 1, ty2 + mask_pad)
            if tx2 >= tx1 and ty2 >= ty1:
                text_mask[ty1:ty2 + 1, tx1:tx2 + 1] = 1

        if overlaps == 0:
            kept.append(p)
            continue

        outside = text_mask == 0
        denom = max(1, int(outside.sum()))
        visual = (gray < 245) | (sat > 20) | edges
        residual_visual = float((visual & outside).sum() / denom)
        residual_sat = float(((sat > 20) & outside).sum() / denom)
        residual_edges = float((edges & outside).sum() / denom)
        text_coverage = float(text_mask.mean())

        # Two conservative gates. The first catches ordinary titles/captions; the
        # second allows a slightly noisier OCR mask but demands that text covers
        # more than half of the geometry candidate.
        is_text_only = (
            text_coverage >= 0.30 and residual_visual <= 0.10 and residual_sat <= 0.04
        ) or (
            text_coverage >= 0.52 and residual_visual <= 0.16 and residual_sat <= 0.06
        )

        p["ocr_text_coverage"] = text_coverage
        p["ocr_residual_visual"] = residual_visual
        p["ocr_residual_edges"] = residual_edges
        p["ocr_text_only"] = bool(is_text_only)
        if is_text_only:
            removed.append(p)
        else:
            kept.append(p)

    kept.sort(key=lambda b: (int(b["y1"]), int(b["x1"])))
    for i, p in enumerate(kept, 1):
        p["panel_id"] = i

    return kept, {
        "removed": len(removed),
        "removed_boxes": removed,
    }


def attach_text_to_panels(panels: Sequence[Dict[str, Any]], text_boxes: Sequence[Dict[str, Any]],
                          canvas_w: int, canvas_h: int) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Attach OCR text to real panels and expand crops in reading order.

    Unlike v1.3, there is no tiny absolute-distance cutoff. Webtoons routinely put a
    caption hundreds of source pixels away from its illustration. A text box between
    two real panels is owned by the nearest panel edge; text already overlapping a
    panel stays with that panel. OCR can expand a crop but still cannot merge two real
    panels or expand through a neighbour's artwork.
    """
    if not panels:
        return [], {"assigned": 0, "expanded_panels": 0, "unassigned": len(text_boxes), "assignments": {}}

    working = [dict(p) for p in sorted(panels, key=lambda b: (int(b["y1"]), int(b["x1"])))]
    assignments: Dict[int, List[Dict[str, Any]]] = {i: [] for i in range(len(working))}
    assigned = 0

    for tb in text_boxes:
        tw = max(1, int(tb["x2"]) - int(tb["x1"]) + 1)
        th = max(1, int(tb["y2"]) - int(tb["y1"]) + 1)
        ta = tw * th
        tcx = (float(tb["x1"]) + float(tb["x2"])) * 0.5
        tcy = (float(tb["y1"]) + float(tb["y2"])) * 0.5

        # 1) Text physically inside a panel is unambiguous.
        overlap_candidates: List[Tuple[float, int]] = []
        for idx, p in enumerate(working):
            inter = _box_intersection_area(tb, p)
            if inter > 0:
                overlap_candidates.append((inter / float(ta), idx))
        if overlap_candidates:
            overlap_ratio, idx = max(overlap_candidates, key=lambda x: x[0])
            if overlap_ratio >= 0.10:
                assignments[idx].append(dict(tb))
                assigned += 1
                continue

        # 2) Standalone caption/title: choose nearest real panel edge in reading order.
        candidates: List[Tuple[float, int, int]] = []
        for idx, p in enumerate(working):
            pw = max(1, int(p["x2"]) - int(p["x1"]) + 1)
            x_overlap = max(0, min(int(tb["x2"]), int(p["x2"])) - max(int(tb["x1"]), int(p["x1"])) + 1)
            x_overlap_ratio = x_overlap / float(max(1, min(tw, pw)))
            pcx = (float(p["x1"]) + float(p["x2"])) * 0.5
            x_distance = abs(tcx - pcx) / float(max(1, canvas_w))
            gap = _vertical_gap(tb, p)
            score = float(gap) + 0.18 * canvas_w * x_distance - 0.08 * canvas_w * x_overlap_ratio
            candidates.append((score, gap, idx))

        if not candidates:
            continue
        _, gap, best_idx = min(candidates, key=lambda x: x[0])

        # Inside the span of the story's detected panels, do not throw text away merely
        # because the artist used a dramatic amount of white space. Outside the story
        # extremes we still keep a generous safety limit to avoid random page metadata.
        inside_story_span = working[0]["y1"] <= tcy <= working[-1]["y2"]
        if inside_story_span or gap <= canvas_w * 2.5:
            assignments[best_idx].append(dict(tb))
            assigned += 1

    expanded_panels = 0
    for idx, p in enumerate(working):
        texts = assignments[idx]
        p["geometry_x1"] = int(p["x1"]); p["geometry_y1"] = int(p["y1"])
        p["geometry_x2"] = int(p["x2"]); p["geometry_y2"] = int(p["y2"])
        p["text_count"] = len(texts)
        p["ocr_expanded"] = False
        if not texts:
            continue

        ux1 = min([int(p["x1"])] + [int(t["x1"]) for t in texts])
        uy1 = min([int(p["y1"])] + [int(t["y1"]) for t in texts])
        ux2 = max([int(p["x2"])] + [int(t["x2"]) for t in texts])
        uy2 = max([int(p["y2"])] + [int(t["y2"]) for t in texts])

        pad_x = max(2, int(round(canvas_w * 0.012)))
        pad_y = max(2, int(round(canvas_w * 0.018)))
        ux1 = max(0, ux1 - pad_x); ux2 = min(canvas_w - 1, ux2 + pad_x)
        uy1 = max(0, uy1 - pad_y); uy2 = min(canvas_h - 1, uy2 + pad_y)

        # Never expand through a neighbouring real panel. The whole gutter may belong
        # to this crop, but the next/previous artwork does not.
        if idx > 0:
            uy1 = max(uy1, int(working[idx - 1]["y2"]) + 1)
        if idx + 1 < len(working):
            uy2 = min(uy2, int(working[idx + 1]["y1"]) - 1)

        changed = (ux1 < p["x1"] or uy1 < p["y1"] or ux2 > p["x2"] or uy2 > p["y2"])
        p["x1"], p["y1"], p["x2"], p["y2"] = int(ux1), int(uy1), int(ux2), int(uy2)
        p["ocr_expanded"] = bool(changed)
        if changed:
            expanded_panels += 1

    for i, p in enumerate(working, 1):
        p["panel_id"] = i

    return working, {
        "assigned": assigned,
        "expanded_panels": expanded_panels,
        "unassigned": max(0, len(text_boxes) - assigned),
        "assignments": assignments,
    }


def draw_ocr_overlay(strip: np.ndarray, geometry_boxes: Sequence[Dict[str, Any]],
                     final_boxes: Sequence[Dict[str, Any]], text_boxes: Sequence[Dict[str, Any]]) -> np.ndarray:
    out = strip.copy()
    # Geometry = red, detected text = orange, final OCR-aware crop = green.
    for b in geometry_boxes:
        cv2.rectangle(out, (int(b["x1"]), int(b["y1"])), (int(b["x2"]), int(b["y2"])), (0, 0, 255), 1)
    for b in text_boxes:
        cv2.rectangle(out, (int(b["x1"]), int(b["y1"])), (int(b["x2"]), int(b["y2"])), (0, 170, 255), 1)
    for b in final_boxes:
        cv2.rectangle(out, (int(b["x1"]), int(b["y1"])), (int(b["x2"]), int(b["y2"])), (0, 210, 80), 2)
        label = f"P{b.get('panel_id','')} T{b.get('text_count',0)}"
        cv2.putText(out, label, (int(b["x1"]) + 2, max(13, int(b["y1"]) + 13)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 210, 80), 1, cv2.LINE_AA)
    return out
