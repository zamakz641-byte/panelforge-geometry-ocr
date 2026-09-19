from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple, Dict, Any
import csv
import re

import cv2
import numpy as np

from image_io import imread_unicode


@dataclass
class DetectorConfig:
    analysis_width: int = 320
    active_threshold: float = 0.11
    min_panel_height: int = 55
    min_width_ratio: float = 0.32
    close_gap: int = 6
    split_threshold: float = 0.22
    split_min_height: int = 18
    white_threshold: int = 245
    saturation_threshold: int = 20
    text_reject: bool = True


def natural_key(path: str | Path):
    s = Path(path).name
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


def load_images(paths: Sequence[str | Path]) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    for p in sorted([str(x) for x in paths], key=natural_key):
        img = imread_unicode(p, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"Impossible de lire l'image: {p}")
        h, w = img.shape[:2]
        items.append({"path": p, "name": Path(p).name, "image": img, "height": h, "width": w})
    return items


def build_virtual_canvas(items: Sequence[Dict[str, Any]]) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    if not items:
        raise ValueError("Aucune image chargée")
    max_w = max(int(x["width"]) for x in items)
    total_h = sum(int(x["height"]) for x in items)
    canvas = np.full((total_h, max_w, 3), 255, dtype=np.uint8)
    meta: List[Dict[str, Any]] = []
    y = 0
    for item in items:
        img = item["image"]
        h, w = img.shape[:2]
        canvas[y:y+h, :w] = img
        meta.append({
            "name": item["name"],
            "path": item["path"],
            "width": w,
            "height": h,
            "y_offset": y,
            "y_end": y + h - 1,
        })
        y += h
    return canvas, meta


def _runs(mask: np.ndarray) -> List[Tuple[int, int]]:
    out: List[Tuple[int, int]] = []
    start = None
    flat = mask.astype(bool).ravel()
    for i, v in enumerate(flat):
        if v and start is None:
            start = i
        if start is not None and ((not v) or i == len(flat) - 1):
            end = i - 1 if not v else i
            out.append((start, end))
            start = None
    return out


def _scale_config(cfg: DetectorConfig, scale: float) -> DetectorConfig:
    return DetectorConfig(
        analysis_width=cfg.analysis_width,
        active_threshold=cfg.active_threshold,
        min_panel_height=max(8, int(round(cfg.min_panel_height * scale))),
        min_width_ratio=cfg.min_width_ratio,
        close_gap=max(1, int(round(cfg.close_gap * scale))),
        split_threshold=cfg.split_threshold,
        split_min_height=max(3, int(round(cfg.split_min_height * scale))),
        white_threshold=cfg.white_threshold,
        saturation_threshold=cfg.saturation_threshold,
        text_reject=cfg.text_reject,
    )


def _nms_merge(boxes: List[Dict[str, Any]], iou_thr: float = 0.35, contain_thr: float = 0.86) -> List[Dict[str, Any]]:
    if not boxes:
        return []
    boxes = sorted(boxes, key=lambda b: (b["y1"], b["x1"], -(b["y2"] - b["y1"])))
    merged: List[Dict[str, Any]] = []

    def area(b):
        return max(1, b["x2"] - b["x1"] + 1) * max(1, b["y2"] - b["y1"] + 1)

    def overlap(a, b):
        x1 = max(a["x1"], b["x1"]); y1 = max(a["y1"], b["y1"])
        x2 = min(a["x2"], b["x2"]); y2 = min(a["y2"], b["y2"])
        if x2 < x1 or y2 < y1:
            return 0, 0.0, 0.0
        inter = (x2 - x1 + 1) * (y2 - y1 + 1)
        aa = area(a); ab = area(b)
        iou = inter / float(aa + ab - inter)
        contain = inter / float(min(aa, ab))
        return inter, iou, contain

    for b in boxes:
        used = False
        for m in merged:
            _, iou, contain = overlap(b, m)
            if iou >= iou_thr or contain >= contain_thr:
                # Keep the box with better confidence, but expand if they are complementary.
                if b.get("confidence", 0) > m.get("confidence", 0):
                    m["x1"] = min(m["x1"], b["x1"])
                    m["y1"] = min(m["y1"], b["y1"])
                    m["x2"] = max(m["x2"], b["x2"])
                    m["y2"] = max(m["y2"], b["y2"])
                    m["confidence"] = max(m.get("confidence", 0), b.get("confidence", 0))
                used = True
                break
        if not used:
            merged.append(dict(b))

    merged.sort(key=lambda b: b["y1"])
    for i, b in enumerate(merged, 1):
        b["panel_id"] = i
    return merged



def _adaptive_valleys(score: np.ndarray, start: int, end: int, min_len: int, min_margin: int, min_peak: float = 0.24) -> List[Tuple[int, int, float]]:
    """Find sustained relative valleys, not just absolute black/white gaps.

    This is designed for webtoon scene transitions hidden inside dark/colored gradients.
    A valley is accepted when the local visual activity drops substantially relative
    to BOTH surrounding regions. The actual background color can be white, black,
    blue, red, etc.
    """
    seg = np.asarray(score[start:end+1], dtype=np.float32)
    if len(seg) < (min_margin * 2 + min_len):
        return []

    q70 = float(np.percentile(seg, 70))
    q35 = float(np.percentile(seg, 35))
    # Relative threshold: on an illustrated scene q70 is often ~0.7-0.9, while
    # a gradient bridge can remain around 0.25-0.4. Clamp for stability.
    thr = max(0.13, min(0.46, max(q35 * 0.92, q70 * 0.58)))
    mask = seg < thr

    candidates: List[Tuple[int, int, float]] = []
    context = max(min_margin, min_len * 2)
    for ra, rb in _runs(mask):
        length = rb - ra + 1
        if length < min_len:
            continue
        if ra < min_margin or (len(seg) - 1 - rb) < min_margin:
            continue

        left = seg[max(0, ra-context):ra]
        right = seg[rb+1:min(len(seg), rb+1+context)]
        if len(left) < min_margin or len(right) < min_margin:
            continue
        left_level = float(np.percentile(left, 65))
        right_level = float(np.percentile(right, 65))
        side = min(left_level, right_level)
        valley = float(np.mean(seg[ra:rb+1]))
        if side < min_peak:
            continue
        ratio = valley / max(side, 1e-6)
        # Require a meaningful relative drop. Longer valleys get a little tolerance.
        max_ratio = 0.68 if length >= min_len * 2 else 0.60
        if ratio > max_ratio:
            continue
        strength = (side - valley) * min(3.0, length / max(1.0, min_len))
        candidates.append((start + ra, start + rb, float(strength)))
    return candidates


def _choose_cuts_from_valleys(valleys: List[Tuple[int, int, float]], a: int, b: int, min_panel_height: int, max_cuts: int = 6) -> List[int]:
    """Greedily keep the strongest cuts while preserving minimum segment height."""
    if not valleys:
        return []
    chosen: List[int] = []
    for va, vb, strength in sorted(valleys, key=lambda t: t[2], reverse=True):
        cut = (va + vb) // 2
        points = sorted([a] + chosen + [b + 1])
        left = max(p for p in points if p <= cut)
        right = min(p for p in points if p > cut)
        if cut - left < min_panel_height or right - cut < min_panel_height:
            continue
        chosen.append(cut)
        if len(chosen) >= max_cuts:
            break
    return sorted(chosen)

def _row_detector(gray: np.ndarray, sat: np.ndarray, edges: np.ndarray, cfgs: DetectorConfig) -> Tuple[List[Dict[str, Any]], Dict[str, np.ndarray]]:
    H, W = gray.shape[:2]
    visual_px = ((gray > 26) & (gray < cfgs.white_threshold)) | (sat > cfgs.saturation_threshold) | (edges > 0)
    support = visual_px.mean(axis=1)
    nonwhite = (gray < cfgs.white_threshold).mean(axis=1)
    sat_fraction = (sat > cfgs.saturation_threshold).mean(axis=1)
    edge_fraction = (edges > 0).mean(axis=1)

    # Key idea: measure how much WIDTH of the row is visually occupied.
    # This works on white pages and on black-background pages, unlike a simple
    # “dark pixel” score that would incorrectly mark a pure black gutter as active.
    score = 0.58 * support + 0.24 * sat_fraction + 0.18 * edge_fraction
    score = np.convolve(score, np.ones(5, dtype=np.float32) / 5.0, mode="same")

    active = (score > cfgs.active_threshold).astype(np.uint8).reshape(-1, 1)
    if cfgs.close_gap > 0:
        kernel = np.ones((cfgs.close_gap * 2 + 1, 1), dtype=np.uint8)
        active = cv2.morphologyEx(active, cv2.MORPH_CLOSE, kernel)
    active = active.ravel().astype(bool)

    raw_runs = [r for r in _runs(active) if (r[1] - r[0] + 1) >= cfgs.min_panel_height]
    split_runs: List[Tuple[int, int]] = []
    for a, b in raw_runs:
        run_h = b - a + 1
        # Small/normal panels do not need scene-change splitting. Adaptive split is
        # reserved for tall webtoon regions, which reduces accidental over-segmentation.
        tall_enough = run_h >= max(cfgs.min_panel_height * 3, int(W * 1.15))
        if not tall_enough:
            split_runs.append((a, b))
            continue

        # 1) classical near-empty gutters
        classic_mask = (score[a:b+1] < cfgs.split_threshold) & (edge_fraction[a:b+1] < 0.08)
        classic = [
            (a + x, a + y, float(cfgs.split_threshold - np.mean(score[a+x:a+y+1])))
            for x, y in _runs(classic_mask)
            if (y - x + 1) >= cfgs.split_min_height
        ]
        # 2) relative valleys, including black/blue/red gradients
        adaptive = _adaptive_valleys(
            score, a, b,
            min_len=max(cfgs.split_min_height, 10),
            min_margin=max(cfgs.min_panel_height, int(W * 0.16)),
        )
        valleys = classic + adaptive
        cuts = _choose_cuts_from_valleys(valleys, a, b, cfgs.min_panel_height)

        prev = a
        for cut in cuts:
            if cut - prev >= cfgs.min_panel_height:
                split_runs.append((prev, cut - 1))
                prev = cut
        if b - prev + 1 >= cfgs.min_panel_height:
            split_runs.append((prev, b))

    boxes_small: List[Dict[str, Any]] = []
    for a, b in split_runs:
        rg = gray[a:b+1]
        rs = sat[a:b+1]
        re_ = edges[a:b+1]
        visual = (rg < cfgs.white_threshold) | (rs > cfgs.saturation_threshold) | (re_ > 0)
        col_support = visual.mean(axis=0)
        xs = np.where(col_support > 0.06)[0]
        if len(xs) == 0:
            continue
        x1, x2 = int(xs[0]), int(xs[-1])
        width_ratio = (x2 - x1 + 1) / float(W)
        nonwhite_mean = float((rg < cfgs.white_threshold).mean())
        sat_mean = float((rs > cfgs.saturation_threshold).mean())
        edge_mean = float((re_ > 0).mean())
        if width_ratio < cfgs.min_width_ratio:
            continue
        if cfgs.text_reject:
            if sat_mean < 0.01 and nonwhite_mean < 0.32 and width_ratio < 0.75:
                continue
        visual_strength = min(1.0, nonwhite_mean * 0.9 + sat_mean * 0.5 + edge_mean * 0.6)
        confidence = float(max(0.0, min(1.0, 0.35 + 0.45 * visual_strength + 0.20 * min(1.0, width_ratio))))
        boxes_small.append({
            "x1": x1, "y1": int(a), "x2": x2, "y2": int(b),
            "confidence": confidence,
            "width_ratio": float(width_ratio),
            "nonwhite": nonwhite_mean,
            "saturation": sat_mean,
            "edge_density": edge_mean,
            "source": "row",
        })
    diagnostics = {
        "score": score,
        "support": support,
        "nonwhite": nonwhite,
        "saturation": sat_fraction,
        "edge_fraction": edge_fraction,
    }
    return boxes_small, diagnostics


def _component_detector(gray: np.ndarray, sat: np.ndarray, edges: np.ndarray, cfgs: DetectorConfig) -> List[Dict[str, Any]]:
    H, W = gray.shape[:2]
    # Works well on black-background pages where row activity merges everything.
    bright = gray > 28
    colorful = sat > max(18, cfgs.saturation_threshold)
    edgy = edges > 0
    local_var = cv2.GaussianBlur(cv2.Laplacian(gray, cv2.CV_32F) ** 2, (0, 0), 1.0)
    textured = local_var > 18.0

    visual = (bright & (gray < 252)) | colorful | edgy | textured
    mask = (visual.astype(np.uint8) * 255)

    # Stitch together content inside a panel, but keep separate islands apart.
    k1 = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 11))
    k2 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 21))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k1)
    mask = cv2.dilate(mask, k2, iterations=1)
    mask = cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 9)), iterations=1)

    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    out: List[Dict[str, Any]] = []
    min_h = max(cfgs.min_panel_height, 14)
    min_w = max(12, int(round(W * max(0.18, cfgs.min_width_ratio * 0.55))))
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if h < min_h or w < min_w:
            continue
        x1, y1, x2, y2 = int(x), int(y), int(x + w - 1), int(y + h - 1)
        bbox_area = max(1, w * h)
        fill_ratio = area / float(bbox_area)
        width_ratio = w / float(W)
        rg = gray[y:y+h, x:x+w]
        rs = sat[y:y+h, x:x+w]
        re_ = edges[y:y+h, x:x+w]
        nonblack_mean = float((rg > 20).mean())
        bright_mean = float((rg > 40).mean())
        sat_mean = float((rs > cfgs.saturation_threshold).mean())
        edge_mean = float((re_ > 0).mean())

        # Reject isolated text balloons / SFX on empty black space.
        if cfgs.text_reject:
            if width_ratio < 0.36 and fill_ratio < 0.22 and sat_mean < 0.05 and edge_mean < 0.09:
                continue
            if h < min_h * 1.25 and width_ratio < 0.45 and bright_mean < 0.35 and sat_mean < 0.08:
                continue

        # Also reject giant full-page captures when everything merged.
        if width_ratio > 0.95 and h / float(H) > 0.92 and fill_ratio > 0.70 and sat_mean < 0.20:
            continue

        confidence = float(max(0.0, min(1.0,
            0.28 + 0.24 * min(1.0, width_ratio / 0.5) + 0.20 * min(1.0, fill_ratio / 0.35)
            + 0.16 * min(1.0, sat_mean / 0.18) + 0.12 * min(1.0, edge_mean / 0.08)
        )))
        out.append({
            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
            "confidence": confidence,
            "width_ratio": float(width_ratio),
            "nonwhite": nonblack_mean,
            "saturation": sat_mean,
            "edge_density": edge_mean,
            "fill_ratio": float(fill_ratio),
            "source": "component",
        })

    out.sort(key=lambda b: b["y1"])
    return out




def _split_component_boxes(boxes: List[Dict[str, Any]], gray: np.ndarray, sat: np.ndarray, edges: np.ndarray, cfgs: DetectorConfig) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    min_h = max(cfgs.min_panel_height, 14)
    for b in boxes:
        y1, y2 = int(b["y1"]), int(b["y2"])
        x1, x2 = int(b["x1"]), int(b["x2"])
        h = y2 - y1 + 1
        if h < min_h * 2:
            out.append(b)
            continue

        rg = gray[y1:y2+1, x1:x2+1]
        rs = sat[y1:y2+1, x1:x2+1]
        re = edges[y1:y2+1, x1:x2+1]
        visual = ((rg > 26) & (rg < cfgs.white_threshold)) | (rs > cfgs.saturation_threshold) | (re > 0)
        support = visual.mean(axis=1)
        satf = (rs > cfgs.saturation_threshold).mean(axis=1)
        edgef = (re > 0).mean(axis=1)
        row_score = 0.62 * support + 0.22 * satf + 0.16 * edgef
        row_score = np.convolve(row_score, np.ones(5, dtype=np.float32) / 5.0, mode="same")

        valley_thr = min(0.12, max(0.045, cfgs.split_threshold * 0.55))
        valleys = []
        for a, b_ in _runs(row_score < valley_thr):
            length = b_ - a + 1
            if length < max(12, cfgs.split_min_height):
                continue
            # Ignore valleys too close to box borders.
            if a < min_h or (h - 1 - b_) < min_h:
                continue
            strength = float((valley_thr - float(row_score[a:b_+1].mean())) * length)
            valleys.append((a, b_, strength))

        if not valleys:
            out.append(b)
            continue

        # Split only on the strongest internal valley. This fixes the common
        # “two panels merged into one dark block” case without exploding into
        # several tiny false panels.
        va, vb, _ = max(valleys, key=lambda t: t[2])
        cut = (va + vb) // 2
        if cut < min_h or (h - cut) < min_h:
            out.append(b)
            continue

        top = dict(b)
        top["y1"] = y1
        top["y2"] = y1 + cut - 1
        bot = dict(b)
        bot["y1"] = y1 + cut
        bot["y2"] = y2
        out.extend([top, bot])
    return out



def _postsplit_seam_profile(roi_bgr: np.ndarray) -> Dict[str, np.ndarray]:
    """Build a vertical seam profile for oversized webtoon crops.

    The original geometry detector is intentionally conservative and can merge several
    scenes when the artist uses a continuous dark/colored background instead of a clean
    white gutter. This profile looks for *scene seams* rather than empty rows only:
    broad color/layout changes, low-activity valleys and quiet full-width separators.
    """
    h, w = roi_bgr.shape[:2]
    if h < 3 or w < 3:
        z = np.zeros(max(1, h), dtype=np.float32)
        return {"score": z, "activity": z, "transition": z, "valley": z, "flat": z, "quiet": z}

    lab = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    gray_u8 = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    gray = gray_u8.astype(np.float32)
    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1]
    edge_mask = cv2.Canny(gray_u8, 50, 120) > 0

    row_std = np.clip(gray.std(axis=1) / 90.0, 0.0, 1.0)
    edgef = np.clip(edge_mask.mean(axis=1) * 7.0, 0.0, 1.0)
    satf = (sat > 25).mean(axis=1).astype(np.float32)
    activity = 0.48 * row_std + 0.34 * edgef + 0.18 * satf
    smooth_r = max(2, int(round(w * 0.018)))
    activity = np.convolve(activity, np.ones(2 * smooth_r + 1, dtype=np.float32) / (2 * smooth_r + 1), mode="same")

    band = max(4, int(round(w * 0.05)))
    transition = np.zeros(h, dtype=np.float32)
    for y in range(band, h - band):
        top = lab[y - band:y].mean(axis=0)  # W x 3
        bot = lab[y:y + band].mean(axis=0)
        per_col = np.linalg.norm(top - bot, axis=1) / 90.0
        coverage = float(np.mean(per_col > 0.12))
        mean_diff = float(np.clip(np.mean(per_col), 0.0, 1.0))

        ha = cv2.calcHist([gray_u8[y - band:y]], [0], None, [16], [0, 256])
        hb = cv2.calcHist([gray_u8[y:y + band]], [0], None, [16], [0, 256])
        cv2.normalize(ha, ha)
        cv2.normalize(hb, hb)
        hist_diff = float(cv2.compareHist(ha, hb, cv2.HISTCMP_BHATTACHARYYA))
        edge_diff = min(1.0, abs(float(edge_mask[y - band:y].mean()) - float(edge_mask[y:y + band].mean())) * 5.0)
        transition[y] = 0.36 * coverage + 0.28 * mean_diff + 0.28 * hist_diff + 0.08 * edge_diff

    context = max(12, int(round(w * 0.22)))
    valley = np.zeros(h, dtype=np.float32)
    for y in range(context, h - context):
        c0 = max(0, y - smooth_r * 2)
        c1 = min(h, y + smooth_r * 2 + 1)
        center = float(np.mean(activity[c0:c1]))
        left = activity[y - context:y - max(1, smooth_r)]
        right = activity[y + max(1, smooth_r):y + context]
        if len(left) == 0 or len(right) == 0:
            continue
        side = min(float(np.percentile(left, 60)), float(np.percentile(right, 60)))
        valley[y] = max(0.0, side - center)
    valley = np.clip(valley * 2.8, 0.0, 1.0)

    near_white = (gray > 245).mean(axis=1)
    near_black = (gray < 18).mean(axis=1)
    flat = (1.0 - row_std) * np.maximum(near_white, near_black) * (1.0 - edgef)
    flat = np.convolve(flat, np.ones(5, dtype=np.float32) / 5.0, mode="same")

    quiet = np.clip(1.0 - activity * 1.4, 0.0, 1.0)
    quiet = np.convolve(quiet, np.ones(7, dtype=np.float32) / 7.0, mode="same")

    score = 0.52 * transition + 0.23 * valley + 0.15 * flat + 0.10 * quiet
    score = np.convolve(score, np.ones(5, dtype=np.float32) / 5.0, mode="same")
    border = max(8, int(round(w * 0.30)))
    score[:border] = 0.0
    score[max(0, h - border):] = 0.0

    return {
        "score": score.astype(np.float32),
        "activity": activity.astype(np.float32),
        "transition": transition.astype(np.float32),
        "valley": valley.astype(np.float32),
        "flat": flat.astype(np.float32),
        "quiet": quiet.astype(np.float32),
    }


def split_oversized_panels(canvas_bgr: np.ndarray, panels: Sequence[Dict[str, Any]],
                           text_boxes: Sequence[Dict[str, Any]] = ()) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Second-pass splitter for merged tall webtoon regions.

    v1.5 could preserve OCR text but OCR was forbidden from influencing splits, so a
    1500-3000 px continuous scene could remain one giant crop forever. This pass keeps
    geometry authoritative while allowing OCR to *protect* text during visual splitting.

    Rules:
    - only oversized boxes are considered;
    - cuts must land on broad/quiet scene seams, never arbitrary fixed-height positions;
    - OCR zones receive a safety margin, so cuts cannot slice through bubbles/captions;
    - no new dependency and no VLM are required.
    """
    if not panels:
        return [], {"split_panels": 0, "new_panels": 0, "cuts": 0}

    H, W = canvas_bgr.shape[:2]
    final: List[Dict[str, Any]] = []
    split_count = 0
    cut_count = 0

    for panel in sorted((dict(p) for p in panels), key=lambda b: (int(b["y1"]), int(b["x1"]))):
        x1 = max(0, min(W - 1, int(panel["x1"])))
        x2 = max(0, min(W - 1, int(panel["x2"])))
        y1 = max(0, min(H - 1, int(panel["y1"])))
        y2 = max(0, min(H - 1, int(panel["y2"])))
        if x2 < x1 or y2 < y1:
            final.append(panel)
            continue

        ph = y2 - y1 + 1
        pw = x2 - x1 + 1
        # A normal illustration may naturally be tall. The second pass starts only
        # when the box is suspiciously elongated relative to its own width.
        if ph <= max(220, int(round(pw * 2.20))):
            final.append(panel)
            continue

        roi = canvas_bgr[y1:y2 + 1, x1:x2 + 1]
        prof = _postsplit_seam_profile(roi)
        score = prof["score"]
        activity = prof["activity"]
        transition = prof["transition"]
        valley = prof["valley"]
        flat = prof["flat"]
        quiet = prof["quiet"]

        min_seg = max(58, int(round(pw * 0.58)))
        local_radius = max(4, int(round(pw * 0.05)))
        text_pad = max(3, int(round(pw * 0.03)))
        forbidden = np.zeros(ph, dtype=bool)
        local_texts: List[Tuple[int, int]] = []

        for tb in text_boxes:
            tx1 = int(tb.get("x1", 0)); tx2 = int(tb.get("x2", -1))
            ty1 = int(tb.get("y1", 0)); ty2 = int(tb.get("y2", -1))
            if tx2 < x1 or tx1 > x2 or ty2 < y1 or ty1 > y2:
                continue
            ly1 = max(0, ty1 - y1)
            ly2 = min(ph - 1, ty2 - y1)
            if ly2 >= ly1:
                local_texts.append((ly1, ly2))
            a = max(0, ly1 - text_pad)
            b = min(ph - 1, ly2 + text_pad)
            if b >= a:
                forbidden[a:b + 1] = True

        # OCR line boxes are grouped into caption/bubble clusters. Gaps between two
        # distinct text clusters are useful anchors on layouts where the artwork itself
        # is continuous (system windows over one tall character, narration on gradients,
        # etc.). OCR still does not create a crop blindly: the actual cut is searched
        # inside the text-free gap using the visual seam profile.
        text_clusters: List[List[int]] = []
        join_gap = max(5, int(round(pw * 0.14)))
        for ta, tb_ in sorted(local_texts):
            if not text_clusters or ta - text_clusters[-1][1] > join_gap:
                text_clusters.append([ta, tb_])
            else:
                text_clusters[-1][1] = max(text_clusters[-1][1], tb_)

        candidates: List[Tuple[int, float]] = []
        for ry in range(local_radius, ph - local_radius):
            if forbidden[ry] or ry < min_seg or (ph - ry) < min_seg:
                continue
            local = score[max(0, ry - local_radius):min(ph, ry + local_radius + 1)]
            if score[ry] + 1e-6 < float(local.max()):
                continue
            if score[ry] < 0.245:
                continue

            # Safety gate. A strong color jump alone is not enough if the row is busy
            # (e.g. eyes, mouth, body contour). At least one separator-like signal must
            # agree with it.
            safe = (
                (quiet[ry] >= 0.32 and activity[ry] <= 0.62) or
                flat[ry] >= 0.22 or
                (valley[ry] >= 0.14 and transition[ry] >= 0.62) or
                (transition[ry] >= 0.70 and activity[ry] <= 0.55)
            )
            if not safe:
                continue
            candidates.append((ry, float(score[ry])))

        # OCR-seeded gap candidates. They are deliberately boosted a little because
        # they encode reading-order structure that pure geometry cannot see, but the
        # selected row still has to live in a genuine text-free interval.
        min_text_gap = max(6, int(round(pw * 0.08)))
        for left_cluster, right_cluster in zip(text_clusters[:-1], text_clusters[1:]):
            ga = int(left_cluster[1] + text_pad)
            gb = int(right_cluster[0] - text_pad)
            ga = max(ga, min_seg)
            gb = min(gb, ph - min_seg)
            if gb - ga + 1 < min_text_gap:
                continue
            rows = np.arange(ga, gb + 1, dtype=np.int32)
            if rows.size == 0:
                continue
            utility = score[rows] + 0.20 * quiet[rows] + 0.08 * (1.0 - np.clip(activity[rows], 0.0, 1.0))
            best_i = int(np.argmax(utility))
            ry = int(rows[best_i])
            gap_ratio = (gb - ga + 1) / float(max(1, pw))
            gap_safe = (
                quiet[ry] >= 0.26 or
                flat[ry] >= 0.18 or
                (transition[ry] >= 0.62 and activity[ry] <= 0.62) or
                (valley[ry] >= 0.10 and transition[ry] >= 0.50 and activity[ry] <= 0.62) or
                (gap_ratio >= 0.65 and activity[ry] <= 0.68)
            )
            if not gap_safe:
                continue
            priority = float(score[ry] + 0.14 + min(0.16, gap_ratio * 0.16))
            candidates.append((ry, priority))

        # Keep only a modest number of the strongest seams. This prevents a long dark
        # page from exploding into dozens of tiny slices merely because it contains SFX.
        target_segments = int(np.ceil(ph / float(max(1, int(round(pw * 2.15))))))
        if len(text_clusters) >= 2:
            text_target = min(len(text_clusters), max(2, int(np.ceil(ph / float(max(1, int(round(pw * 1.55))))))))
            target_segments = max(target_segments, text_target)
        target_segments = max(1, min(8, target_segments))
        max_cuts = max(0, target_segments - 1)

        chosen: List[int] = []
        for ry, strength in sorted(candidates, key=lambda t: t[1], reverse=True):
            if len(chosen) >= max_cuts:
                break
            points = sorted([0] + chosen + [ph])
            left = max(p for p in points if p <= ry)
            right = min(p for p in points if p > ry)
            if ry - left < min_seg or right - ry < min_seg:
                continue
            chosen.append(ry)

        # Hard fallback only on a genuinely quiet valley. Never cut through illustrated
        # content just to obey a height target.
        if not chosen and ph > int(round(pw * 3.6)):
            lo = min_seg
            hi = ph - min_seg
            safe_rows = [
                ry for ry in range(lo, hi)
                if (not forbidden[ry]) and activity[ry] <= 0.27 and quiet[ry] >= 0.55
            ]
            if safe_rows:
                center = ph * 0.5
                best = max(safe_rows, key=lambda ry: float(quiet[ry]) - 0.12 * abs(ry - center) / max(1.0, ph * 0.5))
                chosen = [int(best)]

        if not chosen:
            final.append(panel)
            continue

        chosen.sort()
        bounds = [0] + chosen + [ph]
        pieces: List[Dict[str, Any]] = []
        for a, b in zip(bounds[:-1], bounds[1:]):
            if b - a < max(20, min_seg // 2):
                continue
            piece = dict(panel)
            piece["y1"] = y1 + a
            piece["y2"] = y1 + b - 1
            piece["source"] = str(piece.get("source", "geometry")) + "+postsplit"
            piece["postsplit"] = True
            piece["postsplit_parent_y1"] = y1
            piece["postsplit_parent_y2"] = y2
            pieces.append(piece)

        if len(pieces) >= 2:
            final.extend(pieces)
            split_count += 1
            cut_count += len(pieces) - 1
        else:
            final.append(panel)

    final.sort(key=lambda b: (int(b["y1"]), int(b["x1"])))
    for i, b in enumerate(final, 1):
        b["panel_id"] = i
    return final, {
        "split_panels": int(split_count),
        "new_panels": int(max(0, len(final) - len(panels))),
        "cuts": int(cut_count),
    }


def detect_panels(canvas_bgr: np.ndarray, cfg: DetectorConfig | None = None) -> Tuple[List[Dict[str, Any]], Dict[str, np.ndarray]]:
    cfg = cfg or DetectorConfig()
    H0, W0 = canvas_bgr.shape[:2]

    scale = 1.0
    if cfg.analysis_width > 0 and W0 > cfg.analysis_width:
        scale = cfg.analysis_width / float(W0)
    if scale < 1.0:
        small = cv2.resize(canvas_bgr, (int(round(W0 * scale)), int(round(H0 * scale))), interpolation=cv2.INTER_AREA)
    else:
        small = canvas_bgr
        scale = 1.0

    cfgs = _scale_config(cfg, scale)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1]
    edges = cv2.Canny(gray, 50, 120)

    row_boxes, diagnostics = _row_detector(gray, sat, edges, cfgs)
    comp_boxes = _component_detector(gray, sat, edges, cfgs)

    H, W = gray.shape[:2]
    # If the row detector collapsed into one giant dark page, switch to component-first mode.
    giant_row = False
    if len(row_boxes) == 1:
        b = row_boxes[0]
        giant_row = (b["y2"] - b["y1"] + 1) / float(H) > 0.92 and (b["x2"] - b["x1"] + 1) / float(W) > 0.92

    boxes_small: List[Dict[str, Any]] = []
    if giant_row and len(comp_boxes) >= 2:
        boxes_small = comp_boxes
    else:
        boxes_small.extend(row_boxes)
        for cb in comp_boxes:
            cb_h = cb["y2"] - cb["y1"] + 1
            cb_w = cb["x2"] - cb["x1"] + 1
            cb_area = cb_w * cb_h
            if len(row_boxes) >= 2 and cb_h / float(H) > 0.85 and cb_w / float(W) > 0.85:
                continue

            keep = True
            overlap_rows = 0
            for rb in row_boxes:
                inter_x1 = max(cb["x1"], rb["x1"]); inter_y1 = max(cb["y1"], rb["y1"])
                inter_x2 = min(cb["x2"], rb["x2"]); inter_y2 = min(cb["y2"], rb["y2"])
                if inter_x2 >= inter_x1 and inter_y2 >= inter_y1:
                    inter = (inter_x2 - inter_x1 + 1) * (inter_y2 - inter_y1 + 1)
                    rb_area = (rb["x2"] - rb["x1"] + 1) * (rb["y2"] - rb["y1"] + 1)
                    if inter / float(rb_area) > 0.45:
                        overlap_rows += 1
                    if inter / float(cb_area) > 0.82:
                        keep = False
                        break
            # If one component box swallows multiple row boxes, it is usually a bad
            # dark-layout merge and should not be added.
            if overlap_rows >= 2:
                keep = False
            if keep:
                boxes_small.append(cb)

    boxes_small = _nms_merge(boxes_small)

    inv = 1.0 / scale
    boxes: List[Dict[str, Any]] = []
    for i, b in enumerate(boxes_small, 1):
        x1 = int(round(b["x1"] * inv))
        y1 = int(round(b["y1"] * inv))
        x2 = int(round((b["x2"] + 1) * inv - 1))
        y2 = int(round((b["y2"] + 1) * inv - 1))
        x1 = max(0, min(W0 - 1, x1)); x2 = max(0, min(W0 - 1, x2))
        y1 = max(0, min(H0 - 1, y1)); y2 = max(0, min(H0 - 1, y2))
        out = dict(b)
        out.update({"panel_id": i, "x1": x1, "y1": y1, "x2": x2, "y2": y2})
        boxes.append(out)

    diagnostics.update({
        "analysis_image": small,
        "mode": "component-first" if (giant_row and len(comp_boxes) >= 2) else "hybrid",
        "component_boxes_count": np.array([len(comp_boxes)], dtype=np.int32),
    })
    return boxes, diagnostics


def global_box_to_sources(box: Dict[str, Any], meta: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    y1, y2 = int(box["y1"]), int(box["y2"])
    for m in meta:
        if y2 < m["y_offset"] or y1 > m["y_end"]:
            continue
        ly1 = max(0, y1 - m["y_offset"])
        ly2 = min(m["height"] - 1, y2 - m["y_offset"])
        x1 = max(0, min(m["width"] - 1, int(box["x1"])))
        x2 = max(0, min(m["width"] - 1, int(box["x2"])))
        out.append({
            "panel_id": box.get("panel_id"),
            "file": m["name"],
            "x1": x1, "y1": ly1, "x2": x2, "y2": ly2,
            "global_y1": max(y1, m["y_offset"]),
            "global_y2": min(y2, m["y_end"]),
        })
    return out


def read_boxes_csv(path: str | Path, meta: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    with open(path, "r", newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return []

    by_name = {m["name"].lower(): m for m in meta}
    out = []
    for idx, r in enumerate(rows, 1):
        low = {str(k).strip().lower(): v for k, v in r.items() if k is not None}
        pid = low.get("panel_id") or low.get("id") or idx
        try:
            pid = int(float(pid))
        except Exception:
            pid = idx

        def getnum(*names):
            for n in names:
                if n in low and low[n] not in (None, ""):
                    return int(round(float(low[n])))
            return None

        x1 = getnum("x1", "left")
        x2 = getnum("x2", "right")
        gy1 = getnum("y1_global", "global_y1")
        gy2 = getnum("y2_global", "global_y2")

        file_name = low.get("file") or low.get("filename") or low.get("image") or low.get("source_file")
        if gy1 is None or gy2 is None:
            y1 = getnum("y1", "top")
            y2 = getnum("y2", "bottom")
            if file_name:
                m = by_name.get(Path(str(file_name)).name.lower())
                if m is None:
                    continue
                gy1 = m["y_offset"] + (y1 or 0)
                gy2 = m["y_offset"] + (y2 or 0)
            else:
                gy1, gy2 = y1, y2
        if None in (x1, x2, gy1, gy2):
            continue
        out.append({"panel_id": pid, "x1": x1, "y1": gy1, "x2": x2, "y2": gy2})
    return out


def write_boxes_csv(path: str | Path, boxes: Sequence[Dict[str, Any]], meta: Sequence[Dict[str, Any]]) -> None:
    fields = [
        "panel_id", "x1", "y1_global", "x2", "y2_global", "confidence",
        "width_ratio", "nonwhite", "saturation", "edge_density", "source", "source_files"
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for b in boxes:
            files = [frag["file"] for frag in global_box_to_sources(b, meta)]
            w.writerow({
                "panel_id": b.get("panel_id", ""),
                "x1": b["x1"], "y1_global": b["y1"], "x2": b["x2"], "y2_global": b["y2"],
                "confidence": f"{float(b.get('confidence', 0)):.4f}",
                "width_ratio": f"{float(b.get('width_ratio', 0)):.4f}",
                "nonwhite": f"{float(b.get('nonwhite', 0)):.4f}",
                "saturation": f"{float(b.get('saturation', 0)):.4f}",
                "edge_density": f"{float(b.get('edge_density', 0)):.4f}",
                "source": b.get("source", ""),
                "source_files": "|".join(files),
            })


def draw_overlay(canvas: np.ndarray, predicted: Sequence[Dict[str, Any]] = (), csv_boxes: Sequence[Dict[str, Any]] = ()) -> np.ndarray:
    out = canvas.copy()
    for b in predicted:
        p1 = (int(b["x1"]), int(b["y1"])); p2 = (int(b["x2"]), int(b["y2"]))
        cv2.rectangle(out, p1, p2, (0, 0, 255), 2)
        src = b.get("source", "")
        label = f"A{b.get('panel_id','')}" + (f"/{src[0].upper()}" if src else "")
        cv2.putText(out, label, (p1[0] + 3, max(14, p1[1] + 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1, cv2.LINE_AA)
    for b in csv_boxes:
        p1 = (int(b["x1"]), int(b["y1"])); p2 = (int(b["x2"]), int(b["y2"]))
        cv2.rectangle(out, p1, p2, (0, 180, 0), 2)
        cv2.putText(out, f"C{b.get('panel_id','')}", (p1[0] + 3, max(14, p1[1] + 30)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 180, 0), 1, cv2.LINE_AA)
    return out
