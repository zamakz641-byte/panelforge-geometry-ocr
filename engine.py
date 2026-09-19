from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from collections import OrderedDict
from typing import Callable, Iterable, List, Dict, Any, Tuple
import csv
import json
import os
import re
import shutil
import tempfile
import time
import zipfile

import cv2
import numpy as np
from PIL import Image

from detector import DetectorConfig, detect_panels, draw_overlay, split_oversized_panels
from image_io import imread_unicode, imwrite_unicode
from text_detector import detect_text_regions, filter_text_only_panels, attach_text_to_panels, draw_ocr_overlay

IMAGE_EXTS = {'.png', '.jpg', '.jpeg', '.webp', '.bmp'}


def natural_key(value: str | Path):
    s = str(value).replace('\\', '/').lower()
    return [int(t) if t.isdigit() else t for t in re.split(r'(\d+)', s)]


@dataclass
class SourceImage:
    path: Path
    display_name: str
    orig_w: int
    orig_h: int
    analysis_y0: int = 0
    analysis_y1: int = 0
    analysis_h: int = 0
    analysis_w: int = 0


class ImageCache:
    def __init__(self, max_items: int = 3):
        self.max_items = max(1, max_items)
        self.cache: OrderedDict[str, np.ndarray] = OrderedDict()

    def get(self, path: Path) -> np.ndarray:
        key = str(path)
        if key in self.cache:
            img = self.cache.pop(key)
            self.cache[key] = img
            return img
        img = imread_unicode(key, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"Impossible de lire l'image source: {path}")
        self.cache[key] = img
        while len(self.cache) > self.max_items:
            self.cache.popitem(last=False)
        return img

    def clear(self):
        self.cache.clear()


def _safe_progress(cb: Callable[[Dict[str, Any]], None] | None, **payload):
    if cb:
        try:
            cb(payload)
        except Exception:
            pass


def _scan_folder(folder: Path) -> List[Path]:
    # Prefer direct children because webtoon page folders are normally flat.
    direct = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
    if direct:
        return sorted(direct, key=natural_key)

    # Fallback to recursive discovery, while ignoring our own output folders.
    found: List[Path] = []
    for p in folder.rglob('*'):
        if not p.is_file() or p.suffix.lower() not in IMAGE_EXTS:
            continue
        if any(part.lower().startswith('panelcrops_geometry') for part in p.parts):
            continue
        found.append(p)
    return sorted(found, key=lambda p: natural_key(p.relative_to(folder)))


def _extract_cbz(cbz_path: Path, temp_dir: Path) -> List[Path]:
    paths: List[Path] = []
    with zipfile.ZipFile(cbz_path, 'r') as zf:
        infos = [i for i in zf.infolist() if not i.is_dir() and Path(i.filename).suffix.lower() in IMAGE_EXTS]
        infos.sort(key=lambda i: natural_key(i.filename))
        for idx, info in enumerate(infos, 1):
            ext = Path(info.filename).suffix.lower()
            out = temp_dir / f"source_{idx:06d}{ext}"
            with zf.open(info, 'r') as src, open(out, 'wb') as dst:
                shutil.copyfileobj(src, dst)
            paths.append(out)
    return paths


def _image_size(path: Path) -> Tuple[int, int]:
    with Image.open(path) as im:
        return int(im.width), int(im.height)


def prepare_sources(paths: List[Path], analysis_width: int, progress=None) -> Tuple[np.ndarray, List[SourceImage]]:
    if not paths:
        raise ValueError('Aucune image trouvée.')

    meta: List[SourceImage] = []
    total_h = 0
    n = len(paths)

    # Header-only pass: cheap and avoids keeping originals in memory.
    for i, p in enumerate(paths, 1):
        w, h = _image_size(p)
        if w <= 0 or h <= 0:
            continue
        ah = max(1, int(round(h * (analysis_width / float(w)))))
        item = SourceImage(
            path=p,
            display_name=p.name,
            orig_w=w,
            orig_h=h,
            analysis_y0=total_h,
            analysis_y1=total_h + ah - 1,
            analysis_h=ah,
            analysis_w=analysis_width,
        )
        meta.append(item)
        total_h += ah
        _safe_progress(progress, stage='prepare', progress=0.04 + 0.08 * (i / n), message=f'Lecture {i}/{n}')

    if not meta:
        raise ValueError('Les images n’ont pas pu être lues.')

    # Low-res strip only. This is the key to cross-file panel continuity without HD RAM abuse.
    strip = np.full((total_h, analysis_width, 3), 255, dtype=np.uint8)
    for i, item in enumerate(meta, 1):
        img = imread_unicode(item.path, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"Impossible de lire {item.display_name}")
        small = cv2.resize(img, (analysis_width, item.analysis_h), interpolation=cv2.INTER_AREA)
        strip[item.analysis_y0:item.analysis_y1 + 1, :, :] = small
        del img, small
        _safe_progress(progress, stage='strip', progress=0.12 + 0.18 * (i / len(meta)), message=f'Strip {i}/{len(meta)}')

    return strip, meta


def _panel_fragments(box: Dict[str, Any], meta: List[SourceImage]) -> List[Tuple[SourceImage, Tuple[int, int, int, int]]]:
    """Return source image + original-resolution crop coordinates."""
    ax1, ay1, ax2, ay2 = map(int, (box['x1'], box['y1'], box['x2'], box['y2']))
    frags = []
    for m in meta:
        if ay2 < m.analysis_y0 or ay1 > m.analysis_y1:
            continue
        iy1 = max(ay1, m.analysis_y0)
        iy2 = min(ay2, m.analysis_y1)
        local_ay1 = iy1 - m.analysis_y0
        local_ay2 = iy2 - m.analysis_y0

        sx = m.orig_w / float(m.analysis_w)
        sy = m.orig_h / float(m.analysis_h)
        ox1 = max(0, min(m.orig_w - 1, int(round(ax1 * sx))))
        ox2 = max(0, min(m.orig_w - 1, int(round((ax2 + 1) * sx - 1))))
        oy1 = max(0, min(m.orig_h - 1, int(round(local_ay1 * sy))))
        oy2 = max(0, min(m.orig_h - 1, int(round((local_ay2 + 1) * sy - 1))))
        if ox2 >= ox1 and oy2 >= oy1:
            frags.append((m, (ox1, oy1, ox2, oy2)))
    return frags


def _concat_fragments(images: List[np.ndarray]) -> np.ndarray:
    if len(images) == 1:
        return images[0]
    max_w = max(im.shape[1] for im in images)
    fixed = []
    for im in images:
        if im.shape[1] == max_w:
            fixed.append(im)
            continue
        scale = max_w / float(im.shape[1])
        nh = max(1, int(round(im.shape[0] * scale)))
        fixed.append(cv2.resize(im, (max_w, nh), interpolation=cv2.INTER_CUBIC))
    return np.vstack(fixed)


def _clean_output(output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    for p in output_dir.glob('panel_*.png'):
        try:
            p.unlink()
        except OSError:
            pass
    for name in ('panels.csv', 'report.json'):
        p = output_dir / name
        if p.exists():
            try:
                p.unlink()
            except OSError:
                pass
    debug = output_dir / '_debug'
    if debug.exists():
        shutil.rmtree(debug, ignore_errors=True)


def _save_panels(boxes: List[Dict[str, Any]], meta: List[SourceImage], output_dir: Path, progress=None,
                 progress_start: float = 0.48, progress_end: float = 0.96) -> List[Dict[str, Any]]:
    cache = ImageCache(max_items=3)
    rows: List[Dict[str, Any]] = []
    total = max(1, len(boxes))

    for idx, box in enumerate(boxes, 1):
        fragments = _panel_fragments(box, meta)
        if not fragments:
            continue
        crops: List[np.ndarray] = []
        source_names = []
        for m, (x1, y1, x2, y2) in fragments:
            src = cache.get(m.path)
            crop = src[y1:y2 + 1, x1:x2 + 1].copy()
            if crop.size == 0:
                continue
            crops.append(crop)
            source_names.append(m.display_name)
        if not crops:
            continue

        final = _concat_fragments(crops)
        filename = f'panel_{idx:04d}.png'
        out_path = output_dir / filename
        ok = imwrite_unicode(out_path, final, [cv2.IMWRITE_PNG_COMPRESSION, 2])
        if not ok:
            raise RuntimeError(f'Échec de sauvegarde: {out_path}')

        rows.append({
            'panel_id': idx,
            'file': filename,
            'confidence': float(box.get('confidence', 0.0)),
            'analysis_x1': int(box['x1']),
            'analysis_y1': int(box['y1']),
            'analysis_x2': int(box['x2']),
            'analysis_y2': int(box['y2']),
            'source_files': '|'.join(source_names),
            'cross_file': len(source_names) > 1,
            'width': int(final.shape[1]),
            'height': int(final.shape[0]),
            'text_count': int(box.get('text_count', 0)),
            'ocr_expanded': bool(box.get('ocr_expanded', False)),
        })
        _safe_progress(progress, stage='crop', progress=progress_start + (progress_end - progress_start) * (idx / total), message=f'Crop {idx}/{len(boxes)}')

    cache.clear()
    return rows


def _write_csv(output_dir: Path, rows: List[Dict[str, Any]]):
    if not rows:
        return
    fields = list(rows[0].keys())
    with open(output_dir / 'panels.csv', 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def _save_debug(output_dir: Path, strip: np.ndarray, boxes: List[Dict[str, Any]],
                geometry_boxes: List[Dict[str, Any]] | None = None,
                text_boxes: List[Dict[str, Any]] | None = None) -> Dict[str, Any]:
    """Save the low-resolution debug overlay without ever breaking extraction.

    JPEG encoders have a hard maximum dimension around 65,500 px. A complete
    webtoon chapter can easily exceed that height even at the 192 px analysis
    width. v1.6.1 therefore failed at the very end of otherwise-successful runs.

    Short overlays keep the historical single ``analysis_overlay.jpg`` file.
    Tall overlays are split into numbered JPEG chunks and an index JSON records
    their global Y ranges. Debug output is diagnostic only, so an encoding
    failure is recorded instead of aborting the panel extraction.
    """
    debug = output_dir / '_debug'
    debug.mkdir(parents=True, exist_ok=True)
    if geometry_boxes is not None and text_boxes is not None:
        overlay = draw_ocr_overlay(strip, geometry_boxes, boxes, text_boxes)
    else:
        overlay = draw_overlay(strip, boxes)

    h, w = overlay.shape[:2]
    jpeg_safe_dim = 65000
    chunk_h = 30000
    saved: List[Dict[str, Any]] = []
    errors: List[str] = []

    def save_piece(path: Path, image: np.ndarray, y0: int, y1: int) -> bool:
        ok = imwrite_unicode(path, image, [cv2.IMWRITE_JPEG_QUALITY, 91])
        if ok:
            saved.append({
                'file': path.name,
                'y0': int(y0),
                'y1': int(y1),
                'width': int(image.shape[1]),
                'height': int(image.shape[0]),
            })
            return True
        errors.append(f"Échec de sauvegarde: {path}")
        return False

    if h < jpeg_safe_dim and w < jpeg_safe_dim:
        save_piece(debug / 'analysis_overlay.jpg', overlay, 0, h - 1)
    else:
        part = 1
        for y0 in range(0, h, chunk_h):
            y1 = min(h, y0 + chunk_h)
            piece = overlay[y0:y1].copy()
            save_piece(debug / f'analysis_overlay_{part:03d}.jpg', piece, y0, y1 - 1)
            part += 1

    index = {
        'overlay_width': int(w),
        'overlay_height': int(h),
        'chunked': len(saved) > 1,
        'files': saved,
        'errors': errors,
    }
    try:
        with open(debug / 'analysis_overlay_index.json', 'w', encoding='utf-8') as f:
            json.dump(index, f, indent=2, ensure_ascii=False)
        if errors:
            (debug / 'debug_warning.txt').write_text('\n'.join(errors), encoding='utf-8')
    except OSError:
        pass

    return index


def process_source(source_path: str, source_type: str, analysis_width: int = 192,
                   save_debug: bool = True, use_ocr: bool = True, ocr_width: int = 768,
                   progress=None) -> Dict[str, Any]:
    started = time.perf_counter()
    src = Path(source_path).resolve()
    if analysis_width not in (160, 192, 224, 256, 320):
        analysis_width = 192

    if source_type == 'cbz':
        if not src.is_file() or src.suffix.lower() != '.cbz':
            raise ValueError('Sélectionne un fichier .cbz valide.')
        output_dir = src.parent / f'{src.stem} output'
        temp_ctx = tempfile.TemporaryDirectory(prefix='panelforge_cbz_')
        temp_root = Path(temp_ctx.name)
        _safe_progress(progress, stage='extract', progress=0.01, message='Extraction du CBZ…')
        paths = _extract_cbz(src, temp_root)
    elif source_type == 'folder':
        if not src.is_dir():
            raise ValueError('Sélectionne un dossier valide.')
        output_dir = src / 'PanelCrops_Geometry'
        temp_ctx = None
        paths = _scan_folder(src)
    else:
        raise ValueError('Type de source inconnu.')

    try:
        if not paths:
            raise ValueError('Aucune image PNG/JPG/JPEG/WEBP/BMP trouvée.')

        _clean_output(output_dir)
        _safe_progress(progress, stage='prepare', progress=0.03, message=f'{len(paths)} images trouvées')

        t_strip = time.perf_counter()
        strip, meta = prepare_sources(paths, analysis_width, progress=progress)
        strip_sec = time.perf_counter() - t_strip

        _safe_progress(progress, stage='detect', progress=0.31, message='Détection géométrique…')
        t_detect = time.perf_counter()
        cfg = DetectorConfig(analysis_width=analysis_width)
        geometry_boxes, diagnostics = detect_panels(strip, cfg)
        detect_sec = time.perf_counter() - t_detect
        boxes = geometry_boxes
        text_boxes: List[Dict[str, Any]] = []
        ocr_stats = {'assigned': 0, 'expanded_panels': 0, 'unassigned': 0}
        split_stats = {'split_panels': 0, 'new_panels': 0, 'cuts': 0}
        ocr_sec = 0.0
        _safe_progress(progress, stage='detect', progress=0.34, message=f'{len(boxes)} panels géométriques détectés')

        if use_ocr and boxes:
            t_ocr = time.perf_counter()
            text_boxes = detect_text_regions(meta, Path(__file__).resolve().parent, ocr_width=ocr_width, progress=progress)
            real_panels, text_only_stats = filter_text_only_panels(strip, geometry_boxes, text_boxes)
            real_panels, split_stats = split_oversized_panels(strip, real_panels, text_boxes)
            boxes, ocr_stats = attach_text_to_panels(
                real_panels, text_boxes, canvas_w=int(strip.shape[1]), canvas_h=int(strip.shape[0])
            )
            ocr_stats['text_only_removed'] = int(text_only_stats.get('removed', 0))
            ocr_sec = time.perf_counter() - t_ocr
            _safe_progress(
                progress, stage='ocr', progress=0.59,
                message=(f"OCR : {len(text_boxes)} zones texte • "
                         f"{ocr_stats.get('text_only_removed', 0)} faux panels texte supprimés • "
                         f"{split_stats.get('cuts', 0)} séparations de grands panels • "
                         f"{ocr_stats.get('expanded_panels', 0)} crops élargis")
            )
        elif boxes:
            boxes, split_stats = split_oversized_panels(strip, boxes, ())

        t_crop = time.perf_counter()
        crop_start = 0.60 if use_ocr else 0.48
        rows = _save_panels(boxes, meta, output_dir, progress=progress, progress_start=crop_start, progress_end=0.96)
        crop_sec = time.perf_counter() - t_crop

        _write_csv(output_dir, rows)
        debug_info: Dict[str, Any] = {'files': [], 'errors': [], 'chunked': False}
        if save_debug:
            debug_info = _save_debug(
                output_dir, strip, boxes,
                geometry_boxes if use_ocr else None,
                text_boxes if use_ocr else None,
            )

        cross_file = sum(1 for r in rows if r['cross_file'])
        total_sec = time.perf_counter() - started
        report = {
            'source': str(src),
            'source_type': source_type,
            'output_dir': str(output_dir),
            'source_images': len(paths),
            'panels_detected': len(rows),
            'cross_file_panels': cross_file,
            'analysis_width': analysis_width,
            'analysis_strip_width': int(strip.shape[1]),
            'analysis_strip_height': int(strip.shape[0]),
            'analysis_memory_mb': round(strip.nbytes / (1024 * 1024), 2),
            'strip_seconds': round(strip_sec, 4),
            'detection_seconds': round(detect_sec, 4),
            'crop_seconds': round(crop_sec, 4),
            'ocr_enabled': bool(use_ocr),
            'ocr_width': int(ocr_width) if use_ocr else 0,
            'ocr_text_regions': len(text_boxes),
            'ocr_assigned_regions': int(ocr_stats.get('assigned', 0)),
            'ocr_panels_expanded': int(ocr_stats.get('expanded_panels', 0)),
            'ocr_text_only_panels_removed': int(ocr_stats.get('text_only_removed', 0)),
            'ocr_seconds': round(ocr_sec, 4),
            'postsplit_panels_split': int(split_stats.get('split_panels', 0)),
            'postsplit_new_panels': int(split_stats.get('new_panels', 0)),
            'postsplit_cuts': int(split_stats.get('cuts', 0)),
            'debug_saved': bool(debug_info.get('files')),
            'debug_chunked': bool(debug_info.get('chunked', False)),
            'debug_files': [x.get('file') for x in debug_info.get('files', [])],
            'debug_errors': list(debug_info.get('errors', [])),
            'total_seconds': round(total_sec, 4),
            'detector_mode': str(diagnostics.get('mode', 'geometry')) + ('+ocr' if use_ocr else ''),
        }
        with open(output_dir / 'report.json', 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        _safe_progress(progress, stage='done', progress=1.0, message='Terminé')
        return report
    finally:
        if temp_ctx is not None:
            temp_ctx.cleanup()
