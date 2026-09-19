from __future__ import annotations

from pathlib import Path
from typing import Sequence

import cv2
import numpy as np


def imread_unicode(path: str | Path, flags: int = cv2.IMREAD_COLOR):
    """Unicode-safe OpenCV image read on Windows.

    cv2.imread can fail on Windows when a path contains characters such as
    curly apostrophes, accents or non-Latin characters. Reading bytes with
    numpy.fromfile then decoding avoids that limitation.
    """
    p = str(path)
    try:
        data = np.fromfile(p, dtype=np.uint8)
        if data.size == 0:
            return None
        return cv2.imdecode(data, flags)
    except (OSError, ValueError):
        return None


def imwrite_unicode(path: str | Path, image: np.ndarray,
                    params: Sequence[int] | None = None) -> bool:
    """Unicode-safe OpenCV image write on Windows.

    Encodes in memory, then writes the encoded bytes with ndarray.tofile,
    which supports Unicode Windows paths reliably.
    """
    p = Path(path)
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        ext = p.suffix.lower() or '.png'
        ok, encoded = cv2.imencode(ext, image, list(params or []))
        if not ok:
            return False
        encoded.tofile(str(p))
        return p.exists() and p.stat().st_size > 0
    except (OSError, ValueError, cv2.error):
        return False
