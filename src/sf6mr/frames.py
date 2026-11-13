"""Stage 1: video reading and the optional HUD mask. No models here.

Frames are read in order with OpenCV and never skipped, so frame ``i`` of every
stage output is frame ``i`` (0-based) of the source video.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np


@dataclass
class VideoInfo:
    path: str
    fps: float
    width: int
    height: int
    frame_count: int  # counted by decoding every frame, not from the container header

    def to_dict(self) -> dict:
        return asdict(self)


def count_frames(path: str | Path) -> int:
    """Decode the whole video and count frames (container frame counts can be off)."""
    cap = cv2.VideoCapture(str(path))
    n = 0
    while cap.grab():
        n += 1
    cap.release()
    return n


def read_video_info(path: str | Path, expected_fps: float | None = 60) -> VideoInfo:
    """Read fps and size, and count frames. Raises if the fps isn't ``expected_fps``."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"OpenCV can't open {path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    if expected_fps is not None and abs(fps - expected_fps) > 0.01:
        raise ValueError(f"{path} is {fps:.3f} fps; the pipeline needs {expected_fps} fps footage")
    return VideoInfo(str(path), fps, w, h, count_frames(path))


def iter_frames(path: str | Path) -> Iterator[tuple[int, np.ndarray]]:
    """Yield ``(frame_index, BGR frame)`` for every frame, in order."""
    cap = cv2.VideoCapture(str(path))
    i = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                return
            yield i, frame
            i += 1
    finally:
        cap.release()


def hud_rects_px(regions: list, width: int, height: int) -> np.ndarray:
    """Convert fractional ``[x0, y0, x1, y1]`` regions to integer pixel rects, shape (n, 4)."""
    if not regions:
        return np.zeros((0, 4), dtype=int)
    r = np.asarray(regions, dtype=float).reshape(-1, 4)
    if (r < 0).any() or (r > 1).any() or (r[:, 2] <= r[:, 0]).any() or (r[:, 3] <= r[:, 1]).any():
        raise ValueError(f"HUD regions must be fractions with x0 < x1 and y0 < y1, got {regions}")
    scale = np.array([width, height, width, height], dtype=float)
    return np.round(r * scale).astype(int)


def apply_hud_mask(frame: np.ndarray, rects_px: np.ndarray) -> np.ndarray:
    """Return a copy of ``frame`` with each rect filled black (input unchanged)."""
    if len(rects_px) == 0:
        return frame
    out = frame.copy()
    for x0, y0, x1, y1 in rects_px:
        out[y0:y1, x0:x1] = 0
    return out
