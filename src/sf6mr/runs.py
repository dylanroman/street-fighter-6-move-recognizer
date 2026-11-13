"""Per-video stage outputs in ``data/runs/<video_stem>/``: paths, save and load.

File contracts (frame index = 0-based source frame, never skipped):

frames.json
    path (of the video), fps, width, height, frame_count (counted by decoding),
    hud_mask {enabled, regions (fractions), regions_px}.

detections.npz
    Raw person detections, one row per box (ragged over frames):
      det_frame  (N,)      int32    frame index of each box
      det_boxes  (N, 5)    float32  x0, y0, x1, y1, score (pixels)
      det_hist   (N, 144)  float16  sf6mr.track.color_hist of each box
    The chosen fighters:
      fighters   (T, 2, 5) float32  x0, y0, x1, y1, score for P1, P2; NaN when missing
      det_index  (T, 2)    int32    row in det_* used for each player, -1 when missing
      tracker    ()        str      JSON of the TrackerParams used

keypoints.npz
      keypoints  (T, 2, 17, 3) float32  x, y (pixels), score per COCO-17 joint;
                                        NaN when that player is missing in that frame.
    Scores are RTMPose SimCC confidences: higher is better, but they are not
    probabilities and can exceed 1.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .config import resolve_path


def run_dir(cfg: dict, video: str | Path, create: bool = True) -> Path:
    d = resolve_path(cfg.get("runs_dir", "data/runs")) / Path(video).stem
    if create:
        d.mkdir(parents=True, exist_ok=True)
    return d


# ---------------------------------------------------------------- frames.json

def save_frames_json(path: Path, info: dict) -> None:
    Path(path).write_text(json.dumps(info, indent=2) + "\n")


def load_frames_json(path: Path) -> dict:
    return json.loads(Path(path).read_text())


# ---------------------------------------------------------------- detections.npz

def save_detections(
    path: Path,
    det_frame: np.ndarray,
    det_boxes: np.ndarray,
    det_hist: np.ndarray,
    fighters: np.ndarray,
    det_index: np.ndarray,
    tracker_params=None,
) -> None:
    np.savez_compressed(
        path,
        det_frame=np.asarray(det_frame, np.int32),
        det_boxes=np.asarray(det_boxes, np.float32).reshape(-1, 5),
        det_hist=np.asarray(det_hist, np.float16).reshape(len(det_frame), -1),
        fighters=np.asarray(fighters, np.float32),
        det_index=np.asarray(det_index, np.int32),
        tracker=np.array(json.dumps(asdict(tracker_params) if tracker_params is not None else {})),
    )


def load_detections(path: Path) -> dict:
    with np.load(path) as z:
        out = {k: z[k] for k in z.files}
    out["tracker"] = json.loads(str(out["tracker"]))
    return out


def split_by_frame(det_frame: np.ndarray, values: np.ndarray, n_frames: int) -> list[np.ndarray]:
    """Turn flat per-box rows into one array per frame (frames with no boxes get an empty array)."""
    det_frame = np.asarray(det_frame)
    order = np.argsort(det_frame, kind="stable")
    counts = np.bincount(det_frame, minlength=n_frames)[:n_frames]
    return np.split(np.asarray(values)[order], np.cumsum(counts)[:-1])


def to_flat_index(det_frame: np.ndarray, per_frame_index: np.ndarray) -> np.ndarray:
    """Convert (T, 2) row positions *within each frame* (what assign_fighters returns
    for ``split_by_frame`` input) to rows of the flat ``det_*`` arrays. -1 stays -1.
    ``det_frame`` must be sorted, as the detection loop writes it."""
    det_frame = np.asarray(det_frame)
    if np.any(np.diff(det_frame) < 0):
        raise ValueError("det_frame must be sorted by frame")
    starts = np.searchsorted(det_frame, np.arange(per_frame_index.shape[0]))
    return np.where(per_frame_index >= 0, per_frame_index + starts[:, None], -1).astype(np.int32)


# ---------------------------------------------------------------- keypoints.npz

def save_keypoints(path: Path, keypoints: np.ndarray) -> None:
    kp = np.asarray(keypoints, np.float32)
    if kp.ndim != 4 or kp.shape[1:] != (2, 17, 3):
        raise ValueError(f"keypoints must be (T, 2, 17, 3), got {kp.shape}")
    np.savez_compressed(path, keypoints=kp)


def load_keypoints(path: Path) -> np.ndarray:
    with np.load(path) as z:
        return z["keypoints"]
