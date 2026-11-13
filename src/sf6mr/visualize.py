"""Stage 7 helpers: draw boxes and COCO-17 skeletons, and write 60 fps video."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np

# COCO-17 limbs
SKELETON = [
    (15, 13), (13, 11), (16, 14), (14, 12), (11, 12),          # legs, hips
    (5, 11), (6, 12), (5, 6),                                  # torso
    (5, 7), (7, 9), (6, 8), (8, 10),                           # arms
    (1, 2), (0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6),    # head
]
# BGR. P1 pink, P2 blue, like the SF6 HUD.
PLAYER_COLORS = [(180, 60, 255), (255, 160, 0)]
# Left-side joints drawn a bit lighter, so mirrored poses are easy to spot.
_LEFT = {1, 3, 5, 7, 9, 11, 13, 15}


def _lighter(c, f=0.45):
    return tuple(int(v + (255 - v) * f) for v in c)


def draw_frame(
    frame: np.ndarray,
    fighters: np.ndarray | None = None,
    keypoints: np.ndarray | None = None,
    frame_index: int | None = None,
    min_joint_score: float = 0.3,
    labels: tuple[str, str] = ("P1", "P2"),
) -> np.ndarray:
    """Return a copy of ``frame`` with fighter boxes (2, 5) and skeletons (2, 17, 3) drawn."""
    out = frame.copy()
    s = out.shape[0] / 1080  # scale strokes with resolution
    th = max(1, round(3 * s))
    for p in range(2):
        color = PLAYER_COLORS[p]
        if fighters is not None and not np.isnan(fighters[p, 0]):
            x0, y0, x1, y1, sc = fighters[p]
            cv2.rectangle(out, (int(x0), int(y0)), (int(x1), int(y1)), color, max(1, th - 1))
            cv2.putText(out, f"{labels[p]} {sc:.2f}", (int(x0), int(y0) - round(8 * s)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9 * s, color, th, cv2.LINE_AA)
        if keypoints is None or np.isnan(keypoints[p, 0, 0]):
            continue
        kp = keypoints[p]
        ok = kp[:, 2] >= min_joint_score
        for a, b in SKELETON:
            if ok[a] and ok[b]:
                c = _lighter(color) if a in _LEFT and b in _LEFT else color
                cv2.line(out, (int(kp[a, 0]), int(kp[a, 1])), (int(kp[b, 0]), int(kp[b, 1])), c, th + 1, cv2.LINE_AA)
        for j in np.flatnonzero(ok):
            c = _lighter(color) if j in _LEFT else color
            cv2.circle(out, (int(kp[j, 0]), int(kp[j, 1])), th + 2, c, -1, cv2.LINE_AA)
    if frame_index is not None:
        cv2.putText(out, f"{frame_index}", (round(20 * s), out.shape[0] - round(20 * s)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2 * s, (255, 255, 255), th + 1, cv2.LINE_AA)
    return out


def draw_normalized(
    joints: np.ndarray,
    color=(255, 255, 255),
    size: tuple[int, int] = (480, 540),
    px_per_unit: float = 110,
    observed: np.ndarray | None = None,
    title: str = "",
) -> np.ndarray:
    """Draw one normalized skeleton (17, 2+) on a dark panel: hip at the center,
    x = forward (right), y = up, units = torso lengths. Interpolated joints
    (``observed`` False) are hollow. Returns a BGR image of ``size`` (w, h)."""
    w, h = size
    img = np.full((h, w, 3), 24, np.uint8)
    ox, oy = w // 2, int(h * 0.55)
    # axes: torso-length grid
    for k in range(-3, 4):
        cv2.line(img, (ox + int(k * px_per_unit), 0), (ox + int(k * px_per_unit), h), (40, 40, 40), 1)
        cv2.line(img, (0, oy - int(k * px_per_unit)), (w, oy - int(k * px_per_unit)), (40, 40, 40), 1)
    cv2.arrowedLine(img, (ox, h - 22), (ox + 60, h - 22), (120, 120, 120), 2, tipLength=0.3)
    cv2.putText(img, "forward", (ox + 68, h - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120, 120, 120), 1, cv2.LINE_AA)
    if title:
        cv2.putText(img, title, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
    j = np.asarray(joints, float)
    if np.isnan(j[:, 0]).all():
        cv2.putText(img, "missing", (ox - 50, oy), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (90, 90, 90), 2, cv2.LINE_AA)
        return img
    pts = np.stack([ox + j[:, 0] * px_per_unit, oy - j[:, 1] * px_per_unit], axis=1)
    ok = ~np.isnan(pts[:, 0])
    for a, b in SKELETON:
        if ok[a] and ok[b]:
            c = _lighter(color) if a in _LEFT and b in _LEFT else color
            cv2.line(img, tuple(pts[a].astype(int)), tuple(pts[b].astype(int)), c, 3, cv2.LINE_AA)
    for i in np.flatnonzero(ok):
        c = _lighter(color) if i in _LEFT else color
        filled = observed is None or observed[i]
        cv2.circle(img, tuple(pts[i].astype(int)), 5, c, -1 if filled else 2, cv2.LINE_AA)
    return img


class VideoWriter:
    """Write BGR frames to an H.264 MP4 through ffmpeg (falls back to OpenCV's mp4v).

    H.264 plays in browsers, VS Code and QuickTime; OpenCV's mp4v often doesn't.
    """

    def __init__(self, path: str | Path, fps: float, size: tuple[int, int], crf: int = 20):
        self.path = Path(path)
        self.size = size  # (width, height)
        self.proc = None
        self.cv = None
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg:
            w, h = size
            self.proc = subprocess.Popen(
                [ffmpeg, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
                 "-s", f"{w}x{h}", "-r", str(fps), "-i", "-", "-an", "-c:v", "libx264",
                 "-preset", "veryfast", "-crf", str(crf), "-pix_fmt", "yuv420p",
                 "-movflags", "+faststart", str(self.path)],
                stdin=subprocess.PIPE,
            )
        else:
            self.cv = cv2.VideoWriter(str(self.path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)

    def write(self, frame: np.ndarray) -> None:
        if (frame.shape[1], frame.shape[0]) != self.size:
            raise ValueError(f"frame is {frame.shape[1]}x{frame.shape[0]}, writer is {self.size}")
        if self.proc:
            self.proc.stdin.write(np.ascontiguousarray(frame).tobytes())
        else:
            self.cv.write(frame)

    def close(self) -> None:
        if self.proc:
            self.proc.stdin.close()
            if self.proc.wait() != 0:
                raise RuntimeError(f"ffmpeg failed writing {self.path}")
        elif self.cv:
            self.cv.release()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
