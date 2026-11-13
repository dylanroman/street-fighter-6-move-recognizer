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
