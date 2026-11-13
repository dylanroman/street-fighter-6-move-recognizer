"""Stage 4: turn pixel keypoints into a canonical, body-relative skeleton per fighter.

Input: ``keypoints (T, 2, 17, 3)`` from stage 3 (x, y in pixels, RTMPose score).
Steps, per player:

1. **Low-confidence joints** (score < ``min_score``) become NaN.
2. **Left/right flips.** In side views RTMPose sometimes swaps a left/right pair
   (e.g. the two knees) for a frame or a few. Per pair, a Viterbi pass picks
   "as given" or "swapped" for every frame, trading joint motion against a
   small cost for overriding RTMPose (see ``fix_lr_flips``).
3. **Gaps** of at most ``max_gap`` frames are filled by linear interpolation,
   per joint, including frames where the whole fighter was missing. Longer gaps
   stay NaN. Interpolated joints get score 0 and ``observed = False``.
4. **Smoothing:** Savitzky-Golay over each run of valid frames (light, so a
   move's startup isn't blurred).
5. **Facing** from the two fighters' hip positions: a fighter faces the
   opponent, but (like SF6) only turns around while on the ground, so a
   cross-up jump doesn't flip the skeleton mid-air.
6. **Normalize:** subtract the hip midpoint, divide by torso length (rolling
   median, so bending forward doesn't shrink the skeleton), flip y so it points
   up, and for fighters facing left mirror x and swap left/right joints. After
   this every skeleton faces +x: **x = forward, y = up, units = torso lengths.**

Hip center, torso scale, facing and a grounded flag are saved too, so later
stages can add whole-body motion (jumps, dashes) back in or map forward/back to
screen directions.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

# COCO-17: index of the mirror-image joint (left <-> right; nose stays)
FLIP_INDEX = np.array([0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15])
LR_PAIRS = [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16)]
# Left/right limb chains that are swapped as a unit when fixing flips: head, arms, legs.
LR_CHAINS = [((1, 3), (2, 4)), ((5, 7, 9), (6, 8, 10)), ((11, 13, 15), (12, 14, 16))]
L_SHOULDER, R_SHOULDER, L_HIP, R_HIP, L_ANKLE, R_ANKLE = 5, 6, 11, 12, 15, 16


@dataclass
class NormalizeParams:
    min_score: float = 0.3        # joints below this RTMPose score are dropped (then maybe interpolated)
    fix_flips: bool = True
    flip_penalty: float = 0.4     # cost (torso lengths) per frame of overriding RTMPose's left/right labels
    max_gap: int = 10             # longest gap (frames) filled by interpolation; 10 frames = 1/6 s
    smooth_window: int = 5        # Savitzky-Golay window (odd); < 3 disables smoothing
    smooth_order: int = 2
    scale_window: int = 61        # frames in the rolling median of torso length
    ground_min: float = 0.82      # grounded: lowest ankle below this fraction of frame height
    facing_margin: float = 0.25   # opponent must be this many torso lengths past you before you turn

    @classmethod
    def from_config(cls, cfg: dict | None) -> "NormalizeParams":
        cfg = cfg or {}
        unknown = set(cfg) - set(cls.__dataclass_fields__)
        if unknown:
            raise KeyError(f"Unknown normalize settings: {sorted(unknown)}")
        return cls(**cfg)


# ---------------------------------------------------------------- building blocks

def mirror(joints: np.ndarray) -> np.ndarray:
    """Mirror skeletons about x = 0 and swap left/right joints.

    ``joints`` (..., 17, C) with x in channel 0 (other channels are carried along).
    Applying it twice gives the input back.
    """
    out = np.asarray(joints)[..., FLIP_INDEX, :].copy()
    out[..., 0] = -out[..., 0]
    return out


def fix_lr_flips(joints: np.ndarray, scale: np.ndarray, penalty: float = 0.4):
    """Undo short left/right label swaps for one fighter.

    Works on whole limb chains (``LR_CHAINS``: head, arms, legs), so an arm's
    shoulder, elbow and wrist always swap together. For each chain, every frame is
    either "as given" or "swapped". The labeling chosen minimizes (Viterbi, over
    each run of frames where the whole chain is present):

        sum of frame-to-frame joint motion (torso lengths) + ``penalty`` per swapped frame

    so a swap is undone when it removes a jump bigger than it costs, and RTMPose's
    labels win when it's unclear (e.g. legs crossing while walking). A swap lasting
    k frames is undone only if the two jumps it causes are worth more than
    ``penalty * k``: short glitches are fixed, long consistent labelings are kept.

    Args:
        joints: (T, 17, C) with pixel x, y in channels 0-1 (NaN = missing); other
            channels (e.g. score) move with their joint.
        scale: (T,) torso length in pixels.
    Returns:
        (fixed copy of joints, number of (frame, chain) swaps made)
    """
    out = np.array(joints, dtype=float, copy=True)
    s = np.where(np.isfinite(scale) & (scale > 0), scale, np.nan)
    n_fixed = 0
    for left, right in LR_CHAINS:
        l, r = list(left), list(right)
        present = ~np.isnan(out[:, l + r, :2]).any(axis=(1, 2)) & np.isfinite(s)
        edges = np.diff(np.r_[0, present.astype(int), 0])
        for a, b in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
            if b - a < 2:
                continue
            L, R = out[a:b, l, :2], out[a:b, r, :2]  # (n, k, 2)
            sc = s[a + 1 : b]
            # motion cost between consecutive frames for (prev state, cur state); 0 = as given, 1 = swapped
            d = lambda p, q: np.linalg.norm(p - q, axis=-1).sum(-1)
            same = (d(L[1:], L[:-1]) + d(R[1:], R[:-1])) / sc   # states equal
            cross = (d(L[1:], R[:-1]) + d(R[1:], L[:-1])) / sc  # states differ
            n = b - a
            cost = np.array([0.0, penalty])
            back = np.zeros((n, 2), int)
            for t in range(1, n):
                to0 = cost + np.array([same[t - 1], cross[t - 1]])   # from (0, 1) into 0
                to1 = cost + np.array([cross[t - 1], same[t - 1]])   # from (0, 1) into 1
                back[t] = [np.argmin(to0), np.argmin(to1)]
                cost = np.array([to0.min(), to1.min() + penalty])
            state = np.zeros(n, int)
            state[-1] = int(np.argmin(cost))
            for t in range(n - 1, 0, -1):
                state[t - 1] = back[t, state[t]]
            idx = a + np.flatnonzero(state == 1)
            out[idx[:, None], l + r] = out[idx[:, None], r + l]
            n_fixed += len(idx)
    return out, n_fixed


def interpolate_gaps(values: np.ndarray, max_gap: int) -> np.ndarray:
    """Linearly fill NaN runs of at most ``max_gap`` frames along axis 0.

    Only interior gaps are filled (no extrapolation at the ends). Works on any
    trailing shape; each trailing element is its own series.
    """
    v = np.asarray(values, dtype=float)
    flat = v.reshape(len(v), -1)
    df = pd.DataFrame(flat)
    filled = df.interpolate(method="linear", limit=max_gap, limit_area="inside", axis=0).to_numpy()
    # pandas fills the first max_gap frames of a longer gap; undo that: only fill
    # gaps whose total length is <= max_gap.
    nan = np.isnan(flat)
    for j in range(flat.shape[1]):
        col = nan[:, j]
        if not col.any():
            continue
        edges = np.diff(np.r_[0, col.astype(int), 0])
        for a, b in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
            if b - a > max_gap:
                filled[a:b, j] = np.nan
    return filled.reshape(v.shape)


def smooth(values: np.ndarray, window: int, order: int) -> np.ndarray:
    """Savitzky-Golay smoothing along axis 0, applied to each run of non-NaN frames
    that is at least ``window`` long (shorter runs are left as they are)."""
    v = np.array(values, dtype=float, copy=True)
    if window < 3:
        return v
    if window % 2 == 0:
        raise ValueError("smooth_window must be odd")
    flat = v.reshape(len(v), -1)
    for j in range(flat.shape[1]):
        ok = ~np.isnan(flat[:, j])
        edges = np.diff(np.r_[0, ok.astype(int), 0])
        for a, b in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
            if b - a >= window:
                flat[a:b, j] = savgol_filter(flat[a:b, j], window, order)
    return flat.reshape(v.shape)


def torso_length(xy: np.ndarray) -> np.ndarray:
    """Shoulder-midpoint to hip-midpoint distance, (T,) for xy (T, 17, 2)."""
    sh = (xy[:, L_SHOULDER] + xy[:, R_SHOULDER]) / 2
    hp = (xy[:, L_HIP] + xy[:, R_HIP]) / 2
    return np.linalg.norm(sh - hp, axis=-1)


def rolling_scale(per_frame: np.ndarray, window: int) -> np.ndarray:
    """Rolling median of per-frame torso length, filled forward/backward over gaps."""
    s = pd.Series(per_frame).rolling(window, center=True, min_periods=1).median()
    return s.ffill().bfill().to_numpy()


def estimate_facing(
    hip_x: np.ndarray, grounded: np.ndarray, scale: np.ndarray, margin: float = 0.25
) -> np.ndarray:
    """Which way each fighter faces: +1 right, -1 left, 0 unknown.

    Args:
        hip_x: (T, 2) hip-center x per player (pixels, NaN = unknown).
        grounded: (T, 2) bool.
        scale: (T, 2) torso length in pixels.
    A fighter faces the opponent. It turns around only on a frame where it is
    grounded and the opponent is more than ``margin`` torso lengths on the other
    side. Before both are first seen, facing is taken from the first frame where
    they are. Frames before any information stay 0.
    """
    T = len(hip_x)
    facing = np.zeros((T, 2), np.int8)
    dx = hip_x[:, ::-1] - hip_x  # opponent minus self, per player
    for p in range(2):
        known = np.flatnonzero(np.isfinite(dx[:, p]) & (np.abs(dx[:, p]) > 0))
        if len(known) == 0:
            continue
        cur = int(np.sign(dx[known[0], p]))
        facing[: known[0], p] = cur
        for t in range(known[0], T):
            d = dx[t, p]
            if (
                np.isfinite(d)
                and np.sign(d) != cur
                and abs(d) > margin * scale[t, p]
                and grounded[t, p]
            ):
                cur = int(np.sign(d))
            facing[t, p] = cur
    return facing


# ---------------------------------------------------------------- whole stage

def normalize_sequence(keypoints: np.ndarray, frame_h: int, params: NormalizeParams | None = None) -> dict:
    """Run the whole stage on ``keypoints (T, 2, 17, 3)``. Returns a dict with:

    joints   (T, 2, 17, 3) float32  x forward, y up (torso lengths, hip-centered), score
                                    (0 for interpolated joints); NaN where unavailable
    observed (T, 2, 17)    bool     joint came from RTMPose (not interpolated)
    center   (T, 2, 2)     float32  hip midpoint in pixels
    scale    (T, 2)        float32  torso length in pixels (rolling median)
    facing   (T, 2)        int8     +1 right, -1 left, 0 unknown
    grounded (T, 2)        bool     lowest ankle on the ground line
    lr_fixes (2,)          int      left/right pair swaps undone per player
    """
    p = params or NormalizeParams()
    kp = np.asarray(keypoints, dtype=float)
    if kp.ndim != 4 or kp.shape[1:] != (2, 17, 3):
        raise ValueError(f"keypoints must be (T, 2, 17, 3), got {kp.shape}")
    T = len(kp)

    kp = kp.copy()
    low = ~(kp[..., 2] >= p.min_score)  # also True where the score is NaN
    kp[low] = np.nan

    lr_fixes = np.zeros(2, int)
    if p.fix_flips:
        for q in range(2):
            raw_scale = rolling_scale(torso_length(kp[:, q, :, :2]), p.scale_window)
            kp[:, q], lr_fixes[q] = fix_lr_flips(kp[:, q], raw_scale, p.flip_penalty)

    xy, score = kp[..., :2], kp[..., 2]
    observed = ~np.isnan(xy[..., 0])
    xy = interpolate_gaps(xy, p.max_gap)
    xy = smooth(xy, p.smooth_window, p.smooth_order)
    score = np.where(observed, score, np.where(np.isnan(xy[..., 0]), np.nan, 0.0))

    center = (xy[:, :, L_HIP] + xy[:, :, R_HIP]) / 2  # (T, 2, 2)
    scale = np.stack([rolling_scale(torso_length(xy[:, q]), p.scale_window) for q in range(2)], axis=1)
    lowest_ankle = np.fmax(xy[:, :, L_ANKLE, 1], xy[:, :, R_ANKLE, 1])
    grounded = lowest_ankle >= p.ground_min * frame_h  # NaN compares False
    facing = estimate_facing(center[..., 0], grounded, scale, p.facing_margin)

    joints = np.empty((T, 2, 17, 3))
    joints[..., :2] = (xy - center[:, :, None, :]) / scale[:, :, None, None]
    joints[..., 1] *= -1  # y up
    joints[..., 2] = score
    left = facing == -1
    joints[left] = mirror(joints[left])
    obs = observed.copy()
    obs[left] = obs[left][:, FLIP_INDEX]

    return {
        "joints": joints.astype(np.float32),
        "observed": obs,
        "center": center.astype(np.float32),
        "scale": scale.astype(np.float32),
        "facing": facing,
        "grounded": grounded,
        "lr_fixes": lr_fixes,
    }

