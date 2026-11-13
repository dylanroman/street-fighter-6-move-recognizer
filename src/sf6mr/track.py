"""Stage 2 (model-free part): pick the two fighters out of each frame's person
detections and keep their P1/P2 identity across frames.

SF6 stages have spectators who are detected as people, often with high scores,
and the fighters overlap, cross up and disappear behind effects. Three cues:

* **Ground line.** Fighters stand on the foreground floor, so the bottom of a
  grounded fighter's box is near the bottom of the frame. Spectators stand
  behind the fence, so their boxes end higher up. A fighter is only *found*
  (at the start, or after being lost) among boxes on the ground, and a track
  that stays off the ground longer than any jump is dropped: it was a spectator.
* **Continuity.** At 60 fps a fighter moves a few pixels per frame. Each
  tracked fighter is matched to the detection that best fits its predicted box
  (Hungarian assignment over both fighters at once).
* **Appearance.** Each fighter gets a hue/saturation histogram template, built
  from the first colourful frames where the two are clearly apart, then frozen
  (a match keeps the same stage lighting). It adds to the matching cost,
  picks which box to re-find a lost fighter on, and fixes identity swaps: when
  the boxes match the *other* fighter's template for several frames in a row,
  P1 and P2 are swapped back, all the way to the last frame where they were
  confirmed.

P1 is the fighter on the left when both are first found (SF6 rounds start with
P1 on the left). Missing fighters are NaN, never guessed.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

_INF = 1e6
HIST_BINS = (18, 8)  # hue x saturation
HIST_DIM = HIST_BINS[0] * HIST_BINS[1]


@dataclass
class TrackerParams:
    min_score: float = 0.3       # detections below this are ignored
    ground_min: float = 0.85     # find fighters only among boxes whose bottom is below this fraction of frame height
    min_height: float = 0.25     # ... and that are at least this fraction of frame height tall
    max_lost: int = 30           # frames a fighter can go undetected before it must be re-found on the ground
    gate: float = 0.4            # max center distance from the predicted box, as a fraction of the
                                 # fighter's box height (~250 px at 1080p; limbs extending move the center)
    gate_growth: float = 0.1     # the gate grows by this fraction per missed frame ...
    gate_max: float = 0.9        # ... up to this
    max_size_ratio: float = 1.8  # a match's height must be within this factor of the fighter's smoothed height
    max_airborne: int = 40       # frames a track can stay above the ground line before it is dropped
    velocity_smoothing: float = 0.5
    size_smoothing: float = 0.3
    # Appearance
    appearance_weight: float = 1.0   # weight of (1 - histogram similarity) in the matching cost
    reliable_similarity: float = 0.75  # appearance counts as evidence for a box only when it is at least this
                                       # similar to one fighter's template. Under full-screen effects (Supers)
                                       # nothing is, and only position is used.
    min_air_similarity: float = 0.85   # an off-the-ground box must be at least this similar to one fighter's
                                       # template to be matched at all (spectators stand off the ground)
    identity_margin: float = 0.1     # a fighter never takes a box that matches the *other* fighter's
                                     # template this much better than its own, and a re-found box must
                                     # match its own template this much better than the other's
    template_frames: int = 30        # clean frames averaged into each fighter's template; then frozen
    min_colorful: float = 0.5        # template frames need at least this share of saturated pixels in both
                                     # boxes (skips the black-and-white round intro and flashes)
    clean_max_iou: float = 0.05      # "clean" frame: both fighters found and their boxes overlap less than this
    swap_margin: float = 0.15        # per-frame evidence margin for a swap (or for confirming identities)
    swap_frames: int = 6             # consecutive frames of swap evidence needed to swap P1/P2

    @classmethod
    def from_config(cls, cfg: dict | None) -> "TrackerParams":
        cfg = cfg or {}
        unknown = set(cfg) - set(cls.__dataclass_fields__)
        if unknown:
            raise KeyError(f"Unknown tracker settings: {sorted(unknown)}")
        return cls(**cfg)


# ---------------------------------------------------------------- appearance

def color_hist(frame: np.ndarray, box) -> np.ndarray:
    """Normalized hue/saturation histogram of the middle of a box (BGR frame).

    The box is trimmed (20% each side, 10% top, 5% bottom) to cut background,
    and very dark pixels are ignored. Returns float32 (HIST_DIM,), summing to 1
    (all zeros if the crop is empty).
    """
    x0, y0, x1, y1 = (float(v) for v in box[:4])
    w, h = x1 - x0, y1 - y0
    H, W = frame.shape[:2]
    xa, xb = int(max(x0 + 0.2 * w, 0)), int(min(x1 - 0.2 * w, W))
    ya, yb = int(max(y0 + 0.1 * h, 0)), int(min(y1 - 0.05 * h, H))
    if xb <= xa or yb <= ya:
        return np.zeros(HIST_DIM, np.float32)
    hsv = cv2.cvtColor(frame[ya:yb, xa:xb], cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (0, 0, 40), (180, 255, 255))
    hist = cv2.calcHist([hsv], [0, 1], mask, list(HIST_BINS), [0, 180, 0, 256]).ravel()
    s = hist.sum()
    return (hist / s if s > 0 else hist).astype(np.float32)


def _colorful(hist: np.ndarray) -> float:
    """Share of a histogram's mass outside the lowest-saturation bin (0 for a grey image)."""
    return float(np.asarray(hist).reshape(HIST_BINS)[:, 1:].sum())


def similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Bhattacharyya coefficient between normalized histograms (1 = identical).
    Broadcasts over leading axes."""
    return np.sqrt(np.clip(a, 0, None) * np.clip(b, 0, None)).sum(-1)


# ---------------------------------------------------------------- geometry

def iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU between boxes ``a`` (n, 4) and ``b`` (m, 4) in xyxy; returns (n, m)."""
    a = np.asarray(a, float).reshape(-1, 4)
    b = np.asarray(b, float).reshape(-1, 4)
    x0 = np.maximum(a[:, None, 0], b[None, :, 0])
    y0 = np.maximum(a[:, None, 1], b[None, :, 1])
    x1 = np.minimum(a[:, None, 2], b[None, :, 2])
    y1 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x1 - x0, 0, None) * np.clip(y1 - y0, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-9), 0.0)


def _center(box: np.ndarray) -> np.ndarray:
    return np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2])


# ---------------------------------------------------------------- tracker

class _Track:
    def __init__(self) -> None:
        self.box: np.ndarray | None = None  # last matched box (4,)
        self.velocity = np.zeros(2)          # px/frame of the box center
        self.height = 0.0                    # smoothed box height; resists one-frame fragments
        self.lost = 0                        # frames since last match
        self.airborne = 0                    # consecutive matched frames above the ground line

    @property
    def active(self) -> bool:
        return self.box is not None

    def predicted(self) -> np.ndarray:
        # Extrapolate at most a few frames: a fighter that vanished (cinematic,
        # overlap) is more likely to reappear near where it was than far along its path.
        shift = self.velocity * min(self.lost + 1, 5)
        return self.box + np.array([shift[0], shift[1], shift[0], shift[1]])

    def update(self, box: np.ndarray, p: TrackerParams) -> None:
        box = np.asarray(box, float)
        h = box[3] - box[1]
        if self.box is not None and self.lost == 0:
            v = _center(box) - _center(self.box)
            self.velocity = p.velocity_smoothing * v + (1 - p.velocity_smoothing) * self.velocity
        else:  # first sighting, or re-found after a gap
            self.velocity = np.zeros(2)
        if self.box is None:
            self.height = h
        else:
            self.height = p.size_smoothing * h + (1 - p.size_smoothing) * self.height
        self.box = box.copy()
        self.lost = 0


class _Appearance:
    """Per-fighter histogram templates and the swap check."""

    def __init__(self, p: TrackerParams) -> None:
        self.p = p
        self.sum = np.zeros((2, HIST_DIM))
        self.n = 0
        self.templates: np.ndarray | None = None
        self.swap_run = 0         # consecutive frames of swap evidence
        self.last_confirmed = -1  # last frame where identities matched the templates

    @property
    def ready(self) -> bool:
        return self.templates is not None

    def observe(self, t: int, h0: np.ndarray, h1: np.ndarray, clean: bool) -> bool:
        """Feed the two fighters' histograms for frame t. Returns True if P1/P2 should swap."""
        p = self.p
        if not self.ready:
            if clean and min(_colorful(h0), _colorful(h1)) >= p.min_colorful:
                self.sum += np.stack([h0, h1])
                self.n += 1
                if self.n >= p.template_frames:
                    self.templates = self.sum / self.n
                    self.last_confirmed = t
            return False
        s = similarity(np.stack([h0, h1])[:, None, :], self.templates[None])  # (box, template)
        if s[0].max() < p.reliable_similarity or s[1].max() < p.reliable_similarity:
            return False  # colours unreliable (effects): no evidence either way
        keep = s[0, 0] + s[1, 1]
        swap = s[0, 1] + s[1, 0]
        if keep > swap + p.swap_margin:
            self.swap_run = 0
            self.last_confirmed = t
        elif swap > keep + p.swap_margin:
            self.swap_run += 1
            if self.swap_run >= p.swap_frames:
                self.swap_run = 0
                return True
        else:
            self.swap_run = 0
        return False


def _on_ground(boxes: np.ndarray, frame_h: int, p: TrackerParams) -> np.ndarray:
    h = boxes[:, 3] - boxes[:, 1]
    return (boxes[:, 3] >= p.ground_min * frame_h) & (h >= p.min_height * frame_h)


def assign_fighters(
    detections: list[np.ndarray],
    frame_h: int,
    params: TrackerParams | None = None,
    features: list[np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Pick P1/P2 boxes in every frame.

    Args:
        detections: one array per frame, shape (n_t, 5): x0, y0, x1, y1, score.
        frame_h: frame height in pixels (for the ground-line cue).
        features: optional, one (n_t, HIST_DIM) array per frame of ``color_hist``
            for each detection. Without it, only position is used.

    Returns:
        fighters: (T, 2, 5) float32, x0, y0, x1, y1, score per player; NaN when missing.
        det_index: (T, 2) int, row of ``detections[t]`` used for each player, -1 when missing.
    """
    p = params or TrackerParams()
    T = len(detections)
    fighters = np.full((T, 2, 5), np.nan, dtype=np.float32)
    det_index = np.full((T, 2), -1, dtype=int)
    tracks = [_Track(), _Track()]
    app = _Appearance(p) if features is not None else None

    for t, dets in enumerate(detections):
        dets = np.asarray(dets, float).reshape(-1, 5)
        cand = np.flatnonzero(dets[:, 4] >= p.min_score)
        boxes = dets[cand, :4]
        hists = np.asarray(features[t], float).reshape(-1, HIST_DIM)[cand] if app else None
        templates = app.templates if app and app.ready else None
        taken: dict[int, int] = {}  # player -> candidate position

        # 1) Follow active tracks.
        live = [i for i, tr in enumerate(tracks) if tr.active and tr.lost <= p.max_lost]
        if live and len(boxes):
            hts = boxes[:, 3] - boxes[:, 1]
            grounded = boxes[:, 3] >= p.ground_min * frame_h
            sims = reliable = None
            if templates is not None:
                sims = similarity(hists[:, None, :], templates[None])  # (n, 2)
                reliable = sims.max(1) >= p.reliable_similarity
            cost = np.full((len(live), len(boxes)), _INF)
            for r, i in enumerate(live):
                tr = tracks[i]
                pred = tr.predicted()
                ref_h = max(tr.height, 1.0)
                dist = np.linalg.norm((boxes[:, :2] + boxes[:, 2:]) / 2 - _center(pred), axis=1) / ref_h
                ratio = hts / ref_h
                gate = min(p.gate * (1 + p.gate_growth * tr.lost), p.gate_max)
                ok = (dist <= gate) & (ratio <= p.max_size_ratio) & (ratio >= 1 / p.max_size_ratio)
                c = (1 - iou(pred[None], boxes)[0]) + dist + 0.5 * (1 - dets[cand, 4]) + 0.5 * np.abs(np.log(ratio))
                if sims is not None:
                    c = c + p.appearance_weight * (1 - sims[:, i])
                    # Off the ground, only boxes that clearly look like *this* fighter
                    # (spectators stand off the ground line and look like neither).
                    ok &= grounded | (sims[:, i] >= p.min_air_similarity)
                    # Never take a box that reliably looks like the other fighter.
                    ok &= ~(reliable & (sims[:, 1 - i] > sims[:, i] + p.identity_margin))
                cost[r, ok] = c[ok]
            rows, cols = linear_sum_assignment(cost)
            for r, c in zip(rows, cols):
                if cost[r, c] < _INF:
                    taken[live[r]] = c

        # 2) (Re)acquire missing fighters among unused boxes standing on the ground.
        need = [i for i in range(2) if i not in taken and (not tracks[i].active or tracks[i].lost >= p.max_lost)]
        if need and len(boxes):
            used = set(taken.values())
            ground = [c for c in np.flatnonzero(_on_ground(boxes, frame_h, p)) if c not in used]
            if not any(tr.active for tr in tracks) and templates is None:
                # Start of tracking: need both fighters; P1 = left, P2 = right.
                if len(ground) >= 2:
                    hts = boxes[ground, 3] - boxes[ground, 1]
                    two = sorted((ground[k] for k in np.argsort(-hts)[:2]), key=lambda c: boxes[c, 0] + boxes[c, 2])
                    taken[0], taken[1] = two
            else:
                for i in need:
                    if not ground:
                        break
                    if templates is not None:
                        # Best appearance match, and it must look more like this fighter than the other.
                        own = similarity(hists[ground], templates[i])
                        other = similarity(hists[ground], templates[1 - i])
                        k = int(np.argmax(own - other))
                        if own[k] < max(other[k] + p.identity_margin, p.reliable_similarity):
                            continue
                        c = ground[k]
                    elif (1 - i) in taken:
                        oc = _center(boxes[taken[1 - i]])  # farthest from the other fighter
                        c = max(ground, key=lambda g: np.linalg.norm(_center(boxes[g]) - oc))
                    elif tracks[i].active:
                        lc = _center(tracks[i].box)       # nearest to where it was last seen
                        c = min(ground, key=lambda g: np.linalg.norm(_center(boxes[g]) - lc))
                    else:
                        c = max(ground, key=lambda g: boxes[g, 3] - boxes[g, 1])
                    taken[i] = c
                    ground.remove(c)

        # 3) Update tracks and write outputs.
        for i, tr in enumerate(tracks):
            if i not in taken:
                if tr.active:
                    tr.lost += 1
                continue
            c = taken[i]
            tr.update(boxes[c], p)
            tr.airborne = 0 if boxes[c, 3] >= p.ground_min * frame_h else tr.airborne + 1
            if tr.airborne > p.max_airborne:
                # Off the floor for too long: a spectator, not a fighter. Back-fill NaN
                # over the whole airborne stretch and re-find the fighter.
                fighters[t - tr.airborne + 1 : t + 1, i] = np.nan
                det_index[t - tr.airborne + 1 : t + 1, i] = -1
                tr.__init__()
                del taken[i]
                continue
            fighters[t, i] = dets[cand[c]].astype(np.float32)
            det_index[t, i] = cand[c]

        # 4) Appearance: learn templates on clean frames and undo identity swaps.
        if app and 0 in taken and 1 in taken:
            b0, b1 = boxes[taken[0]], boxes[taken[1]]
            clean = iou(b0[None], b1[None])[0, 0] < p.clean_max_iou
            if app.observe(t, hists[taken[0]], hists[taken[1]], clean):
                s = app.last_confirmed + 1
                fighters[s : t + 1] = fighters[s : t + 1, ::-1].copy()
                det_index[s : t + 1] = det_index[s : t + 1, ::-1].copy()
                tracks.reverse()
                app.last_confirmed = t
    return fighters, det_index


def side_of_p1(fighters: np.ndarray) -> np.ndarray:
    """Per frame: -1 if P1 is left of P2, +1 if right, 0 if either is missing."""
    cx = (fighters[:, :, 0] + fighters[:, :, 2]) / 2
    d = cx[:, 0] - cx[:, 1]
    return np.where(np.isnan(d), 0, np.sign(d)).astype(int)
