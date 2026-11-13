"""Normalization on synthetic skeletons: mirroring, left/right swap, scaling, plus
gap filling, smoothing, facing and left/right flip repair."""

import numpy as np
import pytest

from sf6mr import normalize as nz
from sf6mr.normalize import FLIP_INDEX, NormalizeParams, mirror, normalize_sequence

H = 1080

# A canonical fighter in normalized units (x forward, y up, hip midpoint at 0, torso = 1),
# asymmetric on purpose: lead (right) arm punching forward, left leg forward.
CANON = np.array([
    [0.15, 1.45],  # 0 nose
    [0.20, 1.52], [0.10, 1.52],   # 1, 2 eyes
    [0.12, 1.45], [0.00, 1.45],   # 3, 4 ears
    [0.15, 1.00], [-0.15, 1.00],  # 5, 6 shoulders  (torso: shoulder mid (0, 1) to hip mid (0, 0))
    [0.30, 0.60], [0.40, 1.00],   # 7, 8 elbows
    [0.35, 0.30], [0.85, 1.05],   # 9, 10 wrists
    [0.10, 0.00], [-0.10, 0.00],  # 11, 12 hips
    [0.45, -0.70], [-0.30, -0.75],  # 13, 14 knees
    [0.50, -1.45], [-0.45, -1.50],  # 15, 16 ankles (feet at y ~ -1.5)
])


def to_pixels(canon, hip_x, torso_px, facing, floor_y=1000):
    """Place a normalized pose in the image: hip at hip_x, feet on floor_y.
    A fighter facing left is the mirror image (x negated, left/right labels swapped)."""
    pose = canon if facing == 1 else mirror(np.c_[canon, np.zeros(17)])[:, :2]
    hip_y = floor_y - 1.5 * torso_px
    return np.c_[hip_x + pose[:, 0] * torso_px, hip_y - pose[:, 1] * torso_px]


def make_keypoints(p1_x, p2_x, torso=(200.0, 180.0), facing=(1, -1), score=0.9):
    """(T, 2, 17, 3) for per-frame hip positions of both players."""
    T = len(p1_x)
    kp = np.empty((T, 2, 17, 3))
    for t in range(T):
        for p, xs in enumerate((p1_x, p2_x)):
            kp[t, p, :, :2] = to_pixels(CANON, xs[t], torso[p], facing[p] if np.ndim(facing) == 1 else facing[t][p])
    kp[..., 2] = score
    return kp


# ---------------------------------------------------------------- required: mirroring, L/R swap, scaling

def test_mirror_negates_x_and_swaps_left_right():
    j = np.c_[CANON, np.arange(17)]  # third channel tags each joint
    m = mirror(j)
    assert np.allclose(m[:, 0], -CANON[FLIP_INDEX, 0])
    assert np.allclose(m[:, 1], CANON[FLIP_INDEX, 1])
    assert m[5, 2] == 6 and m[6, 2] == 5 and m[15, 2] == 16 and m[0, 2] == 0  # labels travel
    assert np.allclose(mirror(m), j)  # involution
    assert sorted(FLIP_INDEX.tolist()) == list(range(17))


def test_both_fighters_come_out_facing_forward():
    # P1 on the left facing right, P2 on the right facing left (a mirrored pose).
    T = 20
    kp = make_keypoints(np.full(T, 500.0), np.full(T, 1400.0))
    out = normalize_sequence(kp, H, NormalizeParams(smooth_window=0))
    assert (out["facing"][:, 0] == 1).all() and (out["facing"][:, 1] == -1).all()
    # Both normalize back to the same canonical pose: mirrored and L/R-swapped correctly.
    assert np.allclose(out["joints"][:, 0, :, :2], CANON, atol=1e-6)
    assert np.allclose(out["joints"][:, 1, :, :2], CANON, atol=1e-6)


def test_scale_and_position_invariance():
    T = 15
    kp = make_keypoints(np.full(T, 300.0), np.full(T, 1500.0), torso=(150.0, 260.0))
    out = normalize_sequence(kp, H, NormalizeParams(smooth_window=0, ground_min=0.5))
    assert np.allclose(out["scale"][:, 0], 150) and np.allclose(out["scale"][:, 1], 260)
    assert np.allclose(out["center"][:, 0, 0], 300) and np.allclose(out["center"][:, 1, 0], 1500)
    assert np.allclose(out["joints"][..., :2], CANON, atol=1e-6)
    # hip midpoint at the origin, torso length 1, y up (head above hips)
    j = out["joints"][0, 0]
    assert np.allclose((j[11, :2] + j[12, :2]) / 2, 0, atol=1e-6)
    assert nz.torso_length(out["joints"][:1, 0, :, :2])[0] == pytest.approx(1)
    assert j[0, 1] > 0


# ---------------------------------------------------------------- facing

def test_facing_switches_only_on_the_ground():
    # P1 jumps over P2 (frames 10-29 airborne) and lands on the other side at frame 30.
    T = 50
    p1 = np.r_[np.full(10, 600.0), np.linspace(600, 1400, 20), np.full(20, 1400.0)]
    p2 = np.full(T, 1000.0)
    hip_x = np.c_[p1, p2]
    grounded = np.ones((T, 2), bool)
    grounded[10:30, 0] = False
    facing = nz.estimate_facing(hip_x, grounded, np.full((T, 2), 200.0), margin=0.25)
    assert (facing[:30, 0] == 1).all()        # still facing right while in the air
    assert (facing[30:, 0] == -1).all()       # turns on landing
    # P2 is grounded: turns once P1 is more than 0.25 torso (50 px) past it
    first_turn = np.flatnonzero(facing[:, 1] == 1)[0]
    assert p1[first_turn] - 1000 > 50 and p1[first_turn - 1] - 1000 <= 50


def test_facing_holds_when_opponent_missing_and_unknown_without_data():
    hip_x = np.array([[500, 900], [500, np.nan], [1200, np.nan], [500, 900]], float)
    f = nz.estimate_facing(hip_x, np.ones((4, 2), bool), np.full((4, 2), 200.0))
    assert f[:, 0].tolist() == [1, 1, 1, 1]
    assert nz.estimate_facing(np.full((3, 2), np.nan), np.ones((3, 2), bool), np.ones((3, 2))).tolist() == [[0, 0]] * 3


# ---------------------------------------------------------------- gaps, smoothing, confidence

def test_interpolate_gaps():
    x = np.array([0, 1, np.nan, np.nan, 4, np.nan, 6, np.nan, np.nan], float)
    out = nz.interpolate_gaps(x, max_gap=2)
    assert np.allclose(out[:7], [0, 1, 2, 3, 4, 5, 6])
    assert np.isnan(out[7:]).all()  # no extrapolation past the end
    long = np.r_[0.0, np.full(5, np.nan), 6.0]
    assert np.isnan(nz.interpolate_gaps(long, max_gap=4)[1:6]).all()  # gap longer than max_gap stays NaN
    assert np.allclose(nz.interpolate_gaps(long, max_gap=5), np.arange(7))


def test_smooth_keeps_polynomials_and_short_runs():
    ramp = np.r_[np.arange(10.0) ** 2, np.nan, 1.0, 2.0]
    out = nz.smooth(ramp, 5, 2)
    assert np.allclose(out[:10], ramp[:10])  # order-2 filter keeps a parabola
    assert np.isnan(out[10]) and np.allclose(out[11:], [1, 2])
    noisy = np.r_[np.zeros(10), 1.0, np.zeros(10)]
    assert nz.smooth(noisy, 5, 2)[10] < 1
    with pytest.raises(ValueError):
        nz.smooth(noisy, 4, 2)


def test_low_confidence_joints_are_dropped_then_interpolated():
    T = 20
    kp = make_keypoints(np.full(T, 500.0), np.full(T, 1400.0))
    kp[5:8, 0, 9, 2] = 0.1   # P1 left wrist unreliable for 3 frames
    kp[5:8, 0, 9, :2] = 0    # garbage position
    kp[:, 1, 3, 2] = 0.1     # P2 left ear never reliable
    out = normalize_sequence(kp, H, NormalizeParams(smooth_window=0, max_gap=5))
    assert np.allclose(out["joints"][5:8, 0, 9, :2], CANON[9])  # interpolated back
    assert not out["observed"][5:8, 0, 9].any() and out["observed"][:5, 0, 9].all()
    assert (out["joints"][5:8, 0, 9, 2] == 0).all()               # score 0 = interpolated
    # P2 faces left, so its (dropped) left ear lands on the right-ear slot after mirroring
    assert np.isnan(out["joints"][:, 1, 4, :2]).all()
    assert not np.isnan(out["joints"][:, 1, 3, :2]).any()


def test_missing_fighter_gap():
    T = 30
    kp = make_keypoints(np.linspace(400, 600, T), np.full(T, 1400.0))
    kp[10:14, 0] = np.nan   # short: filled
    kp[20:30, 1] = np.nan   # until the end: not extrapolated
    out = normalize_sequence(kp, H, NormalizeParams(smooth_window=0, max_gap=10))
    assert not np.isnan(out["joints"][:, 0]).any()
    assert np.allclose(out["center"][10:14, 0, 0], np.linspace(400, 600, T)[10:14])
    assert np.isnan(out["joints"][20:, 1]).all()


# ---------------------------------------------------------------- left/right flips

def _walk(T=40):
    """P1 walking forward slowly: hip moves 3 px/frame."""
    return make_keypoints(400 + 3.0 * np.arange(T), np.full(T, 1500.0))


def test_short_leg_swap_is_undone():
    kp = _walk()
    legs_l, legs_r = [11, 13, 15], [12, 14, 16]
    glitch = kp.copy()
    glitch[20:22, 0, legs_l + legs_r] = kp[20:22, 0, legs_r + legs_l]  # RTMPose swaps the legs for 2 frames
    fixed, n = nz.fix_lr_flips(glitch[:, 0], np.full(len(kp), 200.0))
    assert n == 2
    assert np.allclose(fixed, kp[:, 0])


def test_flip_fix_leaves_clean_and_long_swaps_alone():
    kp = _walk()
    fixed, n = nz.fix_lr_flips(kp[:, 0], np.full(len(kp), 200.0))
    assert n == 0 and np.allclose(fixed, kp[:, 0])
    # A swap that lasts 30 frames is RTMPose's consistent labeling, not a glitch.
    arms_l, arms_r = [5, 7, 9], [6, 8, 10]
    long = kp.copy()
    long[10:, 0, arms_l + arms_r] = kp[10:, 0, arms_r + arms_l]
    fixed, n = nz.fix_lr_flips(long[:, 0], np.full(len(kp), 200.0), penalty=0.4)
    assert n == 0


def test_params_from_config():
    assert NormalizeParams.from_config({"max_gap": 3}).max_gap == 3
    with pytest.raises(KeyError):
        NormalizeParams.from_config({"max_gpa": 3})


def test_rejects_bad_shape():
    with pytest.raises(ValueError):
        normalize_sequence(np.zeros((5, 17, 3)), H)
