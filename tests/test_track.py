"""P1/P2 tracker on synthetic boxes (1080p frame, fighters ~600 px tall on the ground)."""

import numpy as np
import pytest

from sf6mr import track
from sf6mr.track import HIST_BINS, HIST_DIM, TrackerParams, assign_fighters, side_of_p1

H = 1080
FLOOR = 1040  # grounded fighters' feet


def fighter(cx, bottom=FLOOR, h=600, w=250, score=0.8):
    return [cx - w / 2, bottom - h, cx + w / 2, bottom, score]


def spectator(cx, score=0.9):
    # behind the fence: feet well above the floor line, smaller
    return [cx - 70, 400, cx + 70, 800, score]


def frames_of(*per_frame):
    return [np.array(f, float).reshape(-1, 5) for f in per_frame]


def cx_of(fighters):
    return (fighters[..., 0] + fighters[..., 2]) / 2


def onehot_hist(hue_bin):
    h = np.zeros(HIST_BINS)
    h[hue_bin, -1] = 1.0  # fully saturated
    return h.ravel()


def test_picks_fighters_not_spectators_and_p1_is_left():
    dets = frames_of(*[[spectator(900), fighter(1500), fighter(400), spectator(200, 0.95)]] * 20)
    f, idx = assign_fighters(dets, H)
    assert not np.isnan(f).any()
    assert np.allclose(cx_of(f)[:, 0], 400) and np.allclose(cx_of(f)[:, 1], 1500)
    assert (idx[:, 0] == 2).all() and (idx[:, 1] == 1).all()
    assert (side_of_p1(f) == -1).all()


def test_identity_survives_side_switch():
    # A walks right, B walks left; they cross in the middle (both boxes always detected).
    n = 101
    a = np.linspace(400, 1400, n)
    b = np.linspace(1400, 400, n)
    dets = frames_of(*[[fighter(b[t]), fighter(a[t])] for t in range(n)])
    f, _ = assign_fighters(dets, H)
    assert np.allclose(cx_of(f)[:, 0], a, atol=1e-3)  # P1 stays A
    assert np.allclose(cx_of(f)[:, 1], b, atol=1e-3)
    assert side_of_p1(f)[0] == -1 and side_of_p1(f)[-1] == 1


def test_missing_frames_are_nan_and_fighter_is_picked_up_again():
    per = []
    for t in range(40):
        boxes = [fighter(400)]
        if not 20 <= t < 26:
            boxes.append(fighter(1500))
        per.append(boxes)
    f, idx = assign_fighters(frames_of(*per), H)
    assert np.isnan(f[20:26, 1]).all() and (idx[20:26, 1] == -1).all()
    assert not np.isnan(f[26:, 1]).any()
    assert np.allclose(cx_of(f)[26:, 1], 1500)


def test_low_scores_ignored():
    dets = frames_of(*[[fighter(400), fighter(1500, score=0.1)]] * 5)
    f, _ = assign_fighters(dets, H, TrackerParams(min_score=0.3))
    assert np.isnan(f).all()  # tracking only starts once both fighters are seen


def test_track_that_never_lands_is_dropped():
    # P2 vanishes at frame 10; an off-the-ground box of similar size sits where P2 was.
    # The tracker may follow it for a while, but must drop it and blank those frames.
    p = TrackerParams(max_airborne=15)
    per = []
    for t in range(60):
        if t < 10:
            per.append([fighter(400), fighter(1500)])
        else:
            per.append([fighter(400), fighter(1500, bottom=880, h=520)])
    f, _ = assign_fighters(frames_of(*per), H, p)
    assert not np.isnan(f[:10]).any()
    assert np.isnan(f[10:, 1]).all()
    assert not np.isnan(f[:, 0]).any()


def test_appearance_keeps_identity_when_fighters_bounce_off_each_other():
    # Apart for 40 frames (templates are built), then A and B walk toward each other,
    # meet, and walk back. Constant-velocity motion alone would carry them through each
    # other and swap identities; the colour templates must keep A as P1.
    pos_a, pos_b = [500.0] * 40, [1300.0] * 40
    for _ in range(20):
        pos_a.append(pos_a[-1] + 20)
        pos_b.append(pos_b[-1] - 20)
    for _ in range(20):
        pos_a.append(pos_a[-1] - 20)
        pos_b.append(pos_b[-1] + 20)
    ha, hb = onehot_hist(0), onehot_hist(12)
    dets = frames_of(*[[fighter(b), fighter(a)] for a, b in zip(pos_a, pos_b)])
    feats = [np.stack([hb, ha])] * len(dets)
    f, _ = assign_fighters(dets, H, features=feats)
    assert np.allclose(cx_of(f)[:, 0], pos_a)
    assert np.allclose(cx_of(f)[:, 1], pos_b)


def test_identity_follows_colours_when_boxes_trade_places():
    # The two fighters trade places in one frame (as if both re-appeared crossed after
    # an overlap). Position alone would keep P1 at 700; colours must not let P1 take B's
    # box. P1 may be missing for a few frames, but never sits on B.
    ha, hb = onehot_hist(0), onehot_hist(12)
    per, feats = [], []
    for t in range(80):
        if t < 50:
            per.append([fighter(700), fighter(1000)])
            feats.append(np.stack([ha, hb]))
        else:  # A is now at 1000, B at 700
            per.append([fighter(700), fighter(1000)])
            feats.append(np.stack([hb, ha]))
    f, _ = assign_fighters(frames_of(*per), H, features=feats)
    cx = cx_of(f)
    assert np.allclose(cx[:50, 0], 700)
    assert not np.any(cx[50:, 0] == 700)          # never on B
    assert np.allclose(cx[55:, 0], 1000)          # picked A up again
    assert np.allclose(cx[55:, 1], 700)


def test_swap_check_swaps_back_to_last_confirmed_frame():
    # The swap check on its own: identities confirmed up to frame 9, then the boxes
    # match the other fighter's template for swap_frames frames in a row.
    p = TrackerParams(template_frames=1, swap_frames=3)
    app = track._Appearance(p)
    ha, hb = onehot_hist(0), onehot_hist(12)
    assert app.observe(0, ha, hb, clean=True) is False  # builds templates
    for t in range(1, 10):
        assert app.observe(t, ha, hb, clean=True) is False
    assert app.last_confirmed == 9
    assert app.observe(10, hb, ha, clean=True) is False
    assert app.observe(11, hb, ha, clean=True) is False
    assert app.observe(12, hb, ha, clean=True) is True
    # unreliable colours (grey) give no evidence either way
    grey = np.zeros(HIST_DIM)
    grey[0] = 1
    assert app.observe(13, grey, grey, clean=True) is False


def test_color_hist_and_similarity():
    img = np.zeros((200, 400, 3), np.uint8)
    img[:, :200] = (0, 0, 220)    # red (BGR)
    img[:, 200:] = (220, 60, 0)   # blue
    red = track.color_hist(img, [0, 0, 200, 200])
    blue = track.color_hist(img, [200, 0, 400, 200])
    assert red.shape == (HIST_DIM,) and red.sum() == pytest.approx(1)
    assert track.similarity(red, red) == pytest.approx(1, abs=1e-5)
    assert track.similarity(red, blue) < 0.1
    grey = track.color_hist(np.full((100, 100, 3), 128, np.uint8), [0, 0, 100, 100])
    assert track._colorful(grey) == pytest.approx(0)
    assert track._colorful(red) > 0.9
    assert track.color_hist(img, [10, 10, 10, 50]).sum() == 0  # empty crop


def test_iou():
    a = np.array([[0, 0, 10, 10]])
    b = np.array([[0, 0, 10, 10], [5, 0, 15, 10], [20, 20, 30, 30]])
    assert np.allclose(track.iou(a, b), [[1, 1 / 3, 0]])


def test_params_from_config_rejects_unknown_keys():
    assert TrackerParams.from_config({"min_score": 0.5}).min_score == 0.5
    assert TrackerParams.from_config(None) == TrackerParams()
    with pytest.raises(KeyError):
        TrackerParams.from_config({"min_scroe": 0.5})
