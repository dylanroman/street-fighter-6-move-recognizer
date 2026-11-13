"""Video reading, HUD mask, stage-file I/O and drawing, on synthetic data."""

import cv2
import numpy as np
import pytest

from sf6mr import frames as fr
from sf6mr import runs, visualize


def test_hud_rects_px():
    r = fr.hud_rects_px([[0, 0, 0.5, 0.1], [0.25, 0.9, 1, 1]], 1920, 1080)
    assert r.tolist() == [[0, 0, 960, 108], [480, 972, 1920, 1080]]
    assert fr.hud_rects_px([], 1920, 1080).shape == (0, 4)
    with pytest.raises(ValueError):
        fr.hud_rects_px([[0.5, 0, 0.4, 0.1]], 1920, 1080)  # x0 > x1
    with pytest.raises(ValueError):
        fr.hud_rects_px([[0, 0, 1.2, 0.1]], 1920, 1080)    # not a fraction


def test_apply_hud_mask_copies():
    img = np.full((10, 20, 3), 255, np.uint8)
    out = fr.apply_hud_mask(img, np.array([[0, 0, 5, 2]]))
    assert (out[:2, :5] == 0).all() and (out[2:] == 255).all() and (out[:, 5:] == 255).all()
    assert (img == 255).all()
    assert fr.apply_hud_mask(img, np.zeros((0, 4), int)) is img


def _write_video(path, n, fps=60.0, size=(64, 48)):
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    for i in range(n):
        frame = np.zeros((size[1], size[0], 3), np.uint8)
        frame[:, : (i % size[0]) + 1] = 200
        w.write(frame)
    w.release()


def test_read_video_info_and_iter_frames(tmp_path):
    p = tmp_path / "clip.mp4"
    _write_video(p, 25)
    info = fr.read_video_info(p)
    assert (info.fps, info.width, info.height, info.frame_count) == (60.0, 64, 48, 25)
    assert [t for t, _ in fr.iter_frames(p)] == list(range(25))


def test_read_video_info_rejects_wrong_fps(tmp_path):
    p = tmp_path / "clip30.mp4"
    _write_video(p, 5, fps=30.0)
    with pytest.raises(ValueError, match="60"):
        fr.read_video_info(p)


def test_split_by_frame_and_flat_index():
    det_frame = np.array([0, 0, 2, 3, 3, 3])
    vals = np.arange(6)
    parts = runs.split_by_frame(det_frame, vals, 5)
    assert [p.tolist() for p in parts] == [[0, 1], [], [2], [3, 4, 5], []]
    per_frame = np.array([[1, 0], [-1, -1], [0, -1], [2, 0], [-1, -1]])
    assert runs.to_flat_index(det_frame, per_frame).tolist() == [[1, 0], [-1, -1], [2, -1], [5, 3], [-1, -1]]
    with pytest.raises(ValueError):
        runs.to_flat_index(np.array([1, 0]), per_frame)


def test_detections_and_keypoints_roundtrip(tmp_path):
    from sf6mr.track import TrackerParams

    T = 3
    fighters = np.full((T, 2, 5), np.nan, np.float32)
    fighters[0, 0] = [1, 2, 3, 4, 0.9]
    runs.save_detections(tmp_path / "d.npz", np.array([0, 1]), np.ones((2, 5)), np.ones((2, 144)) / 144,
                         fighters, np.array([[0, -1], [-1, -1], [-1, -1]]), TrackerParams())
    d = runs.load_detections(tmp_path / "d.npz")
    assert d["det_boxes"].shape == (2, 5) and d["det_hist"].dtype == np.float16
    assert np.array_equal(d["fighters"], fighters, equal_nan=True)
    assert d["tracker"]["min_score"] == TrackerParams().min_score

    kp = np.full((T, 2, 17, 3), np.nan, np.float32)
    kp[1, 0] = 1.0
    runs.save_keypoints(tmp_path / "k.npz", kp)
    assert np.array_equal(runs.load_keypoints(tmp_path / "k.npz"), kp, equal_nan=True)
    with pytest.raises(ValueError):
        runs.save_keypoints(tmp_path / "bad.npz", np.zeros((T, 17, 3)))


def test_draw_frame_handles_missing_players():
    img = np.zeros((108, 192, 3), np.uint8)
    fighters = np.full((2, 5), np.nan)
    fighters[0] = [10, 10, 60, 100, 0.9]
    kp = np.full((2, 17, 3), np.nan)
    kp[0, :, :2] = np.linspace(20, 90, 17)[:, None]
    kp[0, :, 2] = 1.0
    out = visualize.draw_frame(img, fighters, kp, frame_index=7)
    assert out.shape == img.shape and out.any() and not img.any()


def test_video_writer(tmp_path):
    p = tmp_path / "out.mp4"
    with visualize.VideoWriter(p, 60, (64, 48)) as w:
        for i in range(10):
            w.write(np.full((48, 64, 3), i * 20, np.uint8))
    cap = cv2.VideoCapture(str(p))
    assert cap.get(cv2.CAP_PROP_FPS) == pytest.approx(60)
    assert sum(1 for _ in iter(lambda: cap.read()[0], False)) == 10


def test_normalized_roundtrip(tmp_path):
    from sf6mr.normalize import NormalizeParams, normalize_sequence

    kp = np.full((4, 2, 17, 3), np.nan)
    res = normalize_sequence(kp, 1080)
    runs.save_normalized(tmp_path / "n.npz", res, NormalizeParams())
    back = runs.load_normalized(tmp_path / "n.npz")
    assert set(res) <= set(back) and back["params"]["max_gap"] == NormalizeParams().max_gap
    assert back["joints"].shape == (4, 2, 17, 3) and np.isnan(back["joints"]).all()
    assert back["facing"].dtype == np.int8 and (back["facing"] == 0).all()
