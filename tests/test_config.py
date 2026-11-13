from sf6mr.config import apply_overrides, load_config


def test_default_config_loads():
    cfg = load_config()
    assert cfg["video"]["fps"] == 60
    assert set(cfg["characters"]) == {"chunli", "ken"}


def test_overrides_parse_types_and_create_keys():
    cfg = {"device": "cpu", "video": {"fps": 60}}
    out = apply_overrides(cfg, ["device=mps", "video.fps=60", "hud_mask.enabled=true", "x.y=[1, 2]"])
    assert out["device"] == "mps"
    assert out["hud_mask"]["enabled"] is True
    assert out["x"]["y"] == [1, 2]
    assert cfg["device"] == "cpu"  # original untouched
