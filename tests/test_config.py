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


def test_model_configs_resolve():
    """The repo's model configs inherit from the installed mmdet/mmpose packages."""
    from mmengine.config import Config

    from sf6mr.config import resolve_path

    cfg = load_config()
    det = Config.fromfile(resolve_path(cfg["detection"]["config"]))
    assert det.model.type == "RTMDet" and det.model.bbox_head.num_classes == 1
    assert det.model.backbone.init_cfg is None  # nothing downloaded at build time
    pose = Config.fromfile(resolve_path(cfg["pose"]["config"]))
    assert pose.model.head.out_channels == 17 and pose.codec.input_size == (192, 256)
    assert pose.model.backbone.init_cfg is None
