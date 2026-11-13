"""Environment verification: versions, devices, compiled MMCV ops, and a forward
pass through each model architecture the pipeline uses.

Models are built from the configs bundled with the installed mmdet / mmpose /
mmaction2 packages with *random weights*, so nothing is downloaded.
"""

from __future__ import annotations

import importlib
import platform
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# Pinned versions. If you change one, re-run scripts/check_env.py on BOTH the
# Mac and the training machine and update README.md / CLAUDE.md.
EXPECTED_VERSIONS = {
    "torch": "2.1.2",
    "torchvision": "0.16.2",
    "mmengine": "0.10.7",
    "mmcv": "2.1.0",
    "mmdet": "3.3.0",
    "mmpose": "1.3.2",
    "mmaction": "1.2.0",
    "numpy": "1.26.4",
}

# Configs inside each package's bundled `.mim/configs` directory.
RTMDET_CFG = ("mmdet", "rtmdet/rtmdet_m_8xb32-300e_coco.py")
RTMPOSE_CFG = ("mmpose", "body_2d_keypoint/rtmpose/coco/rtmpose-m_8xb256-420e_coco-256x192.py")
STGCNPP_CFG = ("mmaction", "skeleton/stgcnpp/stgcnpp_8xb16-joint-u100-80e_ntu60-xsub-keypoint-2d.py")


@dataclass
class CheckResult:
    name: str
    ok: bool
    detail: str = ""
    required: bool = True


@dataclass
class Report:
    results: list[CheckResult] = field(default_factory=list)

    def add(self, name: str, ok: bool, detail: str = "", required: bool = True) -> None:
        self.results.append(CheckResult(name, ok, detail, required))

    @property
    def ok(self) -> bool:
        return all(r.ok for r in self.results if r.required)

    def format(self) -> str:
        lines = []
        for r in self.results:
            tag = "PASS" if r.ok else ("FAIL" if r.required else "WARN")
            lines.append(f"[{tag}] {r.name}" + (f": {r.detail}" if r.detail else ""))
        return "\n".join(lines)


def bundled_config(package: str, rel: str) -> Path:
    mod = importlib.import_module(package)
    path = Path(mod.__file__).parent / ".mim" / "configs" / rel
    if not path.exists():
        raise FileNotFoundError(f"{package} bundled config not found: {path}")
    return path


def available_devices() -> list[str]:
    import torch

    devs = ["cpu"]
    if torch.backends.mps.is_available():
        devs.append("mps")
    if torch.cuda.is_available():
        devs.append("cuda")
    return devs


def check_versions(report: Report) -> None:
    report.add("python", platform.python_version().startswith("3.10."), platform.python_version())
    for mod_name, expected in EXPECTED_VERSIONS.items():
        try:
            mod = importlib.import_module(mod_name)
            got = getattr(mod, "__version__", "?")
            # torch CUDA builds report e.g. "2.1.2+cu121"
            report.add(f"{mod_name} version", got.split("+")[0] == expected, f"{got} (expected {expected})")
        except Exception as e:  # noqa: BLE001
            report.add(f"import {mod_name}", False, repr(e))
    try:
        import cv2

        report.add("opencv", True, cv2.__version__)
    except Exception as e:  # noqa: BLE001
        report.add("import cv2", False, repr(e))


def check_devices(report: Report) -> None:
    import torch

    report.add("device: cpu", True)
    mps = torch.backends.mps.is_available()
    report.add("device: mps", mps, "available" if mps else "not available", required=False)
    if torch.cuda.is_available():
        report.add("device: cuda", True, f"{torch.cuda.get_device_name(0)}, CUDA {torch.version.cuda}")
    else:
        report.add("device: cuda", False, "not available (expected on Mac)", required=False)


def check_mmcv_ops(report: Report, devices: list[str]) -> None:
    """RTMDet post-processing needs MMCV's compiled ops (batched_nms)."""
    import torch

    try:
        from mmcv.ops import batched_nms
    except Exception as e:  # noqa: BLE001
        report.add("mmcv compiled ops import", False, repr(e))
        return
    for dev in devices:
        if dev == "mps":
            continue  # mmcv nms has no MPS kernel; detection post-processing runs on CPU
        try:
            boxes = torch.tensor([[0, 0, 10, 10], [1, 1, 11, 11], [50, 50, 60, 60]], dtype=torch.float32, device=dev)
            scores = torch.tensor([0.9, 0.8, 0.7], device=dev)
            idxs = torch.zeros(3, dtype=torch.long, device=dev)
            dets, keep = batched_nms(boxes, scores, idxs, dict(type="nms", iou_threshold=0.5))
            report.add(f"mmcv batched_nms on {dev}", keep.numel() == 2, f"kept {keep.numel()}/3 boxes")
        except Exception as e:  # noqa: BLE001
            report.add(f"mmcv batched_nms on {dev}", False, repr(e))


def _timed(fn):
    t0 = time.perf_counter()
    out = fn()
    return out, (time.perf_counter() - t0) * 1000


def check_rtmdet(report: Report, device: str, required: bool) -> None:
    try:
        from mmdet.apis import inference_detector, init_detector
        from mmengine.config import Config

        cfg = Config.fromfile(bundled_config(*RTMDET_CFG))
        cfg.model.backbone.init_cfg = None
        cfg.model.bbox_head.num_classes = 1  # person-only, as in the pipeline
        model = init_detector(cfg, checkpoint=None, device=device)
        img = np.random.randint(0, 255, (360, 640, 3), dtype=np.uint8)
        inference_detector(model, img)  # warm-up
        res, ms = _timed(lambda: inference_detector(model, img))
        n = len(res.pred_instances)
        report.add(f"RTMDet-m forward on {device}", True, f"{ms:.0f} ms/frame (640x360, random weights, {n} boxes)", required)
    except Exception as e:  # noqa: BLE001
        report.add(f"RTMDet-m forward on {device}", False, _short_exc(e), required)


def check_rtmpose(report: Report, device: str, required: bool) -> None:
    try:
        from mmengine.config import Config
        from mmpose.apis import inference_topdown, init_model

        cfg = Config.fromfile(bundled_config(*RTMPOSE_CFG))
        cfg.model.backbone.init_cfg = None
        model = init_model(cfg, checkpoint=None, device=device)
        img = np.random.randint(0, 255, (360, 640, 3), dtype=np.uint8)
        bboxes = np.array([[100, 50, 250, 350], [400, 50, 550, 350]], dtype=np.float32)
        inference_topdown(model, img, bboxes)  # warm-up
        res, ms = _timed(lambda: inference_topdown(model, img, bboxes))
        kps = res[0].pred_instances.keypoints
        ok = kps.shape == (1, 17, 2)
        report.add(f"RTMPose-m forward on {device}", ok, f"{ms:.0f} ms for 2 people, keypoints {kps.shape}", required)
    except Exception as e:  # noqa: BLE001
        report.add(f"RTMPose-m forward on {device}", False, _short_exc(e), required)


def check_stgcnpp(report: Report, device: str, required: bool) -> None:
    try:
        import torch
        from mmaction.registry import MODELS
        from mmaction.utils import register_all_modules
        from mmengine.config import Config

        register_all_modules()
        cfg = Config.fromfile(bundled_config(*STGCNPP_CFG))
        cfg.model.cls_head.num_classes = 16  # ~15 moves + no-move
        model = MODELS.build(cfg.model).to(device).eval()
        # (batch, num_clips, persons, frames, joints, channels): x, y, confidence
        x = torch.randn(2, 1, 1, 30, 17, 3, device=device)
        with torch.no_grad():
            feats = model.extract_feat(x)[0]
            logits = model.cls_head(feats)
        ok = tuple(logits.shape) == (2, 16)
        report.add(f"ST-GCN++ forward on {device}", ok, f"logits {tuple(logits.shape)} for 30-frame windows", required)
    except Exception as e:  # noqa: BLE001
        report.add(f"ST-GCN++ forward on {device}", False, _short_exc(e), required)


def _short_exc(e: Exception) -> str:
    tb = traceback.format_exception(type(e), e, e.__traceback__)
    return f"{e!r} (at {tb[-2].strip().splitlines()[0] if len(tb) > 1 else '?'})"


def run_all(models: bool = True) -> Report:
    report = Report()
    check_versions(report)
    try:
        devices = available_devices()
    except Exception as e:  # noqa: BLE001
        report.add("torch devices", False, repr(e))
        return report
    check_devices(report)
    check_mmcv_ops(report, devices)
    if models:
        for dev in devices:
            required = dev != "mps"  # MPS is a nice-to-have; CPU and CUDA must work
            check_rtmdet(report, dev, required)
            check_rtmpose(report, dev, required)
            check_stgcnpp(report, dev, required)
    return report
