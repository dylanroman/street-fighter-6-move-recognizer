# CLAUDE.md

Applied ML class project (CPP CS 4210): recover the **moves and controller inputs** of
Street Fighter 6 characters from 60 fps gameplay video using 2D pose estimation
(RTMDet + RTMPose) and a skeleton action classifier (ST-GCN++).

Scope: **Chun-Li (primary) and Ken**, Classic controls, ~15 moves each plus a `no_move`
class (idle / walk / block). Chun-Li comes first: each stage is built, tuned and tested
on her first, then extended to Ken.

## Hard rules

- **60 fps, never downsample or skip frames.** Moves are defined in frames; every
  stage keeps a 1:1 mapping to source frame indices (0-based).
- **Don't download datasets or game assets.** The user supplies all footage,
  input-display templates and labels. Pretrained *model weights* from OpenMMLab are fine.
- **Don't change the pinned OpenMMLab stack** without re-running `scripts/check_env.py`
  on both machines (see "Environment").
- Work is done in phases; **stop after each phase** so the user can test on real footage.
- **All model code lives in Jupyter notebooks** (`notebooks/`, one per stage): building
  or loading RTMDet / RTMPose / ST-GCN++, inference loops, training and fine-tuning.
  `src/sf6mr/` holds only model-free helpers (see "Layout"). Each stage notebook runs
  top to bottom on its own (Restart & Run All), reads earlier stages' outputs from
  disk and writes its own output to disk.

## Environment

Python 3.10 venv at `.venv/`. Two setups that install the same pinned stack:

| | Mac (Apple Silicon): dev + inference | NVIDIA home machine: training |
|---|---|---|
| Script | `bash scripts/setup_env_mac.sh` | `bash scripts/setup_env_cuda.sh` (Linux / WSL2) |
| torch | 2.1.2 (PyPI, has MPS) | 2.1.2+cu121 |
| mmcv | 2.1.0 **compiled from source** (`MMCV_WITH_OPS=1`) | 2.1.0 prebuilt cu121/torch2.1 wheel |

Pins: torch 2.1.2, torchvision 0.16.2, mmengine 0.10.7, mmcv 2.1.0, mmdet 3.3.0,
mmpose 1.3.2, mmaction2 1.2.0, numpy 1.26.4, opencv-python 4.10.0.84.

Why these pins and the install quirks:
- mmdet 3.3 and mmpose 1.3 both assert `mmcv<2.2.0`, so 2.1.0 is the newest usable mmcv;
  OpenMMLab's newest prebuilt mmcv 2.1.0 CUDA wheels are for torch 2.1, so torch is 2.1.2.
- torch 2.1 needs numpy 1.x.
- The mmcv source build needs `setuptools<70` (its setup.py imports `pkg_resources`).
- Apple clang ≥ 21 (macOS 26) refuses to compile torch 2.1's `c10/util/strong_type.h`
  (`-Winvalid-specialization`). The Mac script passes `-Wno-invalid-specialization` in
  CFLAGS/CXXFLAGS for the mmcv build.
- `chumpy` (an mmpose dependency) must be installed with `--no-build-isolation`.
- mmaction2 v1.2.0's `mmaction/models/localizers/drn/` has no `__init__.py`, so every
  non-editable install (PyPI or git) drops it and `import mmaction.models` fails. The
  setup scripts clone the tag, `touch` that file, and install from the clone with
  `--no-deps`. Its `decord` dependency has no macOS arm64 wheel and it wants
  `opencv-contrib-python`, which conflicts with `opencv-python`. Its real runtime deps,
  including `importlib-metadata` and `einops`, are in requirements.txt.
- PYSKL was not used: it needs mmcv 1.x, which can't coexist with MMPose 1.x.
- Verified on the Mac (M-series, macOS 26.6, Apple clang 21) on 2026-11-12. The CUDA
  path uses the same pins but hasn't been run yet; run `check_env.py` there first.

Verify: `.venv/bin/python scripts/check_env.py`. It checks versions, devices and MMCV
compiled ops, and runs RTMDet-m, RTMPose-m and ST-GCN++ forward passes on every device
(random weights, nothing downloaded). Use `--fast` to skip the models.

Devices: `device` in the config is `cpu | mps | cuda`. Training notebooks must run on
CUDA, and on CPU/MPS for small smoke tests. Mac measurements from `check_env.py`
(random weights, so the numbers are rough):

| Model | CPU | MPS |
|---|---|---|
| RTMDet-m, 640x360 | ~470 ms/frame | ✗ mmcv `nms` has no MPS kernel |
| RTMPose-m, 2 people | ~145 ms | ~25 ms |
| ST-GCN++, 30-frame window | ✓ | ✓ |

So on the Mac, pose should run on MPS. Detection needs either CPU, or its backbone on
MPS with NMS on CPU (decided in Phase 2).

## Layout

```
notebooks/        stage notebooks NN_<stage>.ipynb (all model code); unnumbered = scratch
src/sf6mr/        model-free helpers the notebooks import (installed editable: `pip install -e .`)
  config.py       YAML config + `a.b=value` overrides; resolve_path() is repo-relative
  envcheck.py     environment verification used by scripts/check_env.py and tests/test_env.py
scripts/          setup_env_*, check_env, download_weights only (no stage scripts)
configs/          default.yaml, movelists/<character>.csv, mmpose/, mmaction/ model configs
data/             (gitignored) footage, templates, labels, and per-video stage outputs
models/           (gitignored) weights/checkpoints
results/          metrics CSVs + plot PNGs (committed); videos ignored
tests/            pytest; synthetic data only
```

What goes where: anything that builds or loads a model, or runs one, goes in a notebook.
Pure functions the tests need (stage-file I/O, normalization, event merging, move list →
input mapping, input-display template matching) go in `src/sf6mr/` and are imported by
the notebooks. `check_env.py` / `envcheck.py` are the one exception: they build models
with random weights only to verify the environment.

## Running

```bash
source .venv/bin/activate
python scripts/download_weights.py    # pretrained RTMDet-m person + RTMPose-m COCO -> models/ (sha256-pinned)
python scripts/check_env.py           # environment
pytest -m "not slow"                  # fast tests
pytest                                # includes model forward passes (~1 min on CPU)
```

Stage notebooks use the `.venv` kernel (`ipykernel` is in requirements.txt). Each one
starts with a single parameters cell, then loads the config with the same override
syntax for every stage:

```python
CONFIG = "configs/default.yaml"    # repo-relative
VIDEO = "data/raw/<clip>.mp4"
OVERRIDES = ["device=mps"]         # same "a.b=value" syntax as load_config()

from sf6mr.config import load_config, resolve_path
cfg = load_config(resolve_path(CONFIG), OVERRIDES)
```

Outputs go to `data/runs/<video_stem>/`. Notebooks are committed **with outputs
cleared**; figures worth keeping are saved to `results/*.png`. Long training runs on the
NVIDIA machine (Phases 5 and 7) can be run headless with `jupyter nbconvert --execute`;
`nbconvert` gets added to requirements.txt in Phase 5.

## Pipeline and stage outputs

Status: ✅ done · ⏳ planned (phase number). File formats for planned stages are the
intended contract and get finalized when that phase is built.

Notebook names are planned and get fixed when each phase is built; helper modules in
`src/sf6mr/` are added only where the tests or several notebooks need them.

| # | Stage | Notebook (helpers) | Output in `data/runs/<video>/` | Status |
|---|---|---|---|---|
| 1 | Frames: read video (OpenCV), optional HUD mask (fractional rects in config) | `02_detect_pose` (`frames`) | `frames.json` (fps, size, count, mask) | ⏳ 2 |
| 2 | Detection: RTMDet person boxes; P1/P2 by matching to the previous frame (survives side switches) | `02_detect_pose` | `detections.npz` | ⏳ 2 |
| 3 | Pose: RTMPose-m 256x192, COCO-17 | `02_detect_pose` | `keypoints.npz` `(T, 2, 17, 3)` x, y, conf in pixels | ⏳ 2 |
| 4 | Normalize: hip-center, torso scale, mirror + L/R swap when facing left, interpolate low-conf joints, smoothing | `03_normalize` (`normalize`) | `normalized.npz` | ⏳ 3 |
| 5 | Classify: ST-GCN++ sliding windows (30 frames), merge into events, drop events shorter than startup | `05_classify` (`events`) | `predictions.npz`, `events.csv` | ⏳ 5 |
| 6 | Move list → inputs: input frame = event start − startup; forward/back → screen direction via facing | `06_timeline` (`movelist`, `timeline`) | `timeline.json`, `timeline.csv` | ⏳ 6 |
| 7 | Annotated video: skeletons + scrolling input list | `02_detect_pose` (skeletons), `06_timeline` (inputs) | `annotated.mp4` | ⏳ 2, 6 |

Side notebooks:
- `04_autolabel`: input-display auto-labeler (template matching against user-provided
  icons, with the screen region set in config) → frame-level labels and clip
  extraction (Phase 4).
- `05_train_stgcn`: ST-GCN++ training per character, on CUDA (Phase 5).
- `07_finetune_rtmpose`: low-confidence frame picker for CVAT labeling + RTMPose
  fine-tuning (Phase 7).
- `08_evaluate`: pose PCK, move top-1 + confusion matrix, input edit distance,
  ablations → `results/*.csv`, `results/*.png` (Phase 8).

## Conventions

- Array axis order: `(frame, player, joint, channel)`. Player 0 = P1, 1 = P2. Channels
  are `x, y, confidence`.
- Keypoints use COCO-17 order: 0 nose, 1/2 eyes, 3/4 ears, 5/6 shoulders, 7/8 elbows,
  9/10 wrists, 11/12 hips, 13/14 knees, 15/16 ankles (odd = left, even = right).
- Missing detections or joints are NaN, never 0.
- Inputs use **numpad notation relative to the character** (6 = forward, 4 = back,
  236 = quarter-circle forward). They are converted to screen directions only in the
  final timeline, using facing.
- Move-list CSV columns: `move_id, display_name, input_notation, startup_frames,
  total_frames`. `total_frames` was added to the original spec; the auto-labeler needs it
  to cut clips. Frame data must be checked by the user in-game; don't fill it in from memory.
- Tests use synthetic data only. Required unit tests: normalization (mirroring, L/R
  swap, scaling), event merging, move→input mapping.
