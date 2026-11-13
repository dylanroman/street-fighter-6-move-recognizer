# SF6 Move & Input Recognizer

Recovers the **moves and controller inputs** of Street Fighter 6 characters from
gameplay video. The pipeline:

1. finds both fighters in every frame,
2. estimates their 2D skeletons,
3. classifies short windows of skeleton motion into moves,
4. maps each move back to the button input that produced it.

Scope: **Chun-Li** (primary) and **Ken**, Classic controls, ~15 moves each.

```
video (60 fps)
  → frames (optional HUD mask)
  → RTMDet person detection + P1/P2 identity
  → RTMPose-m keypoints (COCO-17)
  → normalization (center, scale, face-right)
  → ST-GCN++ move classifier on 30-frame windows → move events
  → move list (startup frames) → input timeline + annotated video
```

Ground-truth labels come from SF6's on-screen **input display**, which we read
automatically with template matching.

> **No game assets or footage are in this repo.** Bring your own recordings; put them
> under `data/`, which is gitignored.

## Status

| Phase | What | Status |
|---|---|---|
| 1 | Project skeleton, environment, docs | ✅ |
| 2 | Detection + pose + skeleton visualization (pretrained) | ⏳ |
| 3 | Normalization + tests | ⏳ |
| 4 | Input-display auto-labeler + clip extraction | ⏳ |
| 5 | ST-GCN++ training, sliding-window inference, event merging | ⏳ |
| 6 | Move list → input timeline + annotated demo video | ⏳ |
| 7 | RTMPose fine-tuning on SF6 frames | ⏳ |
| 8 | Evaluation + ablations | ⏳ |

## Setup

You need **Python 3.10**. The OpenMMLab libraries (MMCV in particular) are picky about
versions, so use the setup scripts rather than installing packages by hand.

### Mac (Apple Silicon)

```bash
brew install python@3.10 ffmpeg
xcode-select --install            # if you don't have the command line tools yet
bash scripts/setup_env_mac.sh     # ~15-25 min: MMCV is compiled from source
source .venv/bin/activate
python scripts/check_env.py
```

### NVIDIA GPU (Linux or Windows WSL2), for training

You need an NVIDIA driver ≥ 530 (check with `nvidia-smi`). You don't need the CUDA
toolkit; everything comes as prebuilt CUDA 12.1 wheels.

```bash
bash scripts/setup_env_cuda.sh    # set PYTHON=/path/to/python3.10 if needed
source .venv/bin/activate
python scripts/check_env.py
```

Then download the pretrained detector and pose weights (~150 MB) into `models/`:

```bash
python scripts/download_weights.py
```

`check_env.py` should end with `OK`. It checks versions, devices and MMCV's compiled
ops, and runs every model once on dummy input. `WARN` lines for CUDA on a Mac, or for
RTMDet on MPS, are expected.

The Mac setup has been tested from a clean install on macOS 26 (Apple Silicon). The
NVIDIA setup uses the same versions with OpenMMLab's official CUDA wheels, but nobody
has run it yet. If it fails for you, open an issue with the `check_env.py` output.

### Pinned versions

| Package | Version | Note |
|---|---|---|
| Python | 3.10 | |
| torch / torchvision | 2.1.2 / 0.16.2 | `+cu121` on NVIDIA |
| mmengine | 0.10.7 | |
| mmcv | 2.1.0 | mmdet/mmpose require `<2.2.0` |
| mmdet | 3.3.0 | RTMDet |
| mmpose | 1.3.2 | RTMPose |
| mmaction2 | 1.2.0 | ST-GCN++, installed from the git tag |
| numpy | 1.26.4 | torch 2.1 needs numpy 1.x |

### Troubleshooting

- **`mmcv` build fails on Mac:** make sure you're on the Homebrew Python 3.10 and that
  `setuptools==69.5.1` is installed (the script does both). Then rerun the script; it
  reuses `.venv`.
- **`'is_arithmetic' cannot be specialized` while compiling mmcv:** newer Apple clang
  (macOS 26) rejects a torch 2.1 header. The Mac script already passes
  `-Wno-invalid-specialization`. If you build by hand, set
  `CFLAGS="-Wno-invalid-specialization" CXXFLAGS="-Wno-invalid-specialization"`.
- **`No matching distribution for mmcv==2.1.0` on NVIDIA:** your torch isn't `2.1.2+cu121`,
  or your Python isn't 3.10. The script refuses to fall back to a source build.
- **`chumpy` fails to install:** install it with `pip install --no-build-isolation chumpy==0.70`.
- **`No module named 'mmaction.models.localizers.drn'`:** mmaction2 1.2.0 is missing an
  `__init__.py` in that folder, so normal installs drop it. Rerun the mmaction2 step of
  the setup script, which adds the file before installing.

## Repo layout

```
src/sf6mr/     library code, one module per pipeline stage
scripts/       command-line entry points (one per stage)
configs/       pipeline config (default.yaml), move lists, model configs
data/          your footage, templates, labels, stage outputs (gitignored)
models/        model weights (gitignored)
results/       metrics (CSV) and plots (PNG)
tests/         unit tests on synthetic data
```

Each stage is a separate command that saves its output to `data/runs/<video_name>/`,
so you can inspect or rerun any step on its own. Usage for each stage will be added
here as the phase lands.

## Tests

```bash
pytest -m "not slow"     # quick
pytest                   # also builds every model once on CPU
```
