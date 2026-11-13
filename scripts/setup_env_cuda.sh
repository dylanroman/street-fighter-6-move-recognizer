#!/usr/bin/env bash
# Environment setup for an NVIDIA training machine (Linux or WSL2). Run from the repo root:
#   bash scripts/setup_env_cuda.sh
#
# Uses prebuilt CUDA 12.1 wheels for torch 2.1.2 and mmcv 2.1.0, so no CUDA
# toolkit / compiler is needed - only an NVIDIA driver >= 530 (`nvidia-smi`).
# Python 3.10 must be available (set PYTHON=/path/to/python3.10 if it isn't `python3.10`).
set -euo pipefail

PY=${PYTHON:-python3.10}
VENV=${VENV:-.venv}

if [ ! -x "$VENV/bin/python" ]; then
  "$PY" -m venv "$VENV"
fi
PIP="$VENV/bin/python -m pip"

$PIP install --upgrade "pip<25" "setuptools==69.5.1" wheel

# 1. torch, CUDA 12.1 build
$PIP install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu121

# 2. shared deps
$PIP install -r requirements.txt

# 3. OpenMMLab stack. --only-binary makes pip fail loudly instead of silently
#    trying a (CPU-only, slow, usually broken) source build of mmcv.
$PIP install mmengine==0.10.7
$PIP install --only-binary=mmcv mmcv==2.1.0 \
  -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.1/index.html
$PIP install mmdet==3.3.0
$PIP install --no-build-isolation chumpy==0.70
$PIP install mmpose==1.3.2
# mmaction2 v1.2.0 ships mmaction/models/localizers/drn/ without an __init__.py,
# so every non-editable install (PyPI or git) silently drops it and
# `import mmaction.models` fails. Clone the tag, add the file, install from the clone.
# --no-deps: it wants decord (no macOS arm64 wheel) and opencv-contrib-python
# (clashes with opencv-python); the deps we need are in requirements.txt.
MMA_SRC=$(mktemp -d)/mmaction2
git clone -q --depth 1 --branch v1.2.0 https://github.com/open-mmlab/mmaction2.git "$MMA_SRC"
touch "$MMA_SRC/mmaction/models/localizers/drn/__init__.py"
$PIP install --no-deps "$MMA_SRC"
rm -rf "$MMA_SRC"

# 4. this project (editable)
$PIP install --no-deps -e .

echo
echo "Done. Verify with:  $VENV/bin/python scripts/check_env.py"
