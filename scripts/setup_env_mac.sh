#!/usr/bin/env bash
# Environment setup for Apple Silicon (CPU + MPS). Run from the repo root:
#   bash scripts/setup_env_mac.sh
#
# Prerequisites: Homebrew python@3.10 (`brew install python@3.10`) and the
# Xcode command line tools (`xcode-select --install`). MMCV has no arm64 wheels,
# so it is compiled from source here (~10-20 minutes).
set -euo pipefail

PY=${PYTHON:-/opt/homebrew/bin/python3.10}
VENV=${VENV:-.venv}

if [ ! -x "$VENV/bin/python" ]; then
  "$PY" -m venv "$VENV"
fi
PIP="$VENV/bin/python -m pip"

# setuptools<70: mmcv 2.1.0's setup.py still imports pkg_resources.
$PIP install --upgrade "pip<25" "setuptools==69.5.1" wheel ninja

# 1. torch (default PyPI wheel on macOS includes MPS support)
$PIP install torch==2.1.2 torchvision==0.16.2

# 2. shared deps
$PIP install -r requirements.txt

# 3. OpenMMLab stack
$PIP install mmengine==0.10.7
# Apple clang >= 21 (Xcode for macOS 26) rejects torch 2.1's specialization of
# std::is_arithmetic in c10/util/strong_type.h as an error. It's benign; disable
# that one diagnostic for the build. Older clangs ignore the unknown flag.
MMCV_WITH_OPS=1 MAX_JOBS=${MAX_JOBS:-8} \
  CFLAGS="${CFLAGS:-} -Wno-invalid-specialization" \
  CXXFLAGS="${CXXFLAGS:-} -Wno-invalid-specialization" \
  $PIP install --no-build-isolation --no-binary mmcv mmcv==2.1.0
$PIP install mmdet==3.3.0
# chumpy (an mmpose dependency) imports pip internals in setup.py and fails
# under build isolation.
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
