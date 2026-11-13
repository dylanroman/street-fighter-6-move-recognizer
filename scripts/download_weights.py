#!/usr/bin/env python
"""Download pretrained OpenMMLab weights into models/ and verify them.

    python scripts/download_weights.py          # all
    python scripts/download_weights.py --list

Each file is checked against a pinned SHA-256. Don't rely on the 8-hex hash in
OpenMMLab filenames: it doesn't match the published RTMPose-m COCO file, even
though that file is byte-identical to what the server stores (MD5 == ETag).
Valid files that are already present are skipped. Standard library only, so it
runs before the venv exists.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path

MODELS_DIR = Path(__file__).resolve().parents[1] / "models"

# name -> (url, sha256)
WEIGHTS = {
    # Person-only RTMDet-m (COCO + Objects365 person), 640x640 input.
    # Config: mmpose projects/rtmpose/rtmdet/person/rtmdet_m_640-8xb32_coco-person.py
    "rtmdet_m_person": (
        "https://download.openmmlab.com/mmpose/v1/projects/rtmpose/"
        "rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth",
        "235e820939cb2ff33c505441e71f7e9532958c281636a963c6829d100867aed9",
    ),
    # RTMPose-m, 256x192, COCO-17 keypoints: the COCO checkpoint (backbone pretrained on AIC+COCO).
    # Config: mmpose configs/body_2d_keypoint/rtmpose/coco/rtmpose-m_8xb256-420e_coco-256x192.py
    "rtmpose_m_coco": (
        "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/"
        "rtmpose-m_simcc-coco_pt-aic-coco_420e-256x192-d8dd5ca4_20230127.pth",
        "929f00f2d84764aa0297decf13a84556642e43cbf5fca8cfb16e7326d92dfd97",
    ),
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url: str, dest: Path) -> None:
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url) as r, open(tmp, "wb") as f:
        total = int(r.headers.get("Content-Length", 0))
        done = 0
        while chunk := r.read(1 << 20):
            f.write(chunk)
            done += len(chunk)
            if total:
                print(f"\r  {done / 1e6:6.1f} / {total / 1e6:.1f} MB", end="", flush=True)
    print()
    tmp.rename(dest)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("names", nargs="*", help=f"subset of: {', '.join(WEIGHTS)}")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    if args.list:
        for name, (url, _) in WEIGHTS.items():
            print(f"{name:18s} {url}")
        return 0

    MODELS_DIR.mkdir(exist_ok=True)
    ok = True
    for name in args.names or WEIGHTS:
        url, digest = WEIGHTS[name]
        dest = MODELS_DIR / url.rsplit("/", 1)[1]
        if dest.exists() and sha256(dest) == digest:
            print(f"[skip] {name}: {dest.name} already present and valid")
            continue
        print(f"[get ] {name}: {url}")
        download(url, dest)
        if sha256(dest) == digest:
            print(f"[ ok ] models/{dest.name} (sha256 verified)")
        else:
            print(f"[FAIL] {dest.name}: sha256 mismatch; deleted")
            dest.unlink()
            ok = False
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
