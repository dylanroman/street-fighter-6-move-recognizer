#!/usr/bin/env python
"""Verify the Python / torch / OpenMMLab environment.

    python scripts/check_env.py            # versions, devices, mmcv ops, model forward passes
    python scripts/check_env.py --fast     # skip model forward passes

Exits non-zero if any required check fails. MPS failures are reported as WARN
(MPS is optional; CPU and CUDA are required where present).
"""

import argparse
import sys
import warnings


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fast", action="store_true", help="skip building models")
    args = ap.parse_args()

    warnings.filterwarnings("ignore")
    from sf6mr.envcheck import run_all

    report = run_all(models=not args.fast)
    print(report.format())
    print("\nOK" if report.ok else "\nENVIRONMENT CHECK FAILED")
    return 0 if report.ok else 1


if __name__ == "__main__":
    sys.exit(main())
