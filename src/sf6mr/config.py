"""Pipeline configuration: YAML files plus dotted command-line overrides.

Every stage script accepts ``--config configs/default.yaml`` and any number of
``--set key.subkey=value`` overrides, e.g. ``--set device=mps``.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Iterable

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "default.yaml"


def _parse_value(raw: str) -> Any:
    """Parse an override value with YAML rules (ints, floats, bools, lists, null)."""
    return yaml.safe_load(raw)


def apply_overrides(cfg: dict, overrides: Iterable[str]) -> dict:
    """Return a copy of ``cfg`` with ``a.b.c=value`` overrides applied."""
    cfg = copy.deepcopy(cfg)
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Override must look like key.path=value, got {item!r}")
        key, raw = item.split("=", 1)
        parts = key.strip().split(".")
        node = cfg
        for p in parts[:-1]:
            if not isinstance(node.get(p), dict):
                node[p] = {}
            node = node[p]
        node[parts[-1]] = _parse_value(raw)
    return cfg


def load_config(path: str | Path | None = None, overrides: Iterable[str] = ()) -> dict:
    path = Path(path) if path else DEFAULT_CONFIG
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    return apply_overrides(cfg, overrides)


def resolve_path(p: str | Path) -> Path:
    """Resolve config paths relative to the repo root (absolute paths pass through)."""
    p = Path(p).expanduser()
    return p if p.is_absolute() else REPO_ROOT / p
