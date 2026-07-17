from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict


def load_config(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)
    if "inherits" in cfg:
        base_path = path.parent / cfg["inherits"]
        base = load_config(base_path)
        merged = deepcopy(base)
        _deep_update(merged, {k: v for k, v in cfg.items() if k != "inherits"})
        return merged
    return cfg


def _deep_update(base: Dict[str, Any], override: Dict[str, Any]) -> None:
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
