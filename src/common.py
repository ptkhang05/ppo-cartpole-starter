from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch


def set_global_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def ensure_directories(root: Path) -> dict[str, Path]:
    paths = {
        "models": root / "models",
        "logs": root / "logs",
        "outputs": root / "outputs",
        "videos": root / "videos",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

