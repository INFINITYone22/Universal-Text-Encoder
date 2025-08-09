from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import torch
import yaml


@dataclass
class Config:
    raw: Dict[str, Any]

    @staticmethod
    def load(path: str) -> "Config":
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return Config(data)

    def get(self, path: str, default: Any = None) -> Any:
        keys = path.split(".")
        node: Any = self.raw
        for k in keys:
            if not isinstance(node, dict) or k not in node:
                return default
            node = node[k]
        return node


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_torch_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def amp_dtype_from_precision(precision: str) -> Optional[torch.dtype]:
    p = precision.lower()
    if p in ("bf16", "bfloat16"):
        return torch.bfloat16
    if p in ("fp16", "float16", "half"):
        return torch.float16
    return None


def save_json(path: str, obj: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_checkpoint(
    output_dir: str,
    model_state: Dict[str, Any],
    optimizer_state: Optional[Dict[str, Any]] = None,
    extra: Optional[Dict[str, Any]] = None,
    filename: str = "last.pt",
) -> str:
    ensure_dir(os.path.join(output_dir, "checkpoints"))
    bundle: Dict[str, Any] = {"model": model_state}
    if optimizer_state is not None:
        bundle["optimizer"] = optimizer_state
    if extra is not None:
        bundle["extra"] = extra
    path = os.path.join(output_dir, "checkpoints", filename)
    torch.save(bundle, path)
    return path


def load_checkpoint(path: str) -> Dict[str, Any]:
    return torch.load(path, map_location="cpu")


