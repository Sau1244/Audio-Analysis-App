from pathlib import Path
from typing import Optional

import torch
import yaml


def collate_fn(batch):
    xs, ys = zip(*batch)  # (T, 80)
    x = torch.stack(xs, dim=0).float()  # (B, T, 80)
    y = torch.tensor(ys, dtype=torch.long)
    return x, y


def load_yaml_config(config_path: Path) -> dict:
    with open(config_path, "r") as f:
        config = yaml.safe_load(f) or {}
    if not isinstance(config, dict):
        raise ValueError(f"Training config must be a mapping: {config_path}")
    return config


def get_section(config: dict, key: str) -> dict:
    section = config.get(key) or {}
    if not isinstance(section, dict):
        raise ValueError(f"Config section '{key}' must be a mapping.")
    return section


def resolve_path(root: Path, value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (root / path).resolve()


def resolve_optional_path(root: Path, value: Optional[str | Path]) -> Optional[Path]:
    if value is None:
        return None
    return resolve_path(root, value)


def _metric_value(metrics: dict, key: str) -> Optional[float]:
    if key in metrics:
        return metrics[key]
    alt = key.replace("_", "/")
    return metrics.get(alt)


def resolve_metric(metrics: dict, requested: str) -> tuple[str, float]:
    requested_key = requested.replace("/", "_")
    value = _metric_value(metrics, requested_key)
    if value is not None:
        return requested_key, value
    if requested_key.startswith("val_"):
        fallback = f"train_{requested_key[len('val_') :]}"
        value = _metric_value(metrics, fallback)
        if value is not None:
            return fallback, value
    for key in ("train_loss", "train_acc"):
        value = _metric_value(metrics, key)
        if value is not None:
            return key, value
    raise ValueError("No valid metrics available to select a best checkpoint.")


def is_better_metric(metric_key: str, current: float, best: Optional[float]) -> bool:
    if best is None:
        return True
    if metric_key.endswith("acc"):
        return current > best
    return current < best


def choose_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)
