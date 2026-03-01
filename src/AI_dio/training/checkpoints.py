from pathlib import Path

import torch


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_checkpoint(path: Path, payload: dict) -> None:
    _ensure_dir(path.parent)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp_path)
    tmp_path.replace(path)


def load_checkpoint_payload(path: Path) -> dict:
    return torch.load(path, map_location="cpu")


def load_checkpoint(
    model: torch.nn.Module, checkpoint_path: Path, device: torch.device
):
    ckpt = load_checkpoint_payload(checkpoint_path)
    state = ckpt.get("model_state", ckpt)
    model.load_state_dict(state)
    model.to(device)
    return ckpt
