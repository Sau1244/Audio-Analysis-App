import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from AI_dio.data_preprocessing.audio_utils import load_audio_mono_resampled
from AI_dio.data_preprocessing.augmentations import AudioAugmenter
from AI_dio.data_preprocessing.features import (
    FeatureParams,
    build_mel_transforms,
    mel_tokens_from_audio,
    num_frames,
    params_from_config,
    stft_params,
)
from AI_dio.data_preprocessing.sources import (
    ManifestRow,
    read_manifest_rows,
    split_manifest_rows,
    validate_manifest,
)
from AI_dio.training.common import (
    choose_device,
    get_section,
    load_yaml_config,
    resolve_path,
)

ROOT = Path(__file__).parents[3].resolve()
DEFAULT_CONFIG = ROOT / "training_config.yml"


def _load_manifest(manifest_csv: Path) -> dict[str, list[ManifestRow]]:
    rows = read_manifest_rows(manifest_csv)
    validate_manifest(rows)
    return split_manifest_rows(rows)


class _AudioDataset(Dataset):
    def __init__(
        self,
        rows: list[ManifestRow],
        target_sr: int,
        chunk_length: int,
        *,
        augment: bool = False,
        augment_root: Path | None = None,
        augment_cfg: dict | None = None,
    ) -> None:
        self._rows = rows
        self._target_sr = target_sr
        self._chunk_length = chunk_length
        self._augmenter: AudioAugmenter | None = None

        if augment:
            cfg = augment_cfg or {}
            self._augmenter = AudioAugmenter(
                target_sr=target_sr,
                chunk_length=chunk_length,
                cfg=cfg,
                augment_root=augment_root,
            )

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, idx: int) -> tuple[int, torch.Tensor, int, bool]:
        row = self._rows[idx]
        ok = True
        try:
            audio_tensor = load_audio_mono_resampled(
                row.path,
                target_sr=self._target_sr,
                target_length=self._chunk_length,
            )
        except Exception:
            ok = False
            audio_tensor = torch.zeros((1, self._chunk_length), dtype=torch.float32)
        if ok and self._augmenter is not None:
            audio_tensor = self._augmenter.apply(audio_tensor)
        return idx, audio_tensor, int(row.label), ok


def _write_split_cache(
    *,
    split: str,
    rows: list[dict],
    output_dir: Path,
    mel,
    to_db,
    chunk_length: int,
    num_frames: int,
    n_mels: int,
    dtype: np.dtype,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    prefetch_factor: int | None,
    params: FeatureParams,
    augment: bool,
    augment_root: Path | None,
    augment_cfg: dict | None,
) -> dict:
    num_samples = len(rows)
    features_path = output_dir / f"features_{split}.mmap"
    labels_path = output_dir / f"labels_{split}.npy"

    features = np.memmap(
        features_path,
        mode="w+",
        dtype=dtype,
        shape=(num_samples, num_frames, n_mels),
    )
    labels = np.empty((num_samples,), dtype=np.int64)

    dataset = _AudioDataset(
        rows=rows,
        target_sr=params.target_sr,
        chunk_length=chunk_length,
        augment=augment,
        augment_root=augment_root,
        augment_cfg=augment_cfg,
    )
    loader_kwargs: dict = {
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "drop_last": False,
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        if prefetch_factor is not None:
            loader_kwargs["prefetch_factor"] = prefetch_factor
    loader = DataLoader(dataset, **loader_kwargs)
    non_blocking = pin_memory and device.type == "cuda"

    with torch.inference_mode():
        failed = 0
        for indices, audio_batch, labels_batch, ok_batch in tqdm(
            loader, desc=f"Cache {split}"
        ):
            ok_tensor = torch.as_tensor(ok_batch)
            failed += int((~ok_tensor).sum().item())
            audio_batch = audio_batch.to(device, non_blocking=non_blocking)
            if audio_batch.dim() == 3 and audio_batch.size(1) == 1:
                audio_batch = audio_batch.squeeze(1)
            tokens = mel_tokens_from_audio(audio_batch, params, mel=mel, to_db=to_db)
            if tokens.dim() != 3:
                raise RuntimeError(
                    f"Unexpected mel shape for split '{split}': {tuple(tokens.shape)}"
                )

            if tokens.shape[1] != num_frames:
                if tokens.shape[1] > num_frames:
                    tokens = tokens[:, :num_frames, :]
                else:
                    pad = num_frames - tokens.shape[1]
                    tokens = torch.nn.functional.pad(tokens, (0, 0, 0, pad))

            tokens_np = tokens.cpu().numpy().astype(dtype, copy=False)
            indices_np = indices.cpu().numpy()
            labels_np = labels_batch.cpu().numpy()
            if indices_np.size > 0:
                start = int(indices_np[0])
                if np.array_equal(
                    indices_np, np.arange(start, start + indices_np.size)
                ):
                    features[start : start + indices_np.size] = tokens_np
                    labels[start : start + indices_np.size] = labels_np
                else:
                    for row_idx, sample_idx in enumerate(indices_np):
                        features[int(sample_idx)] = tokens_np[row_idx]
                        labels[int(sample_idx)] = int(labels_np[row_idx])

    features.flush()
    np.save(labels_path, labels)
    if failed:
        print(
            f"[warn] {failed} audio files failed to decode in split '{split}'. "
            "Silence was cached for those entries."
        )

    return {
        "num_samples": num_samples,
        "features": features_path.name,
        "labels": labels_path.name,
    }


def build_cache(
    splits: dict[str, list[ManifestRow]],
    output_dir: Path,
    params: FeatureParams,
    dtype: np.dtype,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    prefetch_factor: int | None,
    augment: bool,
    augment_root: Path | None,
    augment_cfg: dict | None,
    augment_splits: set[str],
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = output_dir / "metadata.json"
    if metadata_path.exists():
        raise FileExistsError(f"Cache already exists: {metadata_path}")

    win_length, hop_length, n_fft = stft_params(params)
    chunk_length = int(params.chunk_duration * params.target_sr)
    frames_per_clip = num_frames(params, center=True)

    mel, to_db = build_mel_transforms(params, device=device)
    mel.eval()
    to_db.eval()

    metadata = {
        "chunk_duration": params.chunk_duration,
        "target_sr": params.target_sr,
        "win_ms": params.win_ms,
        "hop_ms": params.hop_ms,
        "n_mels": params.n_mels,
        "n_fft": n_fft,
        "chunk_length": chunk_length,
        "num_frames": frames_per_clip,
        "dtype": str(dtype),
        "splits": {},
    }

    for split, rows in splits.items():
        if not rows:
            continue
        split_meta = _write_split_cache(
            split=split,
            rows=rows,
            output_dir=output_dir,
            mel=mel,
            to_db=to_db,
            chunk_length=chunk_length,
            num_frames=frames_per_clip,
            n_mels=params.n_mels,
            dtype=dtype,
            device=device,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            prefetch_factor=prefetch_factor,
            params=params,
            augment=augment and split in augment_splits,
            augment_root=augment_root,
            augment_cfg=augment_cfg,
        )
        metadata["splits"][split] = split_meta

    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    return metadata_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Cache log-mel features into per-split memmap files."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()

    config = load_yaml_config(args.config)
    data_cfg = get_section(config, "data")
    cache_cfg = get_section(config, "cache")
    loader_cfg = get_section(config, "loader")
    features_cfg = get_section(config, "features")

    manifest = resolve_path(
        ROOT, cache_cfg.get("manifest") or data_cfg.get("manifest") or "manifest.csv"
    )
    splits = _load_manifest(manifest)

    output_dir = resolve_path(
        ROOT, cache_cfg.get("output_dir") or data_cfg.get("cache_dir") or "data/cache"
    )
    device = choose_device(cache_cfg.get("device", "cpu"))
    batch_size = int(cache_cfg.get("batch_size", loader_cfg.get("batch_size", 1)))
    num_workers = int(cache_cfg.get("num_workers", loader_cfg.get("num_workers", 0)))
    pin_memory = bool(cache_cfg.get("pin_memory", loader_cfg.get("pin_memory", False)))
    prefetch_factor = int(
        cache_cfg.get("prefetch_factor", loader_cfg.get("prefetch_factor", 2))
    )
    if batch_size < 1:
        raise ValueError("cache.batch_size must be >= 1")
    if num_workers < 0:
        raise ValueError("cache.num_workers must be >= 0")
    prefetch_factor = prefetch_factor if num_workers > 0 else None

    dtype = np.dtype(cache_cfg.get("dtype", "float32"))
    if dtype.kind != "f":
        raise ValueError(f"Expected a float dtype, got {dtype}")

    params = params_from_config(features_cfg)
    augment_cfg = data_cfg.get("augment")
    augment_enabled = bool(cache_cfg.get("augment", False))
    augment_root = resolve_path(ROOT, (augment_cfg or {}).get("root") or "data/augment")
    augment_splits = set(cache_cfg.get("augment_splits") or ["train"])

    metadata_path = build_cache(
        splits=splits,
        output_dir=output_dir,
        params=params,
        dtype=dtype,
        device=device,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory and device.type == "cuda",
        prefetch_factor=prefetch_factor,
        augment=augment_enabled,
        augment_root=augment_root,
        augment_cfg=augment_cfg,
        augment_splits=augment_splits,
    )
    print(f"Wrote cache metadata to {metadata_path}")
