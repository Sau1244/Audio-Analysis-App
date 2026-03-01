import csv
import json
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from AI_dio.data_preprocessing.audio_utils import load_audio_mono_resampled
from AI_dio.data_preprocessing.augmentations import AudioAugmenter
from AI_dio.data_preprocessing.features import (
    FeatureParams,
    build_mel_transforms,
    mel_tokens_from_audio,
)


class AIDetectDataset(Dataset):
    def __init__(
        self,
        manifest_csv: str,
        split: Literal["train", "val", "test"],
        chunk_duration=3.0,
        target_sr=16000,
        win_ms=25.0,
        hop_ms=10.0,
        n_mels=80,
        cache_dir: Optional[str] = None,
        augment: bool = False,
        augment_root: Optional[str] = None,
        augment_cfg: Optional[dict] = None,
        rows: Optional[list[dict]] = None,
    ):
        self._target_sr = target_sr
        self._chunk_length = int(chunk_duration * target_sr)
        self._use_cache = cache_dir is not None
        self._features = None
        self._features_path: Optional[Path] = None
        self._features_shape = None
        self._features_dtype = None
        self._labels = None
        self._augmenter: Optional[AudioAugmenter] = None
        self._feature_augment = None

        if not self._use_cache:
            self._params = FeatureParams(
                chunk_duration=chunk_duration,
                target_sr=target_sr,
                win_ms=win_ms,
                hop_ms=hop_ms,
                n_mels=n_mels,
            )
            self._mel, self._to_db = build_mel_transforms(self._params)

        if augment and not self._use_cache:
            cfg = augment_cfg or {}
            self._augmenter = AudioAugmenter(
                target_sr=target_sr,
                chunk_length=self._chunk_length,
                cfg=cfg,
                augment_root=None if augment_root is None else Path(augment_root),
            )
        elif augment and self._use_cache:
            feature_cfg = (augment_cfg or {}).get("feature_mask") or (
                augment_cfg or {}
            ).get("specaugment")
            if feature_cfg and feature_cfg.get("enabled", True):
                self._feature_augment = _FeatureAugment(
                    time_masks=int(feature_cfg.get("time_masks", 2)),
                    time_width=int(feature_cfg.get("time_width", 12)),
                    freq_masks=int(feature_cfg.get("freq_masks", 2)),
                    freq_width=int(feature_cfg.get("freq_width", 8)),
                    p=float(feature_cfg.get("p", 1.0)),
                    noise_std=float(feature_cfg.get("noise_std", 0.0)),
                )
            else:
                print(
                    "[warn] augment enabled with cache; "
                    "no feature_mask/specaugment config found, disabling augment."
                )

        if rows is not None:
            if cache_dir is not None:
                raise ValueError("cache_dir must be None when passing explicit rows.")
            self.rows = rows
        else:
            with open(manifest_csv) as f:
                reader = csv.DictReader(f.readlines(), delimiter=",")
                self.rows = list(filter(lambda r: r["split"] == split, reader))

        if self._use_cache:
            self._init_cache(
                cache_dir=cache_dir,
                split=split,
                chunk_duration=chunk_duration,
                target_sr=target_sr,
                win_ms=win_ms,
                hop_ms=hop_ms,
                n_mels=n_mels,
            )

    def _init_cache(
        self,
        cache_dir: str,
        split: str,
        chunk_duration: float,
        target_sr: int,
        win_ms: float,
        hop_ms: float,
        n_mels: int,
    ) -> None:
        cache_path = Path(cache_dir)
        metadata_path = cache_path / "metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Missing cache metadata: {metadata_path}")

        with open(metadata_path) as f:
            metadata = json.load(f)

        split_meta = metadata.get("splits", {}).get(split)
        if not split_meta:
            raise ValueError(f"Missing split metadata for '{split}' in {metadata_path}")

        def _assert_same(key: str, expected) -> None:
            got = metadata.get(key)
            if isinstance(expected, float):
                if got is None or abs(float(got) - float(expected)) > 1e-6:
                    raise ValueError(
                        f"Cache mismatch for {key}: expected {expected}, got {got}"
                    )
            else:
                if got != expected:
                    raise ValueError(
                        f"Cache mismatch for {key}: expected {expected}, got {got}"
                    )

        _assert_same("chunk_duration", chunk_duration)
        _assert_same("target_sr", target_sr)
        _assert_same("win_ms", win_ms)
        _assert_same("hop_ms", hop_ms)
        _assert_same("n_mels", n_mels)

        num_samples = int(split_meta["num_samples"])
        num_frames = int(metadata["num_frames"])
        num_mels = int(metadata["n_mels"])

        self._features_path = cache_path / split_meta["features"]
        self._features_shape = (num_samples, num_frames, num_mels)
        self._features_dtype = np.dtype(metadata["dtype"])

        labels_name = split_meta.get("labels")
        if labels_name:
            labels_path = cache_path / labels_name
            if labels_path.exists():
                self._labels = np.load(labels_path, allow_pickle=False)
                if self._labels.shape[0] != num_samples:
                    raise ValueError(
                        "Cache labels length mismatch: "
                        f"{labels_path} has {self._labels.shape[0]}, "
                        f"expected {num_samples}"
                    )

        if len(self.rows) != num_samples:
            raise ValueError(
                "Cache/manifest mismatch: "
                f"manifest has {len(self.rows)} rows for split '{split}', "
                f"cache has {num_samples} samples"
            )

    def _ensure_memmap(self) -> None:
        if self._features is None:
            if self._features_path is None or self._features_shape is None:
                raise RuntimeError("Cache features are not initialized.")
            self._features = np.memmap(
                self._features_path,
                mode="r+",
                dtype=self._features_dtype,
                shape=self._features_shape,
            )

    def _load_from_filepath(self, fp: str) -> torch.Tensor:
        return load_audio_mono_resampled(
            fp,
            target_sr=self._target_sr,
            target_length=self._chunk_length,
        )

    def __getitem__(self, index) -> tuple:
        if self._use_cache:
            self._ensure_memmap()
            tokens = torch.from_numpy(self._features[index])
            if self._labels is not None:
                y = int(self._labels[index])
            else:
                y = int(self.rows[index]["label"])
            if self._feature_augment is not None:
                tokens = self._feature_augment(tokens)
            return tokens, y

        r = self.rows[index]
        path = r["path"]
        y = int(r["label"])
        audio_tensor = self._load_from_filepath(path)
        if self._augmenter is not None:
            audio_tensor = self._augmenter.apply(audio_tensor)
        tokens = mel_tokens_from_audio(
            audio_tensor, self._params, mel=self._mel, to_db=self._to_db
        )
        if tokens.dim() == 3:
            tokens = tokens.squeeze(0)

        return tokens, y

    def __len__(self) -> int:
        return len(self.rows)


class _FeatureAugment:
    def __init__(
        self,
        *,
        time_masks: int,
        time_width: int,
        freq_masks: int,
        freq_width: int,
        p: float,
        noise_std: float,
    ) -> None:
        self._time_masks = max(int(time_masks), 0)
        self._time_width = max(int(time_width), 0)
        self._freq_masks = max(int(freq_masks), 0)
        self._freq_width = max(int(freq_width), 0)
        self._p = float(p)
        self._noise_std = float(noise_std)

    def __call__(self, tokens: torch.Tensor) -> torch.Tensor:
        if self._p <= 0.0 or torch.rand(1).item() > self._p:
            return tokens
        tokens = tokens.clone()
        time_steps, freq_bins = tokens.shape
        if self._time_width > 0 and self._time_masks > 0 and time_steps > 1:
            for _ in range(self._time_masks):
                width = int(torch.randint(0, self._time_width + 1, (1,)).item())
                if width <= 0 or width >= time_steps:
                    continue
                start = int(torch.randint(0, time_steps - width + 1, (1,)).item())
                tokens[start : start + width, :] = 0.0
        if self._freq_width > 0 and self._freq_masks > 0 and freq_bins > 1:
            for _ in range(self._freq_masks):
                width = int(torch.randint(0, self._freq_width + 1, (1,)).item())
                if width <= 0 or width >= freq_bins:
                    continue
                start = int(torch.randint(0, freq_bins - width + 1, (1,)).item())
                tokens[:, start : start + width] = 0.0
        if self._noise_std > 0.0:
            tokens = tokens + torch.randn_like(tokens) * self._noise_std
        return tokens
