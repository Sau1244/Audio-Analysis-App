from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch

from AI_dio.data_preprocessing.audio_utils import (
    crop_or_pad,
    load_audio_mono_resampled,
    resample,
    to_mono,
)
from AI_dio.data_preprocessing.features import (
    FeatureParams,
    build_mel_transforms,
    mel_tokens_from_audio,
)
from AI_dio.training.common import choose_device
from AI_dio.training.models import build_model

_CHECKPOINT_CACHE: dict[
    tuple[str, str], tuple[torch.nn.Module, FeatureParams, torch.device]
] = {}


@dataclass(frozen=True)
class PredictResult:
    wav: Optional[Path]
    score: float
    scores: list[float]
    threshold: float
    label: str
    window_sec: float
    stride_sec: float


def predict_file(
    *,
    checkpoint: str | Path,
    wav: str | Path,
    device: str | torch.device = "auto",
    threshold: float = 0.5,
    window_sec: Optional[float] = None,
    stride_sec: Optional[float] = None,
) -> PredictResult:
    wav_path = Path(wav)
    model, params, device = _load_inference_objects(checkpoint, device)
    audio = load_audio_mono_resampled(wav_path, params.target_sr).squeeze(0)
    return _predict_audio_tensor(
        audio,
        params=params,
        model=model,
        device=device,
        threshold=threshold,
        window_sec=window_sec,
        stride_sec=stride_sec,
        wav_path=wav_path,
    )


def predict_audio(
    *,
    checkpoint: str | Path,
    audio: torch.Tensor,
    sample_rate: int,
    device: str | torch.device = "auto",
    threshold: float = 0.5,
    window_sec: Optional[float] = None,
    stride_sec: Optional[float] = None,
) -> PredictResult:
    model, params, device = _load_inference_objects(checkpoint, device)
    audio_tensor = _prepare_audio(audio, sample_rate, params.target_sr)
    return _predict_audio_tensor(
        audio_tensor,
        params=params,
        model=model,
        device=device,
        threshold=threshold,
        window_sec=window_sec,
        stride_sec=stride_sec,
        wav_path=None,
    )


def _load_inference_objects(
    checkpoint: str | Path, device: str | torch.device
) -> tuple[torch.nn.Module, FeatureParams, torch.device]:
    checkpoint_path = Path(checkpoint)
    resolved_device = (
        device if isinstance(device, torch.device) else choose_device(device)
    )
    cache_key = (str(checkpoint_path.resolve()), str(resolved_device))
    cached = _CHECKPOINT_CACHE.get(cache_key)
    if cached is not None:
        return cached

    ckpt = torch.load(checkpoint_path, map_location="cpu")
    model_name = ckpt.get("model_name")
    feature_params = ckpt.get("feature_params")
    model_state = ckpt.get("model_state")
    if not model_name or not feature_params or model_state is None:
        raise ValueError(
            "Checkpoint missing model metadata. "
            "Re-train to generate compatible checkpoints."
        )
    params = FeatureParams(
        chunk_duration=float(feature_params["chunk_duration"]),
        target_sr=int(feature_params["target_sr"]),
        win_ms=float(feature_params["win_ms"]),
        hop_ms=float(feature_params["hop_ms"]),
        n_mels=int(feature_params["n_mels"]),
    )
    model = build_model(str(model_name))
    model.load_state_dict(model_state)
    model.to(resolved_device)
    model.eval()
    cached = (model, params, resolved_device)
    _CHECKPOINT_CACHE[cache_key] = cached
    return cached


def _prepare_audio(
    audio: torch.Tensor, sample_rate: int, target_sr: int
) -> torch.Tensor:
    audio_tensor = torch.as_tensor(audio).float().cpu()
    if audio_tensor.dim() == 1:
        audio_tensor = audio_tensor.unsqueeze(0)
    elif audio_tensor.dim() != 2:
        raise ValueError(
            f"Expected audio shape [T] or [C, T], got {tuple(audio_tensor.shape)}"
        )
    audio_tensor = to_mono(audio_tensor)
    if sample_rate != target_sr:
        audio_tensor = resample(audio_tensor, sample_rate, target_sr)
    return audio_tensor.squeeze(0)


def _predict_audio_tensor(
    audio: torch.Tensor,
    *,
    params: FeatureParams,
    model: torch.nn.Module,
    device: torch.device,
    threshold: float,
    window_sec: Optional[float],
    stride_sec: Optional[float],
    wav_path: Optional[Path],
) -> PredictResult:
    window_sec = params.chunk_duration if window_sec is None else float(window_sec)
    stride_sec = window_sec if stride_sec is None else float(stride_sec)
    if window_sec <= 0 or stride_sec <= 0:
        raise ValueError("window_sec and stride_sec must be > 0.")
    window_len = int(round(window_sec * params.target_sr))
    stride_len = int(round(stride_sec * params.target_sr))
    windows = _prepare_windows(audio, window_len, stride_len)

    mel, to_db = build_mel_transforms(params, device=device)
    mel.eval()
    to_db.eval()

    scores: list[float] = []
    use_amp = device.type == "cuda"
    with torch.inference_mode():
        for chunk in windows:
            batch = chunk.unsqueeze(0).to(device)
            tokens = mel_tokens_from_audio(batch, params, mel=mel, to_db=to_db)
            if tokens.dim() == 2:
                tokens = tokens.unsqueeze(0)
            with torch.amp.autocast(device_type="cuda", enabled=use_amp):
                logits = model(tokens)
            scores.append(float(torch.softmax(logits, dim=1)[0, 1].item()))

    score = sum(scores) / len(scores) if scores else 0.0
    label = "fake" if score >= float(threshold) else "real"
    return PredictResult(
        wav=wav_path,
        score=score,
        scores=scores,
        threshold=float(threshold),
        label=label,
        window_sec=window_sec,
        stride_sec=stride_sec,
    )


def _window_starts(num_samples: int, window_len: int, stride_len: int) -> list[int]:
    if num_samples <= window_len:
        return [0]
    starts = list(range(0, num_samples - window_len + 1, stride_len))
    last = num_samples - window_len
    if starts[-1] != last:
        starts.append(last)
    return starts


def _prepare_windows(
    audio: torch.Tensor, window_len: int, stride_len: int
) -> list[torch.Tensor]:
    if audio.dim() != 1:
        raise ValueError(f"Expected audio [T], got {tuple(audio.shape)}")
    starts = _window_starts(int(audio.numel()), window_len, stride_len)
    windows: list[torch.Tensor] = []
    for start in starts:
        chunk = audio[start : start + window_len]
        chunk = crop_or_pad(chunk.unsqueeze(0), window_len).squeeze(0)
        windows.append(chunk)
    return windows
