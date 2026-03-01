from dataclasses import dataclass

import torch
import torchaudio.transforms as T


@dataclass
class FeatureParams:
    chunk_duration: float = 3.0
    target_sr: int = 16000
    win_ms: float = 25.0
    hop_ms: float = 10.0
    n_mels: int = 80


def _next_power_of_two(n: int) -> int:
    p = 1
    while p < n:
        p *= 2
    return p


def stft_params(params: FeatureParams) -> tuple[int, int, int]:
    win_length = int(round(params.win_ms / 1000.0 * params.target_sr))
    hop_length = int(round(params.hop_ms / 1000.0 * params.target_sr))
    n_fft = _next_power_of_two(win_length)
    return win_length, hop_length, n_fft


def num_frames(params: FeatureParams, *, center: bool = True) -> int:
    win_length, hop_length, n_fft = stft_params(params)
    chunk_length = int(params.chunk_duration * params.target_sr)
    pad = n_fft // 2 if center else 0
    return 1 + (chunk_length + 2 * pad - n_fft) // hop_length


def build_mel_transforms(
    params: FeatureParams, device: torch.device | None = None
) -> tuple[T.MelSpectrogram, T.AmplitudeToDB]:
    win_length, hop_length, n_fft = stft_params(params)
    mel = T.MelSpectrogram(
        sample_rate=params.target_sr,
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length,
        n_mels=params.n_mels,
        power=2.0,
        center=True,
    )
    to_db = T.AmplitudeToDB(stype="power")
    if device is not None:
        mel = mel.to(device)
        to_db = to_db.to(device)
    return mel, to_db


def mel_tokens_from_audio(
    audio: torch.Tensor,
    params: FeatureParams,
    *,
    mel: T.MelSpectrogram | None = None,
    to_db: T.AmplitudeToDB | None = None,
) -> torch.Tensor:
    if mel is None or to_db is None:
        mel, to_db = build_mel_transforms(params, device=audio.device)
    mel_spec = mel(audio)
    mel_db = to_db(mel_spec)
    if mel_db.dim() == 4:
        mel_db = mel_db.mean(dim=1)
    if mel_db.dim() == 3:
        return mel_db.transpose(1, 2).contiguous()
    if mel_db.dim() == 2:
        return mel_db.transpose(0, 1).contiguous()
    raise RuntimeError(f"Unexpected mel shape: {tuple(mel_db.shape)}")


def params_from_config(cfg: dict | None) -> FeatureParams:
    base = FeatureParams()
    if not cfg:
        return base
    return FeatureParams(
        chunk_duration=float(cfg.get("chunk_duration", base.chunk_duration)),
        target_sr=int(cfg.get("target_sr", base.target_sr)),
        win_ms=float(cfg.get("win_ms", base.win_ms)),
        hop_ms=float(cfg.get("hop_ms", base.hop_ms)),
        n_mels=int(cfg.get("n_mels", base.n_mels)),
    )
