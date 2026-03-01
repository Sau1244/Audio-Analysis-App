from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import torch
import torch.nn.functional as F
import torchaudio.functional as AF


@dataclass
class _CodecProfile:
    name: str
    cutoff_ratio: float
    quant_bits: int
    noise_floor: float


_CODEC_PROFILES = {
    "mp3": _CodecProfile(
        name="mp3", cutoff_ratio=0.45, quant_bits=8, noise_floor=0.002
    ),
    "aac": _CodecProfile(
        name="aac", cutoff_ratio=0.48, quant_bits=10, noise_floor=0.0015
    ),
    "opus": _CodecProfile(
        name="opus", cutoff_ratio=0.40, quant_bits=7, noise_floor=0.003
    ),
}


def _ensure_mono(audio: torch.Tensor) -> torch.Tensor:
    if audio.dim() == 1:
        return audio.unsqueeze(0)
    if audio.dim() == 2 and audio.size(0) == 1:
        return audio
    if audio.dim() == 2:
        return audio.mean(dim=0, keepdim=True)
    raise ValueError(f"Unexpected audio shape: {tuple(audio.shape)}")


def _rms(audio: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(torch.mean(audio**2) + 1e-8)


def _quantize(audio: torch.Tensor, bits: int) -> torch.Tensor:
    bits = max(int(bits), 2)
    levels = float(2**bits - 1)
    audio = torch.clamp(audio, -1.0, 1.0)
    scaled = torch.round((audio + 1.0) * 0.5 * levels)
    return scaled * (2.0 / levels) - 1.0


class AudioAugmenter:
    def __init__(
        self,
        *,
        target_sr: int,
        chunk_length: int,
        cfg: Optional[dict] = None,
        augment_root: Optional[Path] = None,
    ) -> None:
        self._target_sr = target_sr
        self._chunk_length = chunk_length
        self._cfg = cfg or {}

        self._p_codec = float(self._cfg.get("p_codec", 0.4))
        self._p_resample = float(self._cfg.get("p_resample", 0.4))
        self._p_bandlimit = float(self._cfg.get("p_bandlimit", 0.4))
        self._p_compress = float(self._cfg.get("p_compress", 0.4))
        self._p_reverb = float(self._cfg.get("p_reverb", 0.25))
        self._p_noise = float(self._cfg.get("p_noise", 0.5))

        self._codec_types = list(self._cfg.get("codec_types") or ["mp3", "aac", "opus"])
        self._codec_bitrate_min = int(self._cfg.get("codec_bitrate_kbps_min", 16))
        self._codec_bitrate_max = int(self._cfg.get("codec_bitrate_kbps_max", 96))

        self._resample_min = int(self._cfg.get("resample_hz_min", 8000))
        self._resample_max = int(self._cfg.get("resample_hz_max", target_sr))
        self._resample_rates = self._cfg.get("resample_rates")

        self._bandlimit_min = float(self._cfg.get("bandlimit_hz_min", 3000.0))
        self._bandlimit_max = float(
            self._cfg.get("bandlimit_hz_max", target_sr / 2.0 - 100.0)
        )

        self._compress_threshold = float(self._cfg.get("compress_threshold_db", -20.0))
        self._compress_ratio = float(self._cfg.get("compress_ratio", 4.0))
        self._compress_window_ms = float(self._cfg.get("compress_window_ms", 20.0))

        self._reverb_ms_min = float(self._cfg.get("reverb_ms_min", 80.0))
        self._reverb_ms_max = float(self._cfg.get("reverb_ms_max", 300.0))
        self._reverb_decay_min = float(self._cfg.get("reverb_decay_min", 2.0))
        self._reverb_decay_max = float(self._cfg.get("reverb_decay_max", 6.0))
        self._reverb_reflections_min = int(self._cfg.get("reverb_reflections_min", 3))
        self._reverb_reflections_max = int(self._cfg.get("reverb_reflections_max", 6))

        self._noise_snr_min = float(self._cfg.get("noise_snr_db_min", -5.0))
        self._noise_snr_max = float(self._cfg.get("noise_snr_db_max", 20.0))

        self._apply_codec_fn = getattr(AF, "apply_codec", None)

    def _rand_uniform(self, low: float, high: float) -> float:
        if high <= low:
            return float(low)
        return float((high - low) * torch.rand(1).item() + low)

    def _rand_int(self, low: int, high: int) -> int:
        if high <= low:
            return int(low)
        return int(torch.randint(low, high + 1, (1,)).item())

    def _rand_choice(self, values: Iterable) -> Optional[object]:
        values = list(values)
        if not values:
            return None
        idx = int(torch.randint(0, len(values), (1,)).item())
        return values[idx]

    def _codec_corrupt(self, audio: torch.Tensor) -> torch.Tensor:
        if self._p_codec <= 0.0 or not self._codec_types:
            return audio
        if torch.rand(1).item() > self._p_codec:
            return audio
        codec = str(self._rand_choice(self._codec_types) or "mp3")
        bitrate = self._rand_int(self._codec_bitrate_min, self._codec_bitrate_max)
        profile = _CODEC_PROFILES.get(codec, _CODEC_PROFILES["mp3"])

        if self._apply_codec_fn is not None:
            try:
                if codec == "opus":
                    return self._apply_codec_fn(
                        audio,
                        self._target_sr,
                        format="ogg",
                        encoder="opus",
                        compression=bitrate,
                    )
                return self._apply_codec_fn(
                    audio, self._target_sr, format=codec, compression=bitrate
                )
            except Exception:
                pass

        cutoff = min(
            self._target_sr / 2.0 - 100.0, profile.cutoff_ratio * self._target_sr
        )
        audio = AF.lowpass_biquad(audio, self._target_sr, cutoff)
        audio = _quantize(audio, profile.quant_bits)
        noise = torch.randn_like(audio) * profile.noise_floor
        return torch.clamp(audio + noise, -1.0, 1.0)

    def _resample(self, audio: torch.Tensor) -> torch.Tensor:
        if self._p_resample <= 0.0:
            return audio
        if torch.rand(1).item() > self._p_resample:
            return audio
        if self._resample_rates:
            rate = int(self._rand_choice(self._resample_rates) or self._target_sr)
        else:
            rate = int(self._rand_uniform(self._resample_min, self._resample_max))
        rate = max(1000, min(rate, self._target_sr))
        if rate == self._target_sr:
            return audio
        audio = AF.resample(audio, self._target_sr, rate)
        return AF.resample(audio, rate, self._target_sr)

    def _bandlimit(self, audio: torch.Tensor) -> torch.Tensor:
        if self._p_bandlimit <= 0.0:
            return audio
        if torch.rand(1).item() > self._p_bandlimit:
            return audio
        cutoff = self._rand_uniform(self._bandlimit_min, self._bandlimit_max)
        cutoff = min(cutoff, self._target_sr / 2.0 - 100.0)
        return AF.lowpass_biquad(audio, self._target_sr, cutoff)

    def _compress(self, audio: torch.Tensor) -> torch.Tensor:
        if self._p_compress <= 0.0:
            return audio
        if torch.rand(1).item() > self._p_compress:
            return audio
        window = max(1, int(self._target_sr * self._compress_window_ms / 1000.0))
        pad = window // 2
        power = audio.unsqueeze(0) ** 2
        env = F.avg_pool1d(power, kernel_size=window, stride=1, padding=pad)
        env = env[..., : audio.size(1)]
        env = torch.sqrt(env + 1e-8)
        env_db = 20.0 * torch.log10(env + 1e-8)
        threshold = self._compress_threshold
        ratio = max(self._compress_ratio, 1.0)
        gain_db = torch.where(
            env_db > threshold,
            (threshold + (env_db - threshold) / ratio) - env_db,
            torch.zeros_like(env_db),
        )
        gain = torch.pow(10.0, gain_db / 20.0)
        return audio * gain.squeeze(0)

    def _reverb(self, audio: torch.Tensor) -> torch.Tensor:
        if self._p_reverb <= 0.0:
            return audio
        if torch.rand(1).item() > self._p_reverb:
            return audio
        length_ms = self._rand_uniform(self._reverb_ms_min, self._reverb_ms_max)
        ir_len = max(8, int(self._target_sr * length_ms / 1000.0))
        decay = self._rand_uniform(self._reverb_decay_min, self._reverb_decay_max)
        t = torch.linspace(0, length_ms / 1000.0, ir_len)
        ir = torch.exp(-decay * t)
        ir[0] = 1.0
        num_reflections = self._rand_int(
            self._reverb_reflections_min, self._reverb_reflections_max
        )
        for _ in range(num_reflections):
            delay = self._rand_int(1, ir_len - 1)
            ir[delay] += float(self._rand_uniform(0.1, 0.5))
        ir = ir / (ir.abs().max() + 1e-6)
        kernel = ir.flip(0).view(1, 1, -1)
        audio_batch = audio.unsqueeze(0)
        convolved = F.conv1d(audio_batch, kernel, padding=ir_len - 1)
        convolved = convolved[:, :, : audio.size(1)].squeeze(0)
        return torch.clamp(convolved, -1.0, 1.0)

    def _add_noise(self, audio: torch.Tensor) -> torch.Tensor:
        if self._p_noise <= 0.0:
            return audio
        if torch.rand(1).item() > self._p_noise:
            return audio
        snr_db = self._rand_uniform(self._noise_snr_min, self._noise_snr_max)
        clean_rms = _rms(audio)
        noise = torch.randn_like(audio)
        noise_rms = _rms(noise)
        if float(noise_rms) <= 0.0:
            return audio
        scale = clean_rms / (noise_rms * (10.0 ** (snr_db / 20.0)))
        mixed = audio + noise * scale
        return torch.clamp(mixed, -1.0, 1.0)

    def apply(self, audio: torch.Tensor) -> torch.Tensor:
        audio = _ensure_mono(audio).clone()
        audio = self._codec_corrupt(audio)
        audio = self._resample(audio)
        audio = self._bandlimit(audio)
        audio = self._compress(audio)
        audio = self._reverb(audio)
        audio = self._add_noise(audio)
        return torch.clamp(audio, -1.0, 1.0)
