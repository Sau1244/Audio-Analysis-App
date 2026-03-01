import logging
from pathlib import Path
from typing import Dict, Tuple

import librosa
import librosa.display
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
ROOT: Path = Path(__file__).resolve().parents[3]
matplotlib.use("agg")


def plot_waveform(
    audio_channel: np.ndarray,
    output_path: Path = ROOT / "audio_output_files/waveform.png",
) -> None:
    """Plot a waveform and save it to disk.

    Args:
        audio_channel: 1D audio signal (float array).
        output_path: Target PNG path for the rendered waveform.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 4))
    plt.plot(audio_channel)
    plt.title("Audio signal waveform")
    plt.savefig(str(output_path))
    plt.close()


def plot_melspectrogram(
    log_mel: np.ndarray,
    sample_rate: int,
    hop_length: int = 512,
    output_path: Path = ROOT / "audio_output_files/spectrogram.png",
) -> None:
    """Plot a log-mel spectrogram and save it to disk.

    Args:
        log_mel: 2D log-mel spectrogram array (mel bins x frames).
        sample_rate: Audio sample rate in Hz.
        hop_length: Hop length used when computing the spectrogram.
        output_path: Target PNG path for the rendered spectrogram.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(
        log_mel, sr=sample_rate, hop_length=hop_length, x_axis="time", y_axis="mel"
    )
    plt.colorbar(format="%+2.0f dB")
    plt.title("Log-Mel Spectrogram")
    plt.savefig(str(output_path))
    plt.close()


def compute_log_mel_spectrogram(
    audio_data: np.ndarray,
    sample_rate: int,
    mel_band_num: int = 128,
    sample_frame_length: int = 2048,
    hop_length: int = 512,
) -> np.ndarray:
    """Compute a log-mel spectrogram for a mono signal.

    Args:
        audio_data: 1D mono audio array.
        sample_rate: Sampling rate in Hz.
        mel_band_num: Number of mel filter banks.
        sample_frame_length: FFT window size.
        hop_length: Hop length between analysis frames.

    Returns:
        Log-mel spectrogram as a 2D NumPy array.
    """
    S = librosa.feature.melspectrogram(
        y=audio_data,
        sr=sample_rate,
        n_fft=sample_frame_length,
        hop_length=hop_length,
        n_mels=mel_band_num,
    )
    log_mel = librosa.power_to_db(S, ref=np.max)
    return log_mel


def get_sound_parameters(data: np.ndarray, sr: int) -> Dict[str, float]:
    """Calculate basic audio statistics.

    Args:
        data: Audio signal (mono or stereo). Stereo inputs are mixed down.
        sr: Sampling rate in Hz.

    Returns:
        Dict with sample_rate, duration_sec, avg_volume (RMS), peak_amplitude,
        and loudness_db.
    """
    if len(data.shape) == 2:
        data = data.mean(axis=1)
    rms = float(np.sqrt(np.mean(data**2)))
    peak = float(np.max(np.abs(data)))
    epsilon = 1e-10
    loudness_db = float(20 * np.log10(rms + epsilon))
    duration = float(len(data) / sr)
    logging.info(f"Sample rate: {sr} Hz")
    logging.info(f"Duration: {duration:.2f} sec")
    logging.info(f"RMS (volume): {rms:.4f}")
    logging.info(f"Peak amplitude: {peak:.4f}")
    logging.info(f"Loudness (dB): {loudness_db:.2f} dB")
    return {
        "sample_rate": sr,
        "duration_sec": duration,
        "avg_volume": rms,
        "peak_amplitude": peak,
        "loudness_db": loudness_db,
    }


def read_sound(
    file: Path = ROOT / "audio_samples/sample-3s.wav",
    plot_waveform_flag: bool = True,
    plot_melspectrogram_flag: bool = True,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """Load an audio file, optionally plot visuals, and return data with stats.

    Args:
        file: Path to the audio file to read.
        plot_waveform_flag: Whether to plot a waveform plot.
        plot_melspectrogram_flag: Whether to plot a log-mel spectrogram plot.

    Returns:
        Tuple of (audio_data, sound_parameters) where ``sound_parameters``
        contains basic statistics.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file is empty, fails to load, or is too short.
    """
    if not file.exists():
        raise FileNotFoundError(f"Audio file not found: {file}")
    if file.stat().st_size == 0:
        raise ValueError(f"Audio file is empty: {file}")
    audio_data: np.ndarray
    sample_rate: int
    try:
        audio_data, sample_rate = librosa.load(str(file), sr=None, mono=True)
    except Exception as e:
        raise ValueError(f"Failed to load audio file: {file}") from e
    if len(audio_data) < 2048:
        raise ValueError(
            f"Audio file too short: {len(audio_data)} samples, need at least 2048"
        )

    if plot_waveform_flag:
        plot_waveform(audio_data)
    if plot_melspectrogram_flag:
        log_mel = compute_log_mel_spectrogram(
            audio_data,
            sample_rate,
            mel_band_num=128,
            sample_frame_length=2048,
            hop_length=512,
        )
        plot_melspectrogram(log_mel, sample_rate)
    sound_parameters = get_sound_parameters(audio_data, sample_rate)
    return audio_data, sound_parameters


if __name__ == "__main__":
    read_sound()
