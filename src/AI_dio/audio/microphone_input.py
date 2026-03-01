import logging
from pathlib import Path
from typing import Tuple

import numpy as np
import sounddevice as sd

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
ROOT: Path = Path(__file__).resolve().parents[3]


def microphone_input(
    record_sec: int = 5,
    channels: int = 1,
    rate: int = 44100,
) -> Tuple[np.ndarray, int]:
    """Record audio from the default microphone.

    Args:
        record_sec: Recording duration in seconds.
        channels: Number of audio channels (1 = mono, 2 = stereo).
        rate: Sampling rate in Hz.

    Returns:
        A tuple ``(audio, sample_rate)`` where ``audio`` is a 1D float32 NumPy
        array and ``sample_rate`` is the recording sample rate.
    """
    logging.info("Recording...")
    audio: np.ndarray = sd.rec(
        int(record_sec * rate), samplerate=rate, channels=channels, dtype="float32"
    )
    sd.wait()
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    logging.info("Finished recording.")
    return audio, rate


if __name__ == "__main__":
    microphone_input()
