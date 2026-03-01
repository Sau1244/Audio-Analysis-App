import numpy as np
import sounddevice as sd

from AI_dio.audio.microphone_input import microphone_input


def test_microphone_input_params(monkeypatch):
    def mocked_rec(frames, samplerate, channels, dtype):
        return np.zeros((frames, channels), dtype=dtype)

    monkeypatch.setattr(sd, "rec", mocked_rec)
    monkeypatch.setattr(sd, "wait", lambda: None)
    audio, rate = microphone_input(record_sec=2, channels=1, rate=44100)

    assert rate == 44100
    assert audio.shape == (2 * 44100, 1)
    assert audio.dtype == np.float32
