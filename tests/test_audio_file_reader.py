import numpy as np
import pytest
import soundfile as sf

from AI_dio.audio.audio_file_reader import (
    compute_log_mel_spectrogram,
    get_sound_parameters,
    read_sound,
)


def test_log_mel_spectrogram_shape():
    sample_rate = 44100
    duration = 1.0
    t = np.linspace(0, duration, int(sample_rate * duration))
    audio = 0.5 * np.sin(2 * np.pi * 440 * t)
    mel = compute_log_mel_spectrogram(
        audio_data=audio,
        sample_rate=sample_rate,
        mel_band_num=128,
        sample_frame_length=2048,
        hop_length=512,
    )
    params = get_sound_parameters(audio, sample_rate)
    assert pytest.approx(params["avg_volume"], 0.1) == 0.35
    assert pytest.approx(params["peak_amplitude"], 0.01) == 0.5
    assert mel.shape[0] == 128
    assert mel.ndim == 2


def test_sound_parameters_values():
    sample_rate = 44100
    audio = np.ones(sample_rate) * 0.5
    params = get_sound_parameters(audio, sample_rate)

    assert params["sample_rate"] == 44100
    assert pytest.approx(params["duration_sec"], 0.01) == 1.0
    assert pytest.approx(params["avg_volume"], 0.01) == 0.5
    assert pytest.approx(params["peak_amplitude"], 0.01) == 0.5
    assert pytest.approx(params["loudness_db"], 0.01) == -6.02


def test_read_sound_empty_file(tmp_path):
    empty_file = tmp_path / "empty.wav"
    empty_file.touch()
    with pytest.raises(ValueError, match="Audio file is empty"):
        read_sound(empty_file)


def test_read_sound_corrupted_file(tmp_path):
    corrupted = tmp_path / "corrupted.wav"
    corrupted.write_bytes(b"\x00\xff\x00\xffNOTAUDIO")
    with pytest.raises(ValueError, match="Failed to load audio"):
        read_sound(corrupted)


def test_read_sound_very_short_file(tmp_path):
    short = tmp_path / "short.wav"
    data = np.array([0.0], dtype=np.float32)
    sf.write(short, data, 44100)
    with pytest.raises(ValueError, match="Audio file too short"):
        read_sound(short)
