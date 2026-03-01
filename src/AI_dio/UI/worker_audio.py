from pathlib import Path

import numpy as np
import torch
from PySide6.QtCore import QObject, Signal

from AI_dio.audio.audio_file_reader import (
    compute_log_mel_spectrogram,
    get_sound_parameters,
    plot_melspectrogram,
    plot_waveform,
    read_sound,
)
from AI_dio.audio.microphone_input import microphone_input
from AI_dio.inference import predict_audio, predict_file

ROOT = Path(__file__).parents[3].resolve()
CHECKPOINT_PATH = ROOT / "checkpoints/model_best.pt"
print(CHECKPOINT_PATH)


class WorkerAudio(QObject):
    signal_status = Signal(str)
    signal_audio_info = Signal(object)
    signal_model_info = Signal(object)
    signal_update_plots = Signal()
    signal_reset = Signal()
    signal_finished = Signal()
    signal_reset_info = Signal()
    signal_reset_model_info = Signal()

    def __init__(self, is_microphone_used, file_path):
        super().__init__()
        self.is_microphone_used = is_microphone_used
        self.file_path = file_path
        self.is_recording = True

    def run_analysis(self):
        try:
            if self.is_microphone_used:
                self.signal_reset_info.emit()
                self.signal_reset_model_info.emit()
                full_audio = None
                rate = None
                while self.is_recording:
                    self.signal_status.emit("Recording...")
                    audio, rate = microphone_input(3)
                    if full_audio is None:
                        full_audio = audio
                    else:
                        full_audio = np.concatenate((full_audio, audio))
                    sound_params = get_sound_parameters(full_audio, rate)
                    plot_waveform(full_audio)
                    log_mel = compute_log_mel_spectrogram(full_audio, rate)
                    plot_melspectrogram(log_mel, rate)
                    self.signal_update_plots.emit()
                    self.signal_audio_info.emit(sound_params)

                self.signal_status.emit("Recording ended")

                if full_audio is not None:
                    self.signal_status.emit("Analyzing...")
                    audio_tensor = torch.tensor(full_audio)
                    result = predict_audio(
                        checkpoint=CHECKPOINT_PATH, audio=audio_tensor, sample_rate=rate
                    )
                    self.signal_model_info.emit(result)
                    self.signal_status.emit("Analysis ended")
            elif self.file_path:
                self.signal_status.emit("Analyzing...")
                _, sound_params = read_sound(Path(self.file_path))
                self.signal_audio_info.emit(sound_params)
                self.signal_update_plots.emit()

                result = predict_file(checkpoint=CHECKPOINT_PATH, wav=self.file_path)
                self.signal_model_info.emit(result)
                self.signal_status.emit("Analysis ended")
            else:
                self.signal_status.emit("Source not selected")
        except FileNotFoundError:
            self.signal_reset.emit()
            self.signal_status.emit("Audio file not found")
        finally:
            self.signal_finished.emit()
