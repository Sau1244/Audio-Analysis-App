import os
from typing import Tuple

from PySide6.QtCore import Qt, QThread, QUrl, Signal
from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer
from PySide6.QtWidgets import (
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QStyle,
    QVBoxLayout,
    QWidget,
)
from worker_audio import WorkerAudio


class Controls(QWidget):
    signal_file_path = Signal(str)
    signal_status = Signal(str)
    signal_reset = Signal()
    signal_reset_info = Signal()
    signal_reset_model_info = Signal()
    signal_audio_info = Signal(object)
    signal_model_info = Signal(object)
    signal_update_plots = Signal()

    def __init__(self):
        super().__init__()

        self.file_path = None
        self.is_microphone_used = None
        self.thread = None
        self.worker_audio = None

        self.media = QMediaPlayer()
        self.output = QAudioOutput()
        self.media.setAudioOutput(self.output)
        style = self.style()

        controls_box = QGroupBox("Controls")
        media_box = QGroupBox("Media Player")

        box_layout = QVBoxLayout()
        media_layout = QVBoxLayout()

        sound_source_layout = QHBoxLayout()
        app_controls_layout = QHBoxLayout()
        slider_layout = QHBoxLayout()
        media_controls_layout = QHBoxLayout()

        main_layout = QVBoxLayout(self)

        self.button_load_file = QPushButton("Load File")
        self.button_use_microphone = QPushButton("Use microphone")
        self.button_start = QPushButton("Start")
        self.button_controls_stop = QPushButton("Stop")
        self.button_reset = QPushButton("Reset")
        self.current_time_label = QLabel("0:00")
        self.media_slider = QSlider(Qt.Horizontal)
        self.max_time_label = QLabel("0:00")
        self.button_media_stop = QPushButton("Stop")
        self.button_media_stop.setIcon(style.standardIcon(QStyle.SP_MediaStop))
        self.button_play = QPushButton("Play")
        self.button_play.setIcon(style.standardIcon(QStyle.SP_MediaPlay))
        self.button_pause = QPushButton("Pause")
        self.button_pause.setIcon(style.standardIcon(QStyle.SP_MediaPause))

        self.set_media_enabled(False)
        self.button_pause.setHidden(True)
        self.button_controls_stop.setHidden(True)

        self.button_load_file.clicked.connect(self.show_load_dialog)
        self.button_use_microphone.clicked.connect(self.microphone_in_use)
        self.button_start.clicked.connect(self.start_audio)
        self.button_controls_stop.clicked.connect(self.stop_audio)
        self.button_reset.clicked.connect(lambda: self.signal_reset.emit())
        self.button_play.clicked.connect(self.on_play_button)
        self.button_pause.clicked.connect(self.on_pause_button)
        self.button_media_stop.clicked.connect(self.on_media_stop_button)
        self.media.positionChanged.connect(self.position_changed)
        self.media.durationChanged.connect(self.duration_changed)
        self.media_slider.sliderMoved.connect(self.media.setPosition)

        sound_source_layout.addWidget(self.button_load_file)
        sound_source_layout.addWidget(self.button_use_microphone)
        app_controls_layout.addWidget(self.button_start)
        app_controls_layout.addWidget(self.button_controls_stop)
        app_controls_layout.addWidget(self.button_reset)

        slider_layout.addWidget(self.current_time_label)
        slider_layout.addWidget(self.media_slider)
        slider_layout.addWidget(self.max_time_label)
        media_controls_layout.addWidget(self.button_play)
        media_controls_layout.addWidget(self.button_pause)
        media_controls_layout.addWidget(self.button_media_stop)

        box_layout.addLayout(sound_source_layout)
        box_layout.addLayout(app_controls_layout)
        media_layout.addLayout(slider_layout)
        media_layout.addLayout(media_controls_layout)

        controls_box.setLayout(box_layout)
        media_box.setLayout(media_layout)
        main_layout.addWidget(controls_box)
        main_layout.addWidget(media_box)

    def show_load_dialog(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Audio File",
            "",
            "Audio files (*.wav *.mp3 *.flax *.ogg *.m4a *.aiff)",
        )

        if path:
            self.signal_reset.emit()
            self.is_microphone_used = False
            self.file_path = path
            self.signal_file_path.emit(os.path.basename(self.file_path))
            self.set_media_enabled(True)
            self.media.setSource(QUrl.fromLocalFile(self.file_path))
            self.signal_status.emit("File Loaded")

    def microphone_in_use(self):
        self.signal_reset.emit()
        self.is_microphone_used = True
        self.signal_status.emit("Microphone in use")

    def set_buttons_enabled(self, option: bool):
        self.button_load_file.setEnabled(option)
        self.button_use_microphone.setEnabled(option)
        self.button_reset.setEnabled(option)
        self.button_start.setEnabled(option)

        if not self.is_microphone_used:
            self.button_controls_stop.setEnabled(option)

    def set_media_enabled(self, option: bool):
        self.media_slider.setEnabled(option)
        self.button_media_stop.setEnabled(option)
        self.button_play.setEnabled(option)
        self.button_pause.setEnabled(option)

    def on_play_button(self):
        self.media.play()
        self.button_play.setHidden(True)
        self.button_pause.setHidden(False)

    def on_pause_button(self):
        self.media.pause()
        self.button_play.setHidden(False)
        self.button_pause.setHidden(True)

    def on_media_stop_button(self):
        self.media.stop()
        self.button_play.setHidden(False)
        self.button_pause.setHidden(True)

    @staticmethod
    def convert_to_time(time: int) -> Tuple[int, int]:
        full_audio_seconds = time // 1000
        minutes = full_audio_seconds // 60
        seconds = full_audio_seconds % 60
        return minutes, seconds

    def position_changed(self, position: int):
        self.media_slider.setValue(position)
        minutes, seconds = self.convert_to_time(position)
        self.current_time_label.setText(f"{minutes}:{seconds:02d}")

    def duration_changed(self, duration: int):
        self.media_slider.setMaximum(duration)
        minutes, seconds = self.convert_to_time(duration)
        self.max_time_label.setText(f"{minutes}:{seconds:02d}")

    def start_audio(self):
        self.set_buttons_enabled(False)

        self.thread = QThread()
        self.worker_audio = WorkerAudio(self.is_microphone_used, self.file_path)
        self.worker_audio.moveToThread(self.thread)

        self.thread.started.connect(self.worker_audio.run_analysis)

        self.worker_audio.signal_status.connect(self.signal_status)
        self.worker_audio.signal_audio_info.connect(self.signal_audio_info)
        self.worker_audio.signal_reset_info.connect(self.signal_reset_info)
        self.worker_audio.signal_model_info.connect(self.signal_model_info)
        self.worker_audio.signal_reset_model_info.connect(self.signal_reset_model_info)
        self.worker_audio.signal_update_plots.connect(self.signal_update_plots)
        self.worker_audio.signal_reset.connect(self.signal_reset)
        self.worker_audio.signal_reset_info.connect(self.signal_reset_info)

        self.worker_audio.signal_finished.connect(
            lambda: self.set_buttons_enabled(True)
        )
        self.worker_audio.signal_finished.connect(self.thread.quit)
        self.worker_audio.signal_finished.connect(self.worker_audio.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)

        self.thread.start()

        if self.is_microphone_used:
            self.button_start.setHidden(True)
            self.button_controls_stop.setHidden(False)

    def stop_audio(self):
        self.worker_audio.is_recording = False
        self.button_start.setHidden(False)
        self.button_controls_stop.setHidden(True)
        self.worker_audio = None
