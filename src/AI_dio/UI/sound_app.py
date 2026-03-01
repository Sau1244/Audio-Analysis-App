import sys
from pathlib import Path

from audio_info import AudioInfo
from controls import Controls
from header import Header
from model_info import ModelInfo
from plot_area import PlotArea
from PySide6.QtCore import QUrl
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QMainWindow,
    QMessageBox,
    QVBoxLayout,
    QWidget,
)

ROOT: Path = Path(__file__).resolve().parents[3]


class SoundApp(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("The Best Sound App!")
        self.setFixedSize(900, 600)

        main_widget = QWidget()
        layout = QVBoxLayout(main_widget)

        middle_section = QHBoxLayout()
        info_panel = QVBoxLayout()

        self.build_menu()

        self.header = Header()
        self.controls = Controls()
        self.audio_info = AudioInfo()
        self.model_info = ModelInfo()
        self.plot_area = PlotArea()

        self.controls.signal_file_path.connect(
            lambda name: self.header.file_name_label.setText(name)
        )
        self.controls.signal_status.connect(
            lambda status: self.header.status_name_label.setText(status)
        )
        self.controls.signal_reset.connect(self.reset_all)
        self.controls.signal_reset_info.connect(self.reset_info)
        self.controls.signal_reset_model_info.connect(self.reset_model_info)
        self.controls.signal_audio_info.connect(self.display_audio_info)
        self.controls.signal_model_info.connect(self.display_model_info)
        self.controls.signal_update_plots.connect(self.update_plots)

        layout.addWidget(self.header)
        middle_section.addWidget(self.controls)
        info_panel.addWidget(self.audio_info)
        info_panel.addWidget(self.model_info)
        middle_section.addLayout(info_panel)
        layout.addLayout(middle_section)
        layout.addWidget(self.plot_area)

        self.setCentralWidget(main_widget)

    def build_menu(self):
        exit_action, about_action = self.build_menu_action()

        menu_bar = self.menuBar()

        menu_file = menu_bar.addMenu("&File")
        menu_help = menu_bar.addMenu("&Help")

        menu_file.addAction(exit_action)
        menu_help.addAction(about_action)

        self.statusBar()

    def build_menu_action(self):
        exit_action = QAction("&Exit", self)
        about_action = QAction("&About", self)

        exit_action.setStatusTip("Exit app")
        about_action.setStatusTip("About app")

        exit_action.triggered.connect(self.close)
        about_action.triggered.connect(self.show_about)

        return exit_action, about_action

    def display_audio_info(self, sound_params):
        self.audio_info.sample_rate_value_label.setText(
            f"{sound_params['sample_rate']} Hz"
        )
        self.audio_info.duration_value_label.setText(
            f"{sound_params['duration_sec']:.2f} s"
        )
        self.audio_info.volume_value_label.setText(f"{sound_params['avg_volume']:.4f}")
        self.audio_info.peak_amplitude_label.setText(
            f"{sound_params['peak_amplitude']:.4f}"
        )
        self.audio_info.loudness_value_label.setText(
            f"{sound_params['loudness_db']:.2f} dB"
        )

    def display_model_info(self, result):
        self.model_info.result_value_label.setText(result.label)
        self.model_info.spoof_value_label.setText(f"{result.score * 100:0.2f}%")
        self.model_info.min_value_label.setText(f"{min(result.scores) * 100:0.2f}%")
        self.model_info.max_value_label.setText(f"{max(result.scores) * 100:0.2f}%")
        self.model_info.windows_value_label.setText(f"{result.window_sec} s")
        self.model_info.threshold_value_label.setText(f"{result.threshold * 100}%")

    def update_plots(self):
        self.plot_area.update_waveform(Path(f"{ROOT}/audio_output_files/waveform.png"))
        self.plot_area.update_spectrogram(
            Path(f"{ROOT}/audio_output_files/spectrogram.png")
        )

    def reset_model_info(self):
        self.model_info.result_value_label.setText("---")
        self.model_info.spoof_value_label.setText("---")
        self.model_info.min_value_label.setText("---")
        self.model_info.max_value_label.setText("---")
        self.model_info.windows_value_label.setText("---")
        self.model_info.threshold_value_label.setText("---")

    def reset_info(self):
        self.header.file_name_label.setText("---")
        self.header.status_name_label.setText("---")
        self.audio_info.sample_rate_value_label.setText("---")
        self.audio_info.duration_value_label.setText("---")
        self.audio_info.volume_value_label.setText("---")
        self.audio_info.peak_amplitude_label.setText("---")
        self.audio_info.loudness_value_label.setText("---")
        self.plot_area.waveform_label.clear()
        self.plot_area.spectrogram_label.clear()

    def reset_all(self):
        self.controls.file_path = None
        self.controls.is_microphone_used = None
        self.controls.current_time_label.setText("0:00")
        self.controls.media.setSource(QUrl())
        self.controls.max_time_label.setText("0:00")
        self.controls.button_play.setHidden(False)
        self.controls.button_pause.setHidden(True)
        self.controls.set_media_enabled(False)
        self.reset_info()
        self.reset_model_info()

    @staticmethod
    def show_about():
        msg = QMessageBox()
        msg.setWindowTitle("About")
        msg.setText('<p style="margin-left: 20px; font-weight: bold">Authors:</p>')
        msg.exec()


if __name__ == "__main__":
    app = QApplication(sys.argv)

    with open(f"{ROOT}/src/AI_dio/UI/styles/gui_styles.qss", "r") as f:
        app.setStyleSheet(f.read())

    window = SoundApp()
    window.show()
    sys.exit(app.exec())
