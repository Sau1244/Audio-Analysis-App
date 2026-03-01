from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import QGroupBox, QHBoxLayout, QLabel, QWidget


class PlotArea(QWidget):
    def __init__(self):
        super().__init__()

        plot_box = QGroupBox("Plot area")
        box_layout = QHBoxLayout()
        main_layout = QHBoxLayout(self)

        self.waveform_label = QLabel()
        self.spectrogram_label = QLabel()

        self.waveform_label.setFixedSize(400, 170)
        self.spectrogram_label.setFixedSize(400, 170)

        box_layout.addWidget(self.waveform_label)
        box_layout.addWidget(self.spectrogram_label)

        plot_box.setLayout(box_layout)
        main_layout.addWidget(plot_box)

    def update_waveform(self, path: Path):
        pixmap = QPixmap(path)
        pixmap = pixmap.scaled(
            self.waveform_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.waveform_label.setPixmap(pixmap)

    def update_spectrogram(self, path: Path):
        pixmap = QPixmap(path)
        pixmap = pixmap.scaled(
            self.spectrogram_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.spectrogram_label.setPixmap(pixmap)
