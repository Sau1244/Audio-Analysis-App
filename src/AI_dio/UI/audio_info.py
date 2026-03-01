from PySide6.QtWidgets import QGroupBox, QHBoxLayout, QLabel, QVBoxLayout, QWidget


class AudioInfo(QWidget):
    def __init__(self):
        super().__init__()

        info_box = QGroupBox("Audio Info")
        info_box.setFixedWidth(300)
        box_layout = QVBoxLayout()
        first_row_layout = QHBoxLayout()
        second_row_layout = QHBoxLayout()
        third_row_layout = QHBoxLayout()
        main_layout = QVBoxLayout(self)

        sample_rate_label = QLabel("Sample rate: ")
        sample_rate_label.setFixedWidth(91)
        self.sample_rate_value_label = QLabel("---")
        self.sample_rate_value_label.setFixedWidth(59)

        duration_label = QLabel("Duration: ")
        duration_label.setFixedWidth(55)
        self.duration_value_label = QLabel("---")

        volume_label = QLabel("RMS (Volume): ")
        volume_label.setFixedWidth(91)
        self.volume_value_label = QLabel("---")
        self.volume_value_label.setFixedWidth(59)

        peak_amplitude_label = QLabel("Peak amplitude: ")
        peak_amplitude_label.setFixedWidth(91)
        self.peak_amplitude_label = QLabel("---")

        loudness_label = QLabel("Loudness: ")
        loudness_label.setFixedWidth(55)
        self.loudness_value_label = QLabel("---")

        first_row_layout.addWidget(sample_rate_label)
        first_row_layout.addWidget(self.sample_rate_value_label)
        first_row_layout.addWidget(duration_label)
        first_row_layout.addWidget(self.duration_value_label)

        second_row_layout.addWidget(volume_label)
        second_row_layout.addWidget(self.volume_value_label)
        second_row_layout.addWidget(loudness_label)
        second_row_layout.addWidget(self.loudness_value_label)

        third_row_layout.addWidget(peak_amplitude_label)
        third_row_layout.addWidget(self.peak_amplitude_label)

        box_layout.addLayout(first_row_layout)
        box_layout.addLayout(second_row_layout)
        box_layout.addLayout(third_row_layout)

        info_box.setLayout(box_layout)
        main_layout.addWidget(info_box)
