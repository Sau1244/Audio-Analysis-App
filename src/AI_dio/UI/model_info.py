from PySide6.QtWidgets import QGroupBox, QHBoxLayout, QLabel, QVBoxLayout, QWidget


class ModelInfo(QWidget):
    def __init__(self):
        super().__init__()

        model_box = QGroupBox("AI Model")
        model_box.setFixedWidth(300)
        box_layout = QVBoxLayout()
        first_row_layout = QHBoxLayout()
        second_row_layout = QHBoxLayout()
        third_row_layout = QHBoxLayout()
        main_layout = QVBoxLayout(self)

        result_label = QLabel("Result: ")
        result_label.setFixedWidth(63)
        self.result_value_label = QLabel("---")
        self.result_value_label.setFixedWidth(64)

        spoof_label = QLabel("Spoof: ")
        spoof_label.setFixedWidth(60)
        self.spoof_value_label = QLabel("---")

        min_label = QLabel("Min: ")
        min_label.setFixedWidth(63)
        self.min_value_label = QLabel("---")
        self.min_value_label.setFixedWidth(64)

        max_label = QLabel("Max: ")
        max_label.setFixedWidth(60)
        self.max_value_label = QLabel("---")

        windows_label = QLabel("Windows: ")
        windows_label.setFixedWidth(63)
        self.windows_value_label = QLabel("---")
        self.windows_value_label.setFixedWidth(64)

        threshold_label = QLabel("Threshold: ")
        threshold_label.setFixedWidth(60)
        self.threshold_value_label = QLabel("---")

        first_row_layout.addWidget(result_label)
        first_row_layout.addWidget(self.result_value_label)
        first_row_layout.addWidget(spoof_label)
        first_row_layout.addWidget(self.spoof_value_label)

        second_row_layout.addWidget(min_label)
        second_row_layout.addWidget(self.min_value_label)
        second_row_layout.addWidget(max_label)
        second_row_layout.addWidget(self.max_value_label)

        third_row_layout.addWidget(windows_label)
        third_row_layout.addWidget(self.windows_value_label)
        third_row_layout.addWidget(threshold_label)
        third_row_layout.addWidget(self.threshold_value_label)

        box_layout.addLayout(first_row_layout)
        box_layout.addLayout(second_row_layout)
        box_layout.addLayout(third_row_layout)

        model_box.setLayout(box_layout)
        main_layout.addWidget(model_box)
