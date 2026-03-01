from PySide6.QtWidgets import QGroupBox, QHBoxLayout, QLabel, QWidget


class Header(QWidget):
    def __init__(self):
        super().__init__()

        header_box = QGroupBox("Basic info")
        header_box.setFixedHeight(64)
        box_layout = QHBoxLayout()
        main_layout = QHBoxLayout(self)

        file_label = QLabel("File: ")
        file_label.setFixedWidth(26)
        self.file_name_label = QLabel("---")
        status_label = QLabel("Status: ")
        status_label.setFixedWidth(42)
        self.status_name_label = QLabel("---")

        box_layout.addWidget(file_label)
        box_layout.addWidget(self.file_name_label)
        box_layout.addWidget(status_label)
        box_layout.addWidget(self.status_name_label)

        header_box.setLayout(box_layout)
        main_layout.addWidget(header_box)
