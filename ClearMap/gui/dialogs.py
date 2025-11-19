# -*- coding: utf-8 -*-
"""
dialogs
=======

All the independent popup dialogs used by the GUI
"""

__author__ = 'Charly Rousseau <charly.rousseau@icm-institute.org>'
__license__ = 'GPLv3 - GNU General Public License v3 (see LICENSE.txt)'
__copyright__ = 'Copyright © 2022 by Charly Rousseau'
__webpage__ = 'https://idisco.info'
__download__ = 'https://github.com/ClearAnatomics/ClearMap'

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (QLabel, QDialogButtonBox, QDialog, QHBoxLayout, QVBoxLayout, QFormLayout,
                             QPushButton, QLineEdit, QScrollArea, QGroupBox, QCheckBox, QComboBox)

from ClearMap.IO.assets_constants import DATA_CONTENT_TYPES


class AboutDialog(QDialog):
    def __init__(self, about_info, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"About {about_info.software_name}")

        about_label = QLabel()
        about_label.setOpenExternalLinks(True)
        about_label.setTextInteractionFlags(Qt.TextBrowserInteraction)
        about_label.setText(about_info.to_html())

        layout = QVBoxLayout()
        layout.addWidget(about_label)
        self.setLayout(layout)


class RenameChannelsDialog(QDialog):
    def __init__(self, channels, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Redefine channels")
        self.channels = channels
        self.new_channels = {}

        # Set up the layout
        layout = QVBoxLayout()
        form_layout = QFormLayout()

        # Create input fields for each channel
        self.channel_inputs = {}
        for channel in channels:
            # channel_layout = QHBoxLayout()
            name_label = QLabel(f"New name for '{channel}':")
            name_input = QLineEdit()
            form_layout.addRow(name_label, name_input)

            data_type_label = QLabel("Data type:")
            data_type_input = QComboBox()
            data_type_input.addItems(DATA_CONTENT_TYPES)
            form_layout.addRow(data_type_label, data_type_input)

            self.channel_inputs[channel] = (name_input, data_type_input)

        layout.addLayout(form_layout)

        # Add OK and Cancel buttons
        button_layout = QHBoxLayout()

        ok_button = QPushButton("OK")
        ok_button.clicked.connect(self.accept)
        button_layout.addWidget(ok_button)
        cancel_button = QPushButton("Cancel")
        cancel_button.clicked.connect(self.reject)
        button_layout.addWidget(cancel_button)

        layout.addLayout(button_layout)

        self.setLayout(layout)
        self.show()

    def accept(self):
        # Collect the new names and data types
        for channel, (name_input, data_type_input) in self.channel_inputs.items():
            new_name = name_input.text()
            data_type = data_type_input.currentText()
            self.new_channels[channel] = (new_name, data_type)
        super().accept()

    def get_new_channels(self):
        return self.new_channels


class VerifyRenamingDialog(QDialog):
    def __init__(self, files, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Verify Renaming")
        self.files = files
        self.selected_files = []

        # Set up the layout
        layout = QVBoxLayout()
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        group_box = QGroupBox()
        group_layout = QVBoxLayout()

        # Create checkboxes for each file
        self.checkboxes = {}
        for old_name, new_name in files.items():
            checkbox = QCheckBox(f"{old_name} -> {new_name}")
            checkbox.setChecked(True)
            self.checkboxes[(old_name, new_name)] = checkbox
            group_layout.addWidget(checkbox)

        group_box.setLayout(group_layout)
        scroll_area.setWidget(group_box)
        layout.addWidget(scroll_area)

        # Add OK and Cancel buttons
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

        self.setLayout(layout)

    def accept(self):
        # Collect the selected files
        for (old_name, new_name), checkbox in self.checkboxes.items():
            if checkbox.isChecked():
                self.selected_files.append((old_name, new_name))
        super().accept()

    def get_selected_files(self):
        return self.selected_files
