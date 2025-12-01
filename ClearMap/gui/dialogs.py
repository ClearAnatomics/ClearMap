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


class ResourceTypeToFolderDialog(QDialog):
    """
    Simple (k: v) editor for resource_type_to_folder.

    - Left column: resource_type (read-only label).
    - Right column: editable folder (relative path, may be empty '').
    - Optional: checkbox to decide whether to migrate existing folders.
    """
    def __init__(self, mapping: dict[str, str], parent=None):
        super().__init__(parent)
        self.setWindowTitle('Workspace resource sub-folders')
        self.setModal(True)

        self._rows: dict[str, QLineEdit] = {}

        main_layout = QVBoxLayout(self)

        info = QLabel('Configure where each resource type is stored, relative to the workspace directory.\n'
                      'Leave empty to keep files in the main experiment folder.')
        info.setWordWrap(True)
        main_layout.addWidget(info)

        form = QFormLayout()
        for rtype, folder in mapping.items():
            label = QLabel(rtype)
            label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

            edit = QLineEdit(str(folder or ''))
            edit.setPlaceholderText('<root>')
            form.addRow(label, edit)
            self._rows[rtype] = edit

        main_layout.addLayout(form)

        # Whether to migrate existing folders on disk
        self.migrate_checkbox = QCheckBox('Move existing data into new folders (migrate on disk)')
        self.migrate_checkbox.setChecked(False)
        main_layout.addWidget(self.migrate_checkbox)

        # Standard buttons
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        main_layout.addWidget(buttons)

    def result_mapping(self) -> dict[str, str]:
        """
        Return the edited mapping {resource_type: folder_rel_path}.
        """
        out: dict[str, str] = {}
        for rtype, edit in self._edits.items():
            val = edit.text().strip()
            out[rtype] = val
        return out

    def result(self) -> tuple[dict[str, str], bool]:
        """
        Convenience: (mapping, migrate_flag).
        """
        return self.result_mapping(), self.migrate_checkbox.isChecked()

