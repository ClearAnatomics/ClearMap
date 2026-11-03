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

import os
from pathlib import Path
from typing import TYPE_CHECKING, Sequence

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (QLabel, QDialogButtonBox, QDialog, QHBoxLayout, QVBoxLayout, QFormLayout,
                             QPushButton, QLineEdit, QGroupBox, QCheckBox, QComboBox, QListWidget, QTableWidget,
                             QTableWidgetItem)

from ClearMap.IO.FileUtils import FileRenameError, Rename, check_renames
from ClearMap.IO.assets_constants import DATA_CONTENT_TYPES

if TYPE_CHECKING:
    from ClearMap.IO.channel_file_renaming import NotRenamed

RENAMES_TABLE_HEADERS = ['Rename', 'Folder', 'Current name', 'New name']
FOLDER_SEPARATORS = ('/', '\\')  # Of every platform: a new name typed on one does not hold a folder on another
SELECTED_COLUMN, FOLDER_COLUMN, CURRENT_NAME_COLUMN, NEW_NAME_COLUMN = range(len(RENAMES_TABLE_HEADERS))


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
    """
    Let the user vet a batch of renames before anything moves: each rename can be unticked (left as it is) and its
    new name edited. The files which are left alone whatever the user does are listed with the reason.

    Cancel is the default button: renaming files is only done on an explicit choice.
    The renames are checked as a whole (see :func:`vetted_renames`) when the user accepts: the dialog stays open
    and says why if they cannot be applied.
    """
    def __init__(self, renames: Sequence[Rename], *, not_renamed: Sequence['NotRenamed'] = (),
                 title: str = 'Verify renaming', message: str = '', accept_text: str = 'Rename', parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setModal(True)
        self.renames = list(renames)
        self.selected_renames: list[Rename] = []
        root = _common_folder([rename.source for rename in self.renames] + [entry.path for entry in not_renamed])

        layout = QVBoxLayout(self)
        if message:
            message_label = QLabel(message)
            message_label.setWordWrap(True)
            layout.addWidget(message_label)
        if root is not None:
            layout.addWidget(QLabel(f'In {root}'))

        self.renames_table = QTableWidget(len(self.renames), len(RENAMES_TABLE_HEADERS))
        self.renames_table.setHorizontalHeaderLabels(RENAMES_TABLE_HEADERS)
        for row, rename in enumerate(self.renames):
            self._fill_rename_row(row, rename, root)
        self.renames_table.resizeColumnsToContents()
        self.renames_table.horizontalHeader().setStretchLastSection(True)  # The new names get the room left
        self.renames_table.setVisible(bool(self.renames))
        layout.addWidget(self.renames_table)

        if not_renamed:
            not_renamed_box = QGroupBox('Left as they are')
            not_renamed_layout = QVBoxLayout(not_renamed_box)
            not_renamed_list = QListWidget()
            not_renamed_list.addItems([f'{_relative_to(entry.path, root)}: {entry.reason}' for entry in not_renamed])
            not_renamed_layout.addWidget(not_renamed_list)
            layout.addWidget(not_renamed_box)

        self.error_label = QLabel()
        self.error_label.setWordWrap(True)
        self.error_label.setStyleSheet('color: red')
        self.error_label.setVisible(False)
        layout.addWidget(self.error_label)

        button_box = QDialogButtonBox()
        accept_button = button_box.addButton(accept_text, QDialogButtonBox.AcceptRole)
        cancel_button = button_box.addButton(QDialogButtonBox.Cancel)
        accept_button.setAutoDefault(False)
        cancel_button.setDefault(True)  # Enter does not rename anything
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

    def _fill_rename_row(self, row: int, rename: Rename, root: Path | None) -> None:
        selected_item = QTableWidgetItem()
        selected_item.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled)
        selected_item.setCheckState(Qt.Checked)
        self.renames_table.setItem(row, SELECTED_COLUMN, selected_item)
        folder = _relative_to(rename.source.parent, root)
        for column, text in ((FOLDER_COLUMN, '' if folder == Path('.') else str(folder)),  # '.': the root itself
                             (CURRENT_NAME_COLUMN, rename.source.name)):
            read_only_item = QTableWidgetItem(text)
            read_only_item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            self.renames_table.setItem(row, column, read_only_item)
        self.renames_table.setItem(row, NEW_NAME_COLUMN, QTableWidgetItem(rename.target.name))

    def accept(self):
        choices = [(self.renames_table.item(row, SELECTED_COLUMN).checkState() == Qt.Checked,
                    self.renames_table.item(row, NEW_NAME_COLUMN).text())
                   for row in range(len(self.renames))]
        try:
            self.selected_renames = vetted_renames(self.renames, choices)
        except FileRenameError as err:
            self.error_label.setText(str(err))
            self.error_label.setVisible(True)
            return
        super().accept()


def vetted_renames(renames: Sequence[Rename], choices: Sequence[tuple[bool, str]]) -> list[Rename]:
    """
    The renames as vetted by the user: the ones they kept, each to the new name they chose (in the same folder).

    Parameters
    ----------
    renames : Sequence[Rename]
        The renames proposed to the user.
    choices : Sequence[tuple[bool, str]]
        For each rename, whether it is kept and its new name (a file name, not a path).

    Returns
    -------
    list[Rename]
        The renames to apply, checked as a whole (see :func:`ClearMap.IO.FileUtils.check_renames`).

    Raises
    ------
    FileRenameError
        If a new name is not a file name, or if the renames cannot be applied as a whole.
    """
    kept = []
    for rename, (is_kept, new_name) in zip(renames, choices, strict=True):
        if not is_kept:
            continue
        new_name = new_name.strip()
        if not _is_file_name(new_name):
            raise FileRenameError(f'{new_name!r} is not a file name (the new name of {rename.source.name}): '
                                  f'it cannot be empty nor hold a folder')
        kept.append(Rename(rename.source, rename.source.with_name(new_name)))
    return check_renames(kept)


def _is_file_name(name: str) -> bool:
    """Whether name names a file in a folder (not a path, on any platform)"""
    return name not in ('', '.', '..') and not any(separator in name for separator in FOLDER_SEPARATORS)


def _common_folder(paths: Sequence[Path]) -> Path | None:
    """The folder which holds all the paths (None if there are none, or if they share no folder)"""
    if not paths:
        return None
    try:
        common = Path(os.path.commonpath([path.parent for path in paths]))
    except ValueError:  # e.g. on different drives
        return None
    return common if common.parts else None


def _relative_to(path: Path, root: Path | None) -> Path:
    if root is not None and path.is_relative_to(root):
        return path.relative_to(root)
    return path


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

