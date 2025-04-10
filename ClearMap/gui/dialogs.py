# -*- coding: utf-8 -*-
"""
dialogs
=======

All the independent popup dialogs used by the GUI
"""
import os
import functools

from ClearMap.IO.assets_constants import DATA_CONTENT_TYPES
from PyQt5.QtCore import Qt, QSize
from PyQt5.QtGui import QPixmap, QFont, QPainter, QFontMetrics, QPen
from PyQt5.QtWidgets import (QFileDialog, QMessageBox, QLabel, QDialogButtonBox, QSplashScreen,
                             QProgressBar, QDialog, QHBoxLayout, QPushButton, QVBoxLayout, QStyle,
                             QInputDialog, QFormLayout, QLineEdit, QScrollArea, QGroupBox, QCheckBox,
                             QComboBox)

from ClearMap.config.config_loader import clean_path, ConfigLoader
from ClearMap.gui.gui_utils import UI_FOLDER, create_clearmap_widget

__author__ = 'Charly Rousseau <charly.rousseau@icm-institute.org>'
__license__ = 'GPLv3 - GNU General Public License v3 (see LICENSE.txt)'
__copyright__ = 'Copyright © 2022 by Charly Rousseau'
__webpage__ = 'https://idisco.info'
__download__ = 'https://github.com/ClearAnatomics/ClearMap'

cfg_loader = ConfigLoader('')
DISPLAY_CONFIG = cfg_loader.get_cfg('display')


def get_directory_dlg(start_folder, title="Choose the source directory"):
    diag = QFileDialog()  # REFACTOR: move to gui_utils
    opt = QFileDialog.Options(QFileDialog.DontUseNativeDialog)
    start_folder = clean_path(start_folder)
    src_folder = diag.getExistingDirectory(parent=diag, caption=title,
                                           directory=start_folder, options=opt)
    diag.close()
    return src_folder


def prompt_dialog(title, msg):
    pressed_btn = QMessageBox.question(None, title, msg)
    return pressed_btn == QMessageBox.Yes


def input_dialog(title, msg, data_type=str):
    dlg = QInputDialog()
    input_modes_map = {
        str: QInputDialog.TextInput,
        int: QInputDialog.IntInput,
        float: QInputDialog.DoubleInput
    }
    dlg.setInputMode(input_modes_map[data_type])
    dlg.setWindowTitle(title)
    dlg.setLabelText(msg)

    if dlg.exec() == QDialog.Accepted:
        output_functions = {
            str: dlg.textValue,
            int: dlg.intValue,
            float: dlg.doubleValue
        }
        return output_functions[data_type]()


def option_dialog(base_msg, msg, options, parent=None):
    index = [None]

    def get_id(lst, id_):
        lst[0] = id_
        return id_

    dlg = QDialog(parent)

    dlg.setWindowTitle('User input required')
    main_layout = QVBoxLayout()
    dlg.setLayout(main_layout)
    pixmapi = QStyle.SP_MessageBoxQuestion
    icon = dlg.style().standardIcon(pixmapi)
    icon_lbl = QLabel()
    icon_lbl.setPixmap(icon.pixmap(icon.actualSize(QSize(32, 32))))
    h1 = QHBoxLayout()
    main_layout.addLayout(h1)
    h1.addWidget(icon_lbl)
    lbl = QLabel(f'<b>{base_msg}</b>')
    h1.addWidget(lbl)
    lbl2 = QLabel(msg)
    main_layout.addWidget(lbl2)

    layout = QHBoxLayout()
    main_layout.addLayout(layout)

    for i, option in enumerate(options):
        btn = QPushButton(option, parent=dlg)
        btn.clicked.connect(functools.partial(get_id, index, i))
        btn.clicked.connect(dlg.accept)
        layout.addWidget(btn)
    dlg.exec()
    return index[0]


# REFACTOR: make class
def warning_popup(base_msg, msg=''):
    msg = msg or base_msg
    dlg = QMessageBox()
    dlg.setIcon(QMessageBox.Warning)
    dlg.setWindowTitle('Warning')
    dlg.setText(f'<b>{base_msg}</b>')
    dlg.setInformativeText(msg)
    dlg.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel)
    dlg.setDefaultButton(QMessageBox.Ok)
    return dlg.exec()


def abort_retry_popup(base_msg, msg):  # REFACTOR: duplicate code
    dlg = QMessageBox()
    dlg.setIcon(QMessageBox.Warning)
    dlg.setWindowTitle('Warning')
    dlg.setText(f'<b>{base_msg}</b>')
    dlg.setInformativeText(msg)
    dlg.setStandardButtons(QMessageBox.Abort | QMessageBox.Retry)
    dlg.setDefaultButton(QMessageBox.Abort)
    return dlg.exec()


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


# REFACTOR: make class
def make_nested_progress_dialog(title='Processing', overall_maximum=100, sub_process_name='', abort_callback=None, parent=None):
    dlg = create_clearmap_widget('nested_progress_dialog.ui', patch_parent_class='QDialog', window_title='Progress')

    progress_icon_path = os.path.join(UI_FOLDER, 'creator', 'graphics_resources', 'searching_mouse.png')
    dlg.progressImageLabel.setPixmap(QPixmap(progress_icon_path))  # TODO: why doesn't work with qrc ??
    dlg.setWindowTitle('Clearmap progress')
    dlg.mainLabel.setText(f'{title}, please wait.')

    dlg.mainProgressBar.setRange(1, overall_maximum)
    dlg.subProgressLabel.setText(sub_process_name)
    if abort_callback is not None:
        dlg.buttonBox.button(QDialogButtonBox.Abort).clicked.connect(abort_callback)
    dlg.show()
    return dlg


def make_simple_progress_dialog(title='Processing', overall_maximum=100, sub_process_name='', abort_callback=None, parent=None):
    dlg = make_nested_progress_dialog(title=title, overall_maximum=overall_maximum, sub_process_name=sub_process_name,
                                      abort_callback=abort_callback, parent=parent)
    dlg.mainProgressBar.setVisible(False)

    return dlg


def make_splash(img_source=None, bar_max=100, res='hd'):
    if img_source is None:
        img_source = os.path.join(UI_FOLDER, 'creator', 'graphics_resources', 'splash.png')
    splash_pix = QPixmap(img_source)  # .scaled(848, 480)

    # Prepare the painter to overlay text and lines onto the image.
    painter = QPainter(splash_pix)
    painter.setRenderHint(QPainter.Antialiasing)

    # Set up the font for the overlay text.
    font = QFont("Arial", 60, QFont.Bold)  # FIXME: put settting in config
    painter.setFont(font)
    metrics = QFontMetrics(font)

    from packaging.version import Version
    from importlib_metadata import version

    clearmap_version = Version(version('ClearMap'))
    text = f"ClearMap {clearmap_version.major}.{clearmap_version.minor}"

    # Calculate horizontal centering for the text.
    text_width = metrics.horizontalAdvance(text)
    x_text = (splash_pix.width() - text_width) // 5

    # Define a top margin. The first line will overlap the top of the image.
    top_margin = 10
    pen = QPen(Qt.white, 2)
    painter.setPen(pen)

    # Draw the top horizontal line.
    # Calculate the y coordinate for the top line: it is aligned with 80% of the font’s ascent.
    # line_y_top = top_margin + metrics.ascent() * 0.8
    # painter.drawLine(0, int(line_y_top), splash_pix.width(), int(line_y_top))

    # Draw the text just below the top line.
    # Compute a y coordinate for the text baseline.
    y_text = top_margin + metrics.ascent() + 10
    painter.drawText(x_text, y_text, text)

    painter.end()

    splash = QSplashScreen(splash_pix, Qt.WindowStaysOnTopHint)
    splash.setMask(splash_pix.mask())
    progress_bar = QProgressBar(splash)
    progress_bar.setTextVisible(True)
    progress_bar.setFont(QFont('Arial', DISPLAY_CONFIG[res]['splash_font_size'], QFont.Bold))
    progress_bar.setFormat("Loading... \t\t%p%")
    progress_bar.setMaximum(bar_max)
    margin = 50
    progress_bar.setGeometry(margin, splash_pix.height() - margin, splash_pix.width() - 2 * margin, 20)
    return splash, progress_bar


def update_pbar(app, pbar, value):
    pbar.setValue(value)
    app.processEvents()


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
