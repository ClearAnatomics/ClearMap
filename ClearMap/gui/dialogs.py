# -*- coding: utf-8 -*-
"""
dialogs
=======

All the independent popup dialogs used by the GUI
"""
import os
import functools

from PyQt5.QtCore import Qt, QSize
from PyQt5.QtGui import QPixmap, QFont
from PyQt5.QtWidgets import QFileDialog, QMessageBox, QLabel, QDialogButtonBox, QSplashScreen, \
    QProgressBar, QDialog, QHBoxLayout, QPushButton, QVBoxLayout, QStyle

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
def warning_popup(base_msg, msg):
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
