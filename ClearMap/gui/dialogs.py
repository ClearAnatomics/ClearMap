# -*- coding: utf-8 -*-
"""
dialogs
=======

All the independent popup dialogs used by the GUI
"""

import os

from PyQt5 import QtCore
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QFont
from PyQt5.QtWidgets import QFileDialog, QMessageBox, QProgressDialog, QLabel, QDialogButtonBox, QSplashScreen, \
    QProgressBar

from ClearMap.config.config_loader import clean_path, ConfigLoader
from ClearMap.gui.gui_utils import UI_FOLDER, create_clearmap_widget

__author__ = 'Charly Rousseau <charly.rousseau@icm-institute.org>'
__license__ = 'GPLv3 - GNU General Public License v3 (see LICENSE.txt)'
__copyright__ = 'Copyright © 2022 by Charly Rousseau'
__webpage__ = 'https://idisco.info'
__download__ = 'https://www.github.com/ChristophKirst/ClearMap2'

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


# REFACTOR: make class
def warning_popup(base_msg, msg):
    dlg = QMessageBox()
    dlg.setIcon(QMessageBox.Warning)
    dlg.setWindowTitle('Warning')
    dlg.setText('<b>{}</b>'.format(base_msg))
    dlg.setInformativeText(msg)
    dlg.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel)
    dlg.setDefaultButton(QMessageBox.Ok)
    return dlg.exec()


# REFACTOR: make class
def make_progress_dialog(msg, maximum, canceled_callback, parent):
    dlg = QProgressDialog(msg, 'Abort', 0, maximum, parent=parent)
    dlg.setWindowTitle(msg)
    dlg.lbl = QLabel(msg, parent=dlg)
    # dlg.lbl.setText(msg)  # TODO: check why this doesn't work with pixmap
    dlg.lbl.setPixmap(QPixmap(os.path.join(UI_FOLDER, 'creator', 'graphics_resources', 'searching_mouse.png')))
    dlg.lbl.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
    dlg.setLabel(dlg.lbl)
    if canceled_callback is not None:
        dlg.canceled.connect(canceled_callback)
    dlg.setMinimumDuration(0)
    dlg.forceShow()  # To force update
    return dlg


# REFACTOR: make class
def make_nested_progress_dialog(overall_maximum=100, sub_process_name='', abort_callback=None, parent=None):
    dlg = create_clearmap_widget('nested_progress_dialog.ui', patch_parent_class='QDialog', window_title='Progress')

    progress_icon_path = os.path.join(UI_FOLDER, 'creator', 'graphics_resources', 'searching_mouse.png')
    dlg.progressImageLabel.setPixmap(QPixmap(progress_icon_path))  # TODO: why doesn't work with qrc ??

    dlg.mainProgressBar.setRange(1, overall_maximum)
    dlg.subProgressLabel.setText(sub_process_name)
    if abort_callback is not None:
        dlg.buttonBox.button(QDialogButtonBox.Abort).clicked.connect(abort_callback)
    dlg.show()
    return dlg


def make_splash(img_source=os.path.join(UI_FOLDER, 'creator', 'graphics_resources', 'splash.png'), bar_max=100, res='hd'):
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
