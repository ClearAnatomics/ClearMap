import os
import sys

from PyQt5 import QtCore
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QFileDialog, QMessageBox, QProgressDialog, QLabel, QDialogButtonBox

from ClearMap.config.config_loader import clean_path
from ClearMap.gui.gui_utils import runs_from_pycharm
from ClearMap.gui.pyuic_utils import loadUiType


def get_directory_dlg(start_folder, title="Choose the source directory"):
    diag = QFileDialog()  # REFACTOR: move to gui_utils
    if sys.platform == 'win32' or runs_from_pycharm():  # avoids bug with windows COM object init failed
        opt = QFileDialog.Options(QFileDialog.DontUseNativeDialog)
    else:
        opt = QFileDialog.Options()
    start_folder = clean_path(start_folder)
    src_folder = diag.getExistingDirectory(parent=diag, caption=title,
                                           directory=start_folder, options=opt)
    diag.close()
    return src_folder


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
    dlg.lbl.setPixmap(QPixmap('ClearMap/gui/graphics_resources/searching_mouse.png'))
    dlg.lbl.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
    dlg.setLabel(dlg.lbl)
    if canceled_callback is not None:
        dlg.canceled().connect(canceled_callback)
    dlg.setMinimumDuration(0)
    dlg.forceShow()  # To force update
    return dlg


# REFACTOR: make class
def make_nested_progress_dialog(title='Processing', overall_maximum=100, sub_maximum=100, sub_process_name='',
                                abort_callback=None, parent=None):
    dlg_class, _ = loadUiType('ClearMap/gui/nested_progress_dialog.ui', patch_parent_class='QDialog')
    dlg = dlg_class()
    dlg.setWindowTitle('Progress')
    dlg.setupUi()
    dlg.mainLabel.setText(title)
    dlg.progressImageLabel.setPixmap(QPixmap('ClearMap/gui/graphics_resources/searching_mouse.png'))  # TODO: why doesn't work w/ qrc ??
    dlg.mainProgressBar.setRange(1, overall_maximum)
    dlg.mainProgressBar.setValue(1)  # Because we use integer steps
    dlg.subProgressLabel.setText(sub_process_name)
    dlg.subProgressBar.setMaximum(sub_maximum)
    dlg.subProgressBar.setValue(0)
    if abort_callback is not None:
        dlg.buttonBox.button(QDialogButtonBox.Abort).clicked.connect(abort_callback)
    dlg.show()
    return dlg
