import sys

from PyQt5 import QtCore
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QFileDialog, QMessageBox, QProgressDialog, QLabel

from ClearMap.gui.gui_utils import runs_from_pycharm


def get_directory_dlg(start_folder, title="Choose the source directory"):
    diag = QFileDialog()  # REFACTOR: move to gui_utils
    if sys.platform == 'win32' or runs_from_pycharm():  # avoids bug with windows COM object init failed
        opt = QFileDialog.Options(QFileDialog.DontUseNativeDialog)
    else:
        opt = QFileDialog.Options()
    src_folder = diag.getExistingDirectory(parent=diag, caption=title,
                                           directory=start_folder, options=opt)
    diag.close()
    return src_folder


def warning_popup(base_msg, msg):
    dlg = QMessageBox()
    dlg.setIcon(QMessageBox.Warning)
    dlg.setWindowTitle('Warning')
    dlg.setText('<b>{}</b>'.format(base_msg))
    dlg.setInformativeText(msg)
    dlg.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel)
    dlg.setDefaultButton(QMessageBox.Ok)
    return dlg.exec()


def make_progress_dialog(msg, maximum, canceled_callback, parent):
    dlg = QProgressDialog(msg, 'Abort', 0, maximum, parent=parent)  # TODO: see if can have a notnativestyle on unity
    dlg.setMinimumDuration(0)
    dlg.setWindowTitle(msg)
    dlg.lbl = QLabel(msg, parent=dlg)
    # dlg.lbl.setText(msg)  # TODO: check why this doesn't work with pixmap
    dlg.lbl.setPixmap(QPixmap('ClearMap/gui/icons/searching_mouse.png'))
    dlg.lbl.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
    dlg.setLabel(dlg.lbl)
    if canceled_callback is not None:
        dlg.canceled().connect(canceled_callback)
    dlg.setValue(0)  # To force update
    return dlg