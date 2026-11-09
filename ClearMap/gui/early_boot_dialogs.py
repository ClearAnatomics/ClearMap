"""
early_boot_dialogs
==================

The dialogs of :mod:`ClearMap.config.early_boot`, which runs before the temporary folder is set.

.. warning::

    Imports only qtpy and :mod:`ClearMap.gui.gui_utils_base`: nothing imported here may read ``tempfile.tempdir``
    before early_boot sets it (see its module docstring).
"""
import pathlib

from qtpy.QtWidgets import QFileDialog, QMessageBox, QInputDialog

from ClearMap.gui.gui_utils_base import ensure_qapp


def prompt_tmp_folder(*, min_free_gb, is_valid):
    """
    Ask the user for the temporary folder: a folder dialog, then a text entry if none valid was chosen.

    Arguments
    ---------
    min_free_gb : int
        The free space recommended, shown to the user.
    is_valid : callable
        Tells whether a :class:`pathlib.Path` can be the temporary folder.

    Returns
    -------
    str or None
        The folder, or None if the user gave none valid.
    """
    app = ensure_qapp()

    # Plan A, proper folder using QT directory dialog
    tmp_folder = QFileDialog.getExistingDirectory(None, f'Select temporary folder (>={min_free_gb} GB free recommended)',
                                                  str(pathlib.Path.home()),
                                                  QFileDialog.ShowDirsOnly | QFileDialog.DontResolveSymlinks)
    if tmp_folder:
        tmp_folder = pathlib.Path(tmp_folder).expanduser()
        if is_valid(tmp_folder):
            return str(tmp_folder)
        else:
            QMessageBox.warning(None, 'Folder not suitable',
                                f'"{tmp_folder}" is not writeable or does not have enough free space.\n'
                                'Please choose another.')
    # Plan B, pure string for e.g. non existing folders (or cancel)
    text, ok = QInputDialog.getText(None, 'Enter temporary folder',
                                    f'No valid folder selected.\n'
                                    f'Enter an absolute path with ≥ {min_free_gb} GB free:')
    if ok and text.strip():
        tmp_folder = pathlib.Path(text.strip()).expanduser()
        if is_valid(tmp_folder):
            return str(tmp_folder)
        else:
            QMessageBox.critical(None, 'Invalid folder',
                                 f'"{tmp_folder}" is not writeable or does not have enough free space.')
    return None
