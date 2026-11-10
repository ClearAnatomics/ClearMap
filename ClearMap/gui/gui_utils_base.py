# -*- coding: utf-8 -*-
"""
gui_utils_base
==============

Various utility functions specific to the graphical interface

.. warning::

    This file is imported by the early_boot module, so avoid heavy imports here.
    Any non-standard-library imports should be done locally inside functions.
"""
import os
from math import sqrt, ceil, floor

import warnings
import xml.etree.ElementTree as ET
from io import StringIO
from pathlib import Path
from typing import Optional, Callable

from qtpy import QtWidgets
from qtpy.QtCore import Signal
from qtpy.QtWidgets import QLayout, QApplication, QWidget, \
    QToolBox  # REFACTOR: local imports or strings for annotations?

# WARNING: declared here for dependency visibility but imported locally where used
# from ClearMap.gui.widget_monkeypatch_callbacks import recursive_patch_widgets  # WARNING: local import

warnings.filterwarnings('ignore', category=RuntimeWarning, module='ClearMap.gui.gui_utils_base')  # For surface_project

# What QDialog.exec() returns (the values of QDialog.DialogCode, fixed by Qt). Compared as ints: with PyQt6 the
# QDialog.Accepted enum is not guaranteed to compare equal to the int of exec()
DIALOG_REJECTED = 0
DIALOG_ACCEPTED = 1

__author__ = 'Charly Rousseau <charly.rousseau@icm-institute.org>'
__license__ = 'GPLv3 - GNU General Public License v3 (see LICENSE.txt)'
__copyright__ = 'Copyright © 2022 by Charly Rousseau'
__webpage__ = 'https://idisco.info'
__download__ = 'https://github.com/ClearAnatomics/ClearMap'


UI_FOLDER = str(Path(__file__).resolve().parent)
RESOURCES_ELEMENT_TAG = 'resources'



def html_to_ansi(msg: str) -> str:  #  WARNING: Will not work correctly with colours
    codes_dict = {
        '<nobr>': '',
        '</nobr>': '',
        '<br>': '\n',
        '</em>': '\033[0m',
        '<em>': '\033[3m'
    }
    for k, v in codes_dict.items():
        msg = msg.replace(k, v)
    return msg


def html_to_plain_text(msg: str) -> str:
    codes_dict = {
        '<nobr>': '',
        '</nobr>': '',
        '<br>': '\n',
        '</em>': '',
        '<em>': ''
    }
    for k, v in codes_dict.items():
        msg = msg.replace(k, v)
    return msg


def compute_grid(nb: int) -> tuple[int, int]:
    if nb <= 0:
        return 0, 0
    sqr = max(1, floor(sqrt(nb)))
    col = ceil(nb / sqr)
    row = ceil(nb / col)
    return row, col


def format_long_nb(nb: int) -> str:
    """
    Formats a long number by inserting apostrophes every three digits for better readability.

    Parameters
    ----------
    nb : int
        The number to be formatted.

    Returns
    -------
    str
        The formatted number as a string with apostrophes inserted every three digits.

    Examples
    --------
    >>> format_long_nb(123456789)
    "123'456'789"
    """
    out = ''
    for idx, c in enumerate(str(nb)[::-1]):
        tick = "'" if (idx and not idx % 3) else ""
        out += tick+c
    out = out[::-1]
    return out


class TmpDebug:  # FIXME: move as part of workspace
    def __init__(self, workspace):
        self.workspace = workspace

    def __enter__(self):
        self.workspace.debug = True
        return self.workspace

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.workspace.debug = False


def ui_without_resource_includes(ui_file):
    """
    A Qt Designer file without its ``<resources>`` element, as a file-like object.

    The ``<include location="icons.qrc"/>`` of a designer file makes the PyQt loaders import a generated Python module
    (``icons_rc``), which ClearMap does not have: the resources are compiled and registered with Qt by
    :mod:`ClearMap.gui.qt_resources`, so the ``:/icons/...`` paths of the file work without it. The file on disk is
    unchanged, Designer still shows its icons.

    Arguments
    ---------
    ui_file : str or pathlib.Path or file-like
        The designer file.

    Returns
    -------
    io.StringIO
    """
    tree = ET.parse(ui_file)
    root = tree.getroot()
    for resources in root.findall(RESOURCES_ELEMENT_TAG):
        root.remove(resources)
    ui_text = StringIO()
    tree.write(ui_text, encoding='unicode')
    ui_text.seek(0)
    return ui_text


def load_ui_into(widget: QWidget, ui_name: str) -> None:
    """
    Build the content of a Qt Designer file of ``ClearMap/gui/creator`` into an existing widget.

    The properties of the root of the file (title, geometry, sizes...) are set on ``widget``, and its children become
    attributes of ``widget``, named after their objectName. ``widget`` must be an instance of the root class of the file
    (or of a subclass). The ``<resources>`` of the file are ignored (:func:`ui_without_resource_includes`).

    Parameters
    ----------
    widget : QWidget
        The widget to build into.
    ui_name : str
        The name of the file, with or without ``.ui``.
    """
    if not ui_name.endswith('.ui'):
        ui_name += '.ui'
    from qtpy import uic  # Local: see the warning of the module docstring
    uic.loadUi(ui_without_resource_includes(os.path.join(UI_FOLDER, 'creator', ui_name)), widget)


def create_clearmap_widget(ui_name: str, widget_class_name: str, window_title: Optional[str] = None) -> QWidget:
    """
    A widget of the Qt class ``widget_class_name`` (e.g. 'QDialog') built from a Qt Designer file of
    ``ClearMap/gui/creator``, with the methods of its compound boxes (``widget_monkeypatch_callbacks``).
    """
    widget = getattr(QtWidgets, widget_class_name)()
    load_ui_into(widget, ui_name)
    if window_title is not None:
        widget.setWindowTitle(window_title)  # After the load, which sets the title of the file
    from ClearMap.gui.widget_monkeypatch_callbacks import recursive_patch_widgets
    recursive_patch_widgets(widget)
    return widget


def clear_layout(layout: QLayout):
    """
    Clears all widgets in the layout

    Parameters
    ----------
    layout

    Returns
    -------

    """
    for i in range(layout.count(), -1, -1):  # TODO: check if not layout.count() - 1
        item = layout.takeAt(i)
        if item is not None:
            widg = item.widget()
            if widg is not None:
                widg.setParent(None)
                widg.deleteLater()


def find_parent_layout(widget: QWidget) -> QLayout | None:
    warnings.warn('find_parent_layout is unreliable and thus deprecated,'
                  ' use widget.parentWidget().layout() instead', DeprecationWarning, stacklevel=2)
    parent = widget.parent()
    while parent is not None:
        if isinstance(parent, QLayout):
            return parent
        parent = parent.parent()
    return None


# REFACTOR: support both QToolBox and QTabWidget

def delete_widget(widget: Optional[QWidget] = None, layout: Optional[QLayout] = None,
                  tool_box: Optional[QToolBox] = None, toolbox_page_index: Optional[int] = None) -> QWidget:
    if layout is not None and tool_box is not None:
        raise ValueError('Either layout or tool_box should be provided, not both')
    if tool_box is not None:
        index = toolbox_page_index if toolbox_page_index is not None else tool_box.indexOf(widget)
        if index != -1:
            tool_box.removeItem(index)
    else:
        if widget is None:
            raise ValueError('widget must be provided if layout is used')
        if layout is None:
            layout = find_parent_layout(widget)   # WARNING: layout may not be a parent of the widget
        if layout is not None:
            layout.removeWidget(widget)
    if widget is not None:
        widget.setParent(None)
        widget.deleteLater()

  # REFACTOR: rename to get_widget_and_count
def get_widget(layout, key='', widget_type=None, index=0) -> tuple[QWidget | None, int]:
    """
    Retrieve the widget (label or control) based on the key and index.

    Parameters:
    layout (QLayout): The layout containing the widgets.
    key (str): The key to search for in the widget's objectName.
    index (int): The index to specify whether to get the first (0) or second (1) occurrence.

    Returns:
    tuple: A tuple containing the found widget (or None if not found) and the total count of matching widgets.
    """
    if not key and not widget_type:
        raise ValueError('Either key or widget_type must be specified')
    count = 0
    for i in range(layout.count()):
        widget = layout.itemAt(i).widget()
        if widget:
            if (key and key in widget.objectName() or
                    widget_type and isinstance(widget, widget_type)):
                if count == index:
                    return widget, count
                count += 1
    return None, count


def replace_widget(old_widget: QWidget, new_widget: QWidget, layout: Optional[QLayout] = None) -> QWidget:
    """
    Replace a widget in a layout. If no layout is provided, the parent layout of the old widget is used
    The old widget is deleted.

    Parameters
    ----------
    old_widget: QWidget
        The widget to replace
    new_widget: QWidget
        The new widget to insert
    layout: QLayout | None
        The layout in which to replace the widget. If None, the parent layout of the old widget is used

    Returns
    -------
    QWidget
        The new widget
    """
    if layout is None:
        layout = find_parent_layout(old_widget)

    # Transfer objectName so ParamLink.findChild can locate the replacement
    obj_name = old_widget.objectName()
    if obj_name and not new_widget.objectName():
        new_widget.setObjectName(obj_name)

    layout.replaceWidget(old_widget, new_widget)

    # Poison: make _is_alive return False immediately (don't wait for event loop)
    old_widget.setObjectName('')
    old_widget.setProperty('_cm_replaced', True)
    old_widget.setParent(None)
    old_widget.deleteLater()

    return new_widget


def disconnect_widget_signal(widget_signal, max_iter: int = 10,
                             slot: Optional[Callable] = None) -> None:
    # Try the precise disconnect first
    if slot is not None:
        try:
            widget_signal.disconnect(slot)
            return
        except (TypeError, RuntimeError):
            pass

    for _ in range(max_iter):
        try:
            widget_signal.disconnect()
        except TypeError:  # i.e. not connected, skip
            return
    raise ValueError(f'Could not disconnect signal after {max_iter} iterations')


def unique_connect(signal: Signal, slot: Callable, *,
                   max_disconnect_iter: int = 10, disconnect_all: bool = False) -> None:
    """
    Connect a signal to a slot, ensuring no duplicate connections.

    Parameters
    ----------
    signal : Signal
        The signal to connect.
    slot : callable
        The slot to connect.
    max_disconnect_iter : int
        Maximum number of attempts to disconnect (safety valve against
        multiply-connected signals).
    disconnect_all : bool
        If False (default), only disconnect *slot* if already connected
        (idempotent re-connect of the same slot).
        If True, disconnect ALL existing slots first, then connect *slot*
        (ensures this is the ONLY slot on the signal — useful when the
        slot target changes between calls, e.g. progress dialog widgets).
    """
    disconnect_widget_signal(signal, max_iter=max_disconnect_iter,
                             slot=None if disconnect_all else slot)
    signal.connect(slot)


def add_missing_combobox_items(combobox: "QComboBox", items: list[str]):
    """Add the items to the combobox if they are not already present"""
    combobox.blockSignals(True)
    try:
        for item in items:
            if combobox.findText(item) == -1:
                combobox.addItem(item)
    finally:
        combobox.blockSignals(False)


def populate_combobox(box: "QComboBox", items: list[str]):
    """Replace all items in the combobox with the given items"""
    box.blockSignals(True)
    try:
        box.clear()
        box.addItems(items)
    finally:
        box.blockSignals(False)


def ensure_qapp(*, style_sheet: str = '') -> QApplication:
    """
    The running QApplication, else a new one with ``style_sheet``.

    A running application keeps its style sheet: the GUI has the one of its theme, which a dialog opened from it (e.g.
    the upgrade of an experiment) must not replace.
    """
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
        if style_sheet:
            app.setStyleSheet(style_sheet)
    return app


def run_with_responsive_ui(app: QApplication, progress_watcher, func: Callable, *args, **kwargs):
    """
    Run func in a worker thread while keeping the Qt UI responsive.

    Blocks until func completes, periodically processing Qt events and polling progress logs.

    Leaves the progress dialog open for the caller to manage.
    """
    from multiprocessing.pool import ThreadPool

    progress_watcher.start_polling()
    try:
        with ThreadPool(processes=1) as pool:
            result = pool.apply_async(func, args, kwargs)

            while not result.ready():
                result.wait(0.25)
                app.processEvents()

            return result.get()
    finally:
        progress_watcher.stop_polling()
