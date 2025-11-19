# -*- coding: utf-8 -*-
"""
widget_monkeypatch_callbacks
============================

A set of functions that will be monkey patched as methods to the graphical widgets,
allowing notably compound widgets from QtCreator without defining plugins
"""
import types
from typing import Iterable

from PyQt5.QtWidgets import (QSpinBox, QDoubleSpinBox, QCheckBox, QLineEdit,
                             QDialogButtonBox, QDockWidget, QPlainTextEdit, QTextEdit, QFrame)


__author__ = 'Charly Rousseau <charly.rousseau@icm-institute.org>'
__license__ = 'GPLv3 - GNU General Public License v3 (see LICENSE.txt)'
__copyright__ = 'Copyright © 2022 by Charly Rousseau'
__webpage__ = 'https://idisco.info'
__download__ = 'https://github.com/ClearAnatomics/ClearMap'


def _get_list_box_value(instance):
    """
    Get the value(s) of the spin box(es) in the instance.
    Individual sentinel values (-1) are left as is for the caller to interpret
    (e.g. as meaning 'auto' or 'None' depending on the context)
    If the control is entirely disabled (box unchecked), None is returned

    Parameters
    ----------
    instance: QFrame
        The instance of the compound box to get the value(s) from

    Returns
    -------
    int, float, list or None
        The value(s) of the spin box(es) or None if the box is disabled
    """
    if not instance.controlsEnabled():
        return None
    values = [spin.value() for spin in _get_sorted_spin_boxes(instance)]

    return values[0] if len(values) == 1 else values  # return single value for singlet


def _set_list_box_value(instance, values):
    """
    Set the value(s) of the spin box(es) in the instance
    values are expected to have been set to None if the box is disabled or
    to the sentinel value -1 if the box is disableable but not all are None
    (i.e. at least one value is not None)

    Parameters
    ----------
    instance: QFrame
        The instance of the compound box to set the value(s) for
    values: Iterable, int, float or None
        The value(s) to set. If None, the box is disabled.
    """
    if values is None:
        instance.disableControls()
        return
    elif isinstance(values, (list, tuple)) and len(values) != len(_get_sorted_spin_boxes(instance)):
        raise ValueError(f'Expected {len(_get_sorted_spin_boxes(instance))} values, got {len(values)}')
    else:
        if not isinstance(values, Iterable) or isinstance(values, str):
            values = [values]

        instance.enableControls()
        spins = _get_sorted_spin_boxes(instance)

        for i, (val, spin_box) in enumerate(zip(values, spins)):
            spin_box.blockSignals(i != len(values) - 1)  # block signals except for last to avoid multiple calls
            if isinstance(spin_box, QSpinBox):
                spin_box.setValue(int(val))
            elif isinstance(spin_box, QDoubleSpinBox):
                spin_box.setValue(float(val))
            spin_box.blockSignals(False)


def _get_sorted_spin_boxes(instance):
    cached = getattr(instance, "_cm_sorted_spin_boxes", None)
    if cached is not None:
        return cached

    indices = []
    spin_boxes = instance.findChildren(QSpinBox) or instance.findChildren(QDoubleSpinBox)
    for spin_box in spin_boxes:
        try:  # Sort by integer following the last underscore in the objectName
            indices.append(int(spin_box.objectName().split('_')[-1]))
        except ValueError:
            raise ValueError(f'Could not extract index from "{spin_box.objectName()}" in "{instance.objectName()}"')
    sorted_spin_boxes = [box for _, box in sorted(zip(indices, spin_boxes))]
    instance._cm_sorted_spin_boxes = sorted_spin_boxes  # cache the result
    return sorted_spin_boxes


def _connect_value_changed(instance, callback):
    spin_boxes = _get_sorted_spin_boxes(instance)
    for bx in spin_boxes:
        # bx.valueChanged.connect(callback)
        bx.setKeyboardTracking(False)
        bx.editingFinished.connect(callback)
    chk_bx = _get_check_box(instance)
    if chk_bx is not None:
        chk_bx.toggled.connect(callback)


def _connect_text_changed(instance, callback):
    _get_text_edit(instance).textChanged.connect(callback)
    chk = _get_check_box(instance)
    if chk is not None:
        chk.toggled.connect(callback)


def _controls_enabled(instance):
    chk = _get_check_box(instance)
    if chk is not None:
        return chk.isChecked()
    # No checkbox, check if the contained control(s) are enabled
    if instance.findChildren(QSpinBox) or instance.findChildren(QDoubleSpinBox):
        spin_boxes = _get_sorted_spin_boxes(instance)
        return spin_boxes[0].isEnabled()
    elif instance.findChildren(QLineEdit):
        return instance.findChildren(QLineEdit)[0].isEnabled()
    elif instance.findChildren(QPlainTextEdit):
        return instance.findChildren(QPlainTextEdit)[0].isEnabled()
    else:
        raise NotImplementedError(f'Control type "{instance}" is not yet supported')


def _get_check_box(instance):
    check_boxes = instance.findChildren(QCheckBox)
    return check_boxes[0] if check_boxes else None


def _enable_controls(instance):
    check_box = instance.getCheckBox()
    if check_box:
        check_box.setChecked(True)


def _disable_controls(instance):
    check_box = instance.getCheckBox()
    if check_box:
        check_box.setChecked(False)


def _get_text_edit(instance):
    for cls in (QLineEdit, QPlainTextEdit, QTextEdit):
        children = instance.findChildren(cls)
        if children:
            return children[0]


def _set_text(instance, txt):
    if txt is None or txt == 'auto' or txt == '':
        instance.disableControls()
    else:
        instance.enableControls()
        text_edit = _get_text_edit(instance)
        if isinstance(text_edit, QLineEdit):
            text_edit.setText(txt)
        elif isinstance(text_edit, QPlainTextEdit):
            text_edit.setPlainText(txt)
        else:
            raise ValueError(f'Expected QLineEdit or QPlainTextEdit, got "{type(text_edit)}"')


def _get_text(instance):
    if instance.controlsEnabled():
        text_edit = _get_text_edit(instance)
        if isinstance(text_edit, QLineEdit):
            return text_edit.text()
        elif isinstance(text_edit, QPlainTextEdit):
            return text_edit.toPlainText()
        else:
            raise ValueError(f'Expected QLineEdit or QPlainTextEdit, got "{type(text_edit)}"')
    else:  # Disabled, return disabled value
        return None


def _connect_apply(instance, func):
    instance.button(QDialogButtonBox.Apply).clicked.connect(func)


def _connect_close(instance, func):
    instance.button(QDialogButtonBox.Close).clicked.connect(func)


def _connect_open(instance, func):
    instance.button(QDialogButtonBox.Open).clicked.connect(func)


def _connect_save(instance, func):
    instance.button(QDialogButtonBox.Save).clicked.connect(func)


def _connect_ok(instance, func):
    instance.button(QDialogButtonBox.Ok).clicked.connect(func)


def _connect_cancel(instance, func):
    instance.button(QDialogButtonBox.Cancel).clicked.connect(func)


def dock_resize_event(instance, event):
    super(QDockWidget, instance).__init__()
    instance.resized.emit()


def recursive_patch_compound_boxes(parent):
    """
    Since it is difficult to create real custom widgets in PyQt which can be used in QtCreator,
    we chose a different approach based on the dynamic nature of Python.
    We define new compound types (e.g. checkable text edit or triplets of values) based on
    dynamic properties and the objectNames in QtCreator and then patch the behaviour of these
    widgets in this method

    Parameters
    ----------
    parent: QWidget
        The main widget to start the recursive search
    """
    frames = parent.findChildren(QFrame)
    for bx in frames:
        bx_name = bx.objectName().lower()
        if not is_compound_box(bx_name):
            continue

        bx.controlsEnabled = types.MethodType(_controls_enabled, bx)
        bx.getCheckBox = types.MethodType(_get_check_box, bx)
        bx.enableControls = types.MethodType(_enable_controls, bx)
        bx.disableControls = types.MethodType(_disable_controls, bx)
        if is_list_box(bx_name):
            bx.getValue = types.MethodType(_get_list_box_value, bx)
            bx.setValue = types.MethodType(_set_list_box_value, bx)
            bx.valueChangedConnect = types.MethodType(_connect_value_changed, bx)
        elif is_optional_box(bx_name):
            bx.setText = types.MethodType(_set_text, bx)
            bx.text = types.MethodType(_get_text, bx)
            bx.textChangedConnect = types.MethodType(_connect_text_changed, bx)
        else:
            print(f'Skipping box "{bx_name}", type not recognised')


def is_list_box(bx_name):
    """
    Check whether the name of the box is a list box i.e. a custom widget
    ending in singlet double triplet that can be used to store a list of numbers
    """
    return bx_name.startswith('triplet') or bx_name.endswith('let')


def is_optional_box(box_name):
    return box_name.endswith('optionallineedit') or box_name.endswith('optionalplaintextedit')


def is_compound_box(bx_name):
    """
    Check whether the name of the box is a compound box i.e. a custom widget
    that is not natively supported by Qt but is based on elements inserted in a QFrame
    in the .ui file. The behaviour is defined by the objectName

    Parameters
    ----------
    bx_name: str
        The objectName of the QFrame

    Returns
    -------
    bool
    """
    return (bx_name.startswith('triplet') or
            bx_name.endswith('let') or
            bx_name.endswith('optionallineedit') or
            bx_name.endswith('optionalplaintextedit'))


def recursive_patch_button_boxes(parent):
    """
    To shorten the syntax, QDialogButtonBoxes are patched by this method
    so that e.g.
    bx.connectApply(f) replaces bx.button(QDialogButtonBox.Apply).clicked.connect(f)

    Parameters
    ----------
    parent: QWidget
        The main widget to start the recursive search
    """
    for bx in parent.findChildren(QDialogButtonBox):
        bx.connectApply = types.MethodType(_connect_apply, bx)
        bx.connectClose = types.MethodType(_connect_close, bx)
        bx.connectSave = types.MethodType(_connect_save, bx)
        bx.connectOpen = types.MethodType(_connect_open, bx)
        bx.connectOk = types.MethodType(_connect_ok, bx)
        bx.connectCancel = types.MethodType(_connect_cancel, bx)


def fix_btn_boxes_text(widget):
    """
    Rewrite the text on top of QDialogButtonBox(es) based on the
    dynamic properties 'applyText', 'okText' and 'openText' defined
    in the ui files in QtCreator

    Returns
    -------

    """
    for btn_box in widget.findChildren(QDialogButtonBox):
        if btn_box.property('applyText'):
            btn_box.button(QDialogButtonBox.Apply).setText(btn_box.property('applyText'))
        if btn_box.property('okText'):
            btn_box.button(QDialogButtonBox.Ok).setText(btn_box.property('okText'))
        if btn_box.property('openText'):
            btn_box.button(QDialogButtonBox.Open).setText(btn_box.property('openText'))


def recursive_patch_widgets(parent):
    """
    Recursively patch the widgets in the parent widget
    This is used to add custom methods to the widgets that are not natively supported by Qt
    yet do it mainly from the .ui file as it is based on the objectName

    Parameters
    ----------
    parent: QWidget
        The main widget to start the recursive search
    """
    recursive_patch_compound_boxes(parent)
    recursive_patch_button_boxes(parent)
    fix_btn_boxes_text(parent)
