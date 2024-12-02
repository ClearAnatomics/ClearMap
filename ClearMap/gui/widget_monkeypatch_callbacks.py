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


def is_disableable(instance):
    return has_prop(instance, 'disabledValue') and not has_prop(instance, 'individuallyDisableable')


def minus_1_to_disabled(instance, vals):
    for i, val in enumerate(vals):
        if val == -1:
            vals[i] = eval(instance.property('disabledValue'))
    return vals


def disabled_to_minus_1(instance, vals):
    if vals is None:
        return
    vals = list(vals)
    for i, val in enumerate(vals):
        if val is None or val in ('auto', 'None', instance.property('disabledValue')):
            vals[i] = -1
    return vals


def has_prop(instance, prop_name):
    return prop_name in instance.dynamicPropertyNames()


def get_value(instance):
    values = []
    if not instance.controlsEnabled():
        disabled_value = instance.property('disabledValue')
        if disabled_value is None:  # FIXME: check whether this case exists
            return
        else:
            return eval(disabled_value)
    sorted_spin_boxes = get_sorted_spin_boxes(instance)
    for spin_box in sorted_spin_boxes:
        values.append(spin_box.value())
    if is_disableable(instance) or has_prop(instance, 'individuallyDisableable'):
        values = minus_1_to_disabled(instance, values)
    if len(values) == 1:  # Singlet
        return values[0]
    else:
        return values


def set_value(instance, values):
    if values is None or values == 'auto':
        instance.disableControls()
    else:
        if not isinstance(values, Iterable):
            values = [values]
        if all([v is None for v in values]) or all([v == 'auto' for v in values]):
            instance.disableControls()
            return
        instance.enableControls()
        if is_disableable(instance) or has_prop(instance, 'individuallyDisableable'):
            values = disabled_to_minus_1(instance, values)
        sorted_spin_boxes = get_sorted_spin_boxes(instance)
        for val, spin_box in zip(values, sorted_spin_boxes):
            spin_box.setValue(val)


def get_sorted_spin_boxes(instance):
    indices = []
    spin_boxes = instance.findChildren(QSpinBox)
    if not spin_boxes:  # probably double
        spin_boxes = instance.findChildren(QDoubleSpinBox)
    for spin_box in spin_boxes:
        try:
            indices.append(int(spin_box.objectName().split('_')[-1]))
        except ValueError:
            raise ValueError(f'Could not extract index from "{spin_box.objectName()}" in "{instance.objectName()}"')
    sorted_spin_boxes = [box for _, box in sorted(zip(indices, spin_boxes))]
    return sorted_spin_boxes


def connect_value_changed(instance, callback):
    spin_boxes = get_sorted_spin_boxes(instance)
    for bx in spin_boxes:
        bx.valueChanged.connect(callback)
    chk_bx = get_check_box(instance)
    if chk_bx is not None:
        chk_bx.stateChanged.connect(callback)


def connect_text_changed(instance, callback):
    get_text_edit(instance).textChanged.connect(callback)
    get_check_box(instance).stateChanged.connect(callback)


def controls_enabled(instance):
    if instance.findChildren(QSpinBox) or instance.findChildren(QDoubleSpinBox):
        spin_boxes = get_sorted_spin_boxes(instance)
        return spin_boxes[0].isEnabled()
    elif instance.findChildren(QLineEdit):
        return instance.findChildren(QLineEdit)[0].isEnabled()
    elif instance.findChildren(QPlainTextEdit):
        return instance.findChildren(QPlainTextEdit)[0].isEnabled()
    else:
        raise NotImplementedError(f'Control type "{instance}" is not yet supported')


def get_check_box(instance):
    check_boxes = instance.findChildren(QCheckBox)
    if not check_boxes:
        return
    else:
        return check_boxes[0]


def enable_controls(instance):
    check_box = instance.getCheckBox()
    if check_box:
        check_box.setChecked(True)


def disable_controls(instance):
    check_box = instance.getCheckBox()
    if check_box:
        check_box.setChecked(False)


def get_text_edit(instance):
    children = instance.findChildren(QLineEdit)
    if not children:
        children = instance.findChildren(QPlainTextEdit)
    if not children:
        children = instance.findChildren(QTextEdit)
    return children[0]


def set_text(instance, txt):
    if txt is None or txt == 'auto' or txt == '':
        instance.disableControls()
    else:
        instance.enableControls()
        text_edit = get_text_edit(instance)
        if isinstance(text_edit, QLineEdit):
            text_edit.setText(txt)
        elif isinstance(text_edit, QPlainTextEdit):
            text_edit.setPlainText(txt)
        else:
            raise ValueError(f'Expected QLineEdit or QPlainTextEdit, got "{type(text_edit)}"')


def get_text(instance):
    if instance.controlsEnabled():
        text_edit = get_text_edit(instance)
        if isinstance(text_edit, QLineEdit):
            return text_edit.text()
        elif isinstance(text_edit, QPlainTextEdit):
            return text_edit.toPlainText()
        else:
            raise ValueError(f'Expected QLineEdit or QPlainTextEdit, got "{type(text_edit)}"')
    else:
        disabled_value = instance.property('disabledValue')
        if disabled_value == 'None':
            return None
        elif disabled_value == 'auto':
            return disabled_value
        elif disabled_value == "[auto, auto]":
            return ['auto', 'auto']
        else:
            raise ValueError(f'Unsupported value for disabledValue: "{disabled_value}"')


def connect_apply(instance, func):
    instance.button(QDialogButtonBox.Apply).clicked.connect(func)


def connect_close(instance, func):
    instance.button(QDialogButtonBox.Close).clicked.connect(func)


def connect_open(instance, func):
    instance.button(QDialogButtonBox.Open).clicked.connect(func)


def connect_save(instance, func):
    instance.button(QDialogButtonBox.Save).clicked.connect(func)


def connect_ok(instance, func):
    instance.button(QDialogButtonBox.Ok).clicked.connect(func)


def connect_cancel(instance, func):
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

        bx.controlsEnabled = types.MethodType(controls_enabled, bx)
        bx.getCheckBox = types.MethodType(get_check_box, bx)
        bx.enableControls = types.MethodType(enable_controls, bx)
        bx.disableControls = types.MethodType(disable_controls, bx)
        if is_list_box(bx_name):
            bx.getValue = types.MethodType(get_value, bx)
            bx.setValue = types.MethodType(set_value, bx)
            bx.valueChangedConnect = types.MethodType(connect_value_changed, bx)
        elif is_optional_box(bx_name):
            bx.setText = types.MethodType(set_text, bx)
            bx.text = types.MethodType(get_text, bx)
            bx.textChangedConnect = types.MethodType(connect_text_changed, bx)
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
        bx.connectApply = types.MethodType(connect_apply, bx)
        bx.connectClose = types.MethodType(connect_close, bx)
        bx.connectSave = types.MethodType(connect_save, bx)
        bx.connectOpen = types.MethodType(connect_open, bx)
        bx.connectOk = types.MethodType(connect_ok, bx)
        bx.connectCancel = types.MethodType(connect_cancel, bx)


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
