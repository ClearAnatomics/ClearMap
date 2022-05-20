from typing import Iterable

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QSpinBox, QDoubleSpinBox, QCheckBox, QLineEdit, QDialogButtonBox, QDockWidget, \
    QPlainTextEdit, QTextEdit


def none_str_to_literal(txt):
    if txt.lower() == 'none':
        return None
    else:
        return txt


def minus_1_to_disabled(instance, vals):
    for i, val in enumerate(vals):
        if val == -1:
            vals[i] = none_str_to_literal(instance.property('disabledValue'))
    return vals


def disabled_to_minus_1(instance, vals):
    if vals is None:
        return
    vals = list(vals)
    for i, val in enumerate(vals):
        if val is None or val in ('auto', 'None', instance.property('disabledValue')):
            vals[i] = -1
    return vals


def is_disableable(instance):
    return has_prop(instance, 'disabledValue') and not has_prop(instance, 'individuallyDisableable')


def has_prop(instance, prop_name):
    return prop_name in instance.dynamicPropertyNames()


def get_value(instance):
    values = []
    if not instance.controlsEnabled():
        disabled_value = instance.property('disabledValue')
        if disabled_value == 'None' or disabled_value is None:  # FIXME: check is
            return None
        elif disabled_value == 'auto':
            return 'auto'
        elif disabled_value == "[auto, auto]":
            return ['auto', 'auto']
        else:
            raise ValueError('Unsupported value for disabledValue: "{}"'.format(disabled_value))
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
        indices.append(int(spin_box.objectName().split('_')[-1]))
    sorted_spin_boxes = [box for _, box in sorted(zip(indices, spin_boxes))]
    return sorted_spin_boxes


def connect_value_changed(instance, callback):
    spin_boxes = get_sorted_spin_boxes(instance)
    for bx in spin_boxes:
        bx.valueChanged.connect(callback)


def connect_text_changed(instance, callback):
    get_text_edit(instance).textChanged.connect(callback)


def controls_enabled(instance):
    if instance.findChildren(QSpinBox) or instance.findChildren(QDoubleSpinBox):
        spin_boxes = get_sorted_spin_boxes(instance)
        return spin_boxes[0].isEnabled()
    elif instance.findChildren(QLineEdit):
        return instance.findChildren(QLineEdit)[0].isEnabled()
    elif instance.findChildren(QPlainTextEdit):
        return instance.findChildren(QPlainTextEdit)[0].isEnabled()
    else:
        raise NotImplementedError('Control type "{}" is not yet supported'.format(instance))


def get_check_box(instance):
    check_boxes = instance.findChildren(QCheckBox)
    if not check_boxes:
        return
    else:
        return check_boxes[0]


def enable_controls(instance):
    check_box = instance.getCheckBox()
    if check_box:
        check_box.setCheckState(Qt.Checked)


def disable_controls(instance):
    check_box = instance.getCheckBox()
    if check_box:
        check_box.setCheckState(Qt.Unchecked)


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
            raise ValueError('Expected QLineEdit or QPlainTextEdit, got "{}"'.format(type(text_edit)))


def get_text(instance):
    if instance.controlsEnabled():
        text_edit = get_text_edit(instance)
        if isinstance(text_edit, QLineEdit):
            return text_edit.text()
        elif isinstance(text_edit, QPlainTextEdit):
            return text_edit.toPlainText()
        else:
            raise ValueError('Expected QLineEdit or QPlainTextEdit, got "{}"'.format(type(text_edit)))
    else:
        disabled_value = instance.property('disabledValue')
        if disabled_value == 'None':
            return None
        elif disabled_value == 'auto':
            return disabled_value
        elif disabled_value == "[auto, auto]":
            return ['auto', 'auto']
        else:
            raise ValueError('Unsupported value for disabledValue: "{}"'.format(disabled_value))


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
