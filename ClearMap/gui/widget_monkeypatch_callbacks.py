# -*- coding: utf-8 -*-
"""
widget_monkeypatch_callbacks
============================

A set of functions that will be monkey patched as methods to the button boxes of the designer files
"""
import types

from qtpy.QtWidgets import QDialogButtonBox


__author__ = 'Charly Rousseau <charly.rousseau@icm-institute.org>'
__license__ = 'GPLv3 - GNU General Public License v3 (see LICENSE.txt)'
__copyright__ = 'Copyright © 2022 by Charly Rousseau'
__webpage__ = 'https://idisco.info'
__download__ = 'https://github.com/ClearAnatomics/ClearMap'


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
    recursive_patch_button_boxes(parent)
    fix_btn_boxes_text(parent)
