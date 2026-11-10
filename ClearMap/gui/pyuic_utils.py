# -*- coding: utf-8 -*-
"""
pyuic_utils
===========

Essentially a reimplementation of loadUiType from PyQt5 to allow monkey patching the classes
"""

import xml.etree.ElementTree as ET
from io import StringIO

from PyQt5 import QtWidgets
from PyQt5.uic.Compiler import compiler

RESOURCES_ELEMENT_TAG = 'resources'


def ui_without_resource_includes(ui_file):
    """
    A Qt Designer file without its ``<resources>`` element, as a file-like object.

    The ``<include location="icons.qrc"/>`` of a designer file makes the loader import a generated Python module
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


def loadUiType(uifile, patch_parent_class='', replace_pairs=None, customWidgets=None):
    """
    loadUiType(uifile, patch_parent_class='', replace_pairs=None, customWidgets=None) -> (form class, base class)

    Load a Qt Designer .ui file and return the generated form class and the Qt base class.

    The ``<resources>`` of the file are ignored (see :func:`ui_without_resource_includes`): the
    ``:/icons/...`` paths work because :mod:`ClearMap.gui.qt_resources` registers the compiled resources with Qt,
    not because the generated code imports a module (as the PyQt5 function this one reimplements does).

    Parameters
    ----------
    uifile : str or pathlib.Path or file-like
        The .ui file.
    patch_parent_class : str
        Name of the Qt class (e.g. 'QDialog') that the form class derives from, instead of ``object``. The
        form then sets itself up (``setupUi(self)`` takes no parent argument).
    replace_pairs : list of tuple, optional
        (old, new) texts replaced in each line of the generated code (only with ``patch_parent_class``).
    customWidgets : dict, optional
        The classes of the custom widgets of the file, by name.
    """
    code_string = StringIO()
    # The imports of the resource modules, the only use of these three arguments, are not generated: no resources
    winfo = compiler.UICompiler().compileUi(ui_without_resource_includes(uifile), code_string, from_imports=False,
                                            resource_suffix='_rc', import_from='.')

    ui_globals = {}
    if customWidgets:
        ui_globals.update(customWidgets)
    if patch_parent_class:
        winfo['baseclass'] = patch_parent_class
        cls_list = code_string.getvalue().splitlines()
        class_import = f'from PyQt5.QtWidgets import {patch_parent_class}'
        cls_list.insert(0, class_import)
        parent_name = None
        for i, ln in enumerate(cls_list):
            if ln.startswith('class'):
                ln = ln.replace('object', patch_parent_class)
            elif 'setupUi' in ln:
                parent_name = ln.split(',')[-1].strip(' ):')
                if parent_name.lower() == 'dialog':
                    raise ValueError('Parent class name "Dialog" is not allowed.'
                                     'It would conflict with the Dialog class in PyQt5.'
                                     ' Please use a different name.')
                ln = '    def setupUi(self):'
            # elif parent_name is not None and ln.strip().startswith(parent_name):
            elif parent_name is not None:
                if f'({parent_name})' in ln:  # Parent class as arg
                    ln = ln.replace(parent_name, 'self')
                if f'{parent_name}.' in ln:  # Call to parent method
                    # if f'{parent_name}.setObjectName' not in ln:
                    ln = 'remove_me'  # TODO: find better keyword
                    # continue
            if replace_pairs is not None:
                for pair in replace_pairs:
                    old, new = pair
                    if old in ln:
                        ln = ln.replace(old, new)
            cls_list[i] = ln
        cls_list = [ln for ln in cls_list if ln != 'remove_me']
        code_string = '\n'.join(cls_list)
    else:
        code_string = code_string.getvalue()
    exec(code_string, ui_globals)

    return ui_globals[winfo["uiclass"]], getattr(QtWidgets, winfo["baseclass"])