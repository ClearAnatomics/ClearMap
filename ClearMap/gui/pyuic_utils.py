# -*- coding: utf-8 -*-
"""
pyuic_utils
===========

Essentially a reimplementation of loadUiType from PyQt5 to allow monkey patching the classes
"""

from io import StringIO

from PyQt5 import QtWidgets
from PyQt5.uic.Compiler import compiler


def loadUiType(uifile, from_imports=False, resource_suffix='_rc', import_from='.', patch_parent_class='', replace_pairs=None):
    """
    loadUiType(uifile, from_imports=False, resource_suffix='_rc', import_from='.') -> (form class, base class)

    Load a Qt Designer .ui file and return the generated form class and the Qt
    base class.

    uifile is a file name or file-like object containing the .ui file.
    from_imports is optionally set to generate relative import statements.  At
    the moment this only applies to the import of resource modules.
    resource_suffix is the suffix appended to the basename of any resource file
    specified in the .ui file to create the name of the Python module generated
    from the resource file by pyrcc4.  The default is '_rc', i.e. if the .ui
    file specified a resource file called foo.qrc then the corresponding Python
    module is foo_rc.
    import_from is optionally set to the package used for relative import
    statements.  The default is ``'.'``.
    """
    code_string = StringIO()
    winfo = compiler.UICompiler().compileUi(uifile, code_string, from_imports, resource_suffix, import_from)

    ui_globals = {}
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