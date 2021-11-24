from io import StringIO

from PyQt5 import QtWidgets
from PyQt5.uic.Compiler import compiler


def loadUiType(uifile, from_imports=False, resource_suffix='_rc', import_from='.', patch_parent_class=''):
    """loadUiType(uifile, from_imports=False, resource_suffix='_rc', import_from='.') -> (form class, base class)

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
        class_import = 'from PyQt5.QtWidgets import {}'.format(patch_parent_class)
        cls_list.insert(0, class_import)
        parent_name = None
        for i, l in enumerate(cls_list):
            if l.startswith('class'):
                cls_list[i] = l.replace('object', patch_parent_class)
            elif 'setupUi' in l:
                parent_name = l.split(',')[-1].strip(' ):')
                cls_list[i] = '    def setupUi(self):'
            elif parent_name is not None and l.strip().startswith(parent_name):
                cls_list[i] = ''
            elif parent_name is not None and '({})'.format(parent_name) in l:
                cls_list[i] = l.replace(parent_name, 'self')

        code_string = '\n'.join(cls_list)
    else:
        code_string = code_string.getvalue()
    exec(code_string, ui_globals)

    return ui_globals[winfo["uiclass"]], getattr(QtWidgets, winfo["baseclass"])