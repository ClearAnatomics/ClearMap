"""
ClearMap imports Qt only through qtpy, so that it runs with the binding chosen by ``gui/qt_binding`` (PyQt6 or
PyQt5). A module importing a binding directly (``from PyQt5.QtWidgets import ...``, ``from PyQt5 import sip``) or using
PyQt's own names (``pyqtSignal``, ``pyqtSlot``) would tie it to one.
"""
import ast

from tests.source_scan import maintained_syntax_trees

SCANNED_DIRECTORIES = ('ClearMap', 'tests')
ALLOWED_MODULES = (
    'ClearMap/gui/pyuic_utils.py',  # FIXME: replaced by qtpy.uic.loadUi in the Qt 6 series (B6)
    'ClearMap/gui/creator/graphics_resources_rc.py',  # FIXME: pyrcc5 output, moved to qtpy in the Qt 6 series (B7)
)
QT_BINDINGS = ('PyQt5', 'PyQt6', 'PySide2', 'PySide6', 'sip', 'shiboken2', 'shiboken6')
PYQT_NAMES = ('pyqtSignal', 'pyqtSlot', 'pyqtProperty')


def _syntax_trees():
    return maintained_syntax_trees(SCANNED_DIRECTORIES, allowed_modules=ALLOWED_MODULES)


def _binding_imported_by(node):
    if isinstance(node, ast.Import):
        names = [alias.name for alias in node.names]
    elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module is not None:
        names = [node.module]
    else:
        return None
    return next((name for name in names if name.split('.')[0] in QT_BINDINGS), None)


def _uses_a_pyqt_name(node):
    return (isinstance(node, ast.Name) and node.id in PYQT_NAMES) or \
        (isinstance(node, ast.Attribute) and node.attr in PYQT_NAMES) or \
        (isinstance(node, ast.ImportFrom) and any(alias.name in PYQT_NAMES for alias in node.names))


def test_no_maintained_module_imports_a_qt_binding_directly():
    imports = [f'{path}:{node.lineno} ({_binding_imported_by(node)})'
               for path, tree in _syntax_trees() for node in ast.walk(tree) if _binding_imported_by(node)]

    assert not imports, f'Import Qt through qtpy (from qtpy.QtWidgets import ...) at {imports}'


def test_no_maintained_module_uses_the_pyqt_names_of_signals_and_slots():
    uses = [f'{path}:{node.lineno}' for path, tree in _syntax_trees()
            for node in ast.walk(tree) if _uses_a_pyqt_name(node)]

    assert not uses, f'Use qtpy.QtCore.Signal and Slot at {uses}'
