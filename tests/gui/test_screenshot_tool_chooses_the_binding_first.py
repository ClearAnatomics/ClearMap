"""
The screenshot tool takes its Qt binding from ``ClearMap.gui``, whose package ``__init__`` chooses it (QT_API, else
PyQt6, else PyQt5). qtpy and pyqtgraph read the binding when they are first imported: with both PyQt5 and PyQt6
installed, one qtpy import before ``import ClearMap.gui`` gives the tool the binding qtpy likes, not the one ClearMap
chose, and pyqtgraph may pick another. Nothing fails, the screenshots are of the wrong binding.
"""
import ast
from pathlib import Path

import pytest

TOOL_PATH = Path(__file__).resolve().parents[1] / 'gui_screenshots.py'
FIRST_IMPORT_OF_THE_APPLICATION = 'open_application'
QT_MODULES = ('qtpy', 'pyqtgraph', 'PyQt5', 'PyQt6')


@pytest.fixture(scope='module')
def functions():
    tree = ast.parse(TOOL_PATH.read_text(encoding='utf-8'))
    return {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}


def _imported_module(node):
    if isinstance(node, ast.Import):
        return node.names[0].name
    if isinstance(node, ast.ImportFrom) and node.level == 0:
        return node.module
    return None


def _imports_qt(node):
    module = _imported_module(node)
    return module is not None and module.split('.')[0] in QT_MODULES


def _imports_the_gui_package(node):
    module = _imported_module(node)
    return module is not None and module.split('.')[:2] == ['ClearMap', 'gui']


def _lines_where(function, predicate):
    return sorted(node.lineno for node in ast.walk(function) if predicate(node))


def _calls_of(function, names):
    return sorted(node.lineno for node in ast.walk(function)
                  if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in names)


def test_the_tool_imports_no_qt_at_module_level():
    tree = ast.parse(TOOL_PATH.read_text(encoding='utf-8'))
    assert not [node.lineno for node in tree.body if _imports_qt(node)]


def test_the_helpers_which_import_qt_are_called_after_the_gui_package_is_imported(functions):
    qt_helpers = {name for name, function in functions.items() if _lines_where(function, _imports_qt)}
    assert qt_helpers, 'the tool does not import Qt any more: this test has nothing to guard'

    opening = functions[FIRST_IMPORT_OF_THE_APPLICATION]
    gui_import = _lines_where(opening, _imports_the_gui_package)[0]
    assert all(line > gui_import for line in _calls_of(opening, qt_helpers)), (
        f'{FIRST_IMPORT_OF_THE_APPLICATION} calls a helper importing Qt before it imports ClearMap.gui')

    main = functions['main']
    first_opening = _calls_of(main, {FIRST_IMPORT_OF_THE_APPLICATION})[0]
    assert all(line > first_opening for line in _calls_of(main, qt_helpers)), (
        'main calls a helper importing Qt before open_application, which imports ClearMap.gui')
    assert not [line for line in _lines_where(main, _imports_qt) if line < first_opening]
