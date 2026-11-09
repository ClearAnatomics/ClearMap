"""
The dark style sheet is the one of the Qt binding that qtpy loaded.

The GUI and the config upgrade dialogs asked qdarkstyle for the style sheet of PyQt5 by name (qt_api='pyqt5',
load_stylesheet_pyqt5()), whatever binding ran: with PyQt6, the style sheet of the wrong binding.
"""
import ast

import pytest

from ClearMap.gui import style
from tests.source_scan import maintained_syntax_trees

STYLE_SHEET = 'QWidget { background: dark }'


class FakeQdarkstyle:
    requested_bindings = []

    @classmethod
    def load_stylesheet(cls, qt_api):
        cls.requested_bindings.append(qt_api)
        return STYLE_SHEET


@pytest.mark.parametrize('binding', ['pyqt6', 'pyqt5'])
def test_the_dark_style_sheet_is_asked_for_the_binding_in_use(monkeypatch, binding):
    monkeypatch.setattr(style, 'qdarkstyle', FakeQdarkstyle)
    monkeypatch.setattr(FakeQdarkstyle, 'requested_bindings', [])
    monkeypatch.setattr(style.qtpy, 'API', binding)

    assert style.dark_style_sheet() == STYLE_SHEET
    assert FakeQdarkstyle.requested_bindings == [binding]


def _asks_for_a_named_binding(node):
    # load_stylesheet_pyqt5(...) or load_stylesheet(qt_api='pyqt5')
    if isinstance(node, ast.Attribute) and node.attr.startswith('load_stylesheet_'):
        return True
    return isinstance(node, ast.Call) and any(keyword.arg == 'qt_api' and isinstance(keyword.value, ast.Constant)
                                              for keyword in node.keywords)


def test_no_module_asks_qdarkstyle_for_a_binding_by_name():
    uses = [f'{path}:{node.lineno}' for path, tree in maintained_syntax_trees()
            for node in ast.walk(tree) if _asks_for_a_named_binding(node)]

    assert not uses, f'Use ClearMap.gui.style.dark_style_sheet() at {uses}'
