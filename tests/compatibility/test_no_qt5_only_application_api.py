"""
No maintained module calls QApplication.desktop(), which Qt 6 removed: the GUI centred its window with it, so it would
not start with PyQt6. QWidget.screen() (Qt >= 5.14) gives the screen of a window with both. A scan of the syntax trees:
gui/app.py starts the application when imported, so it cannot be imported by a test.
"""
import ast

from tests.source_scan import maintained_syntax_trees

REMOVED_METHOD_NAME = 'desktop'


def _calls_desktop(node):
    return isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == REMOVED_METHOD_NAME


def test_no_maintained_module_calls_the_desktop_of_the_application():
    calls = [f'{path}:{node.lineno}' for path, tree in maintained_syntax_trees()
             for node in ast.walk(tree) if _calls_desktop(node)]

    assert not calls, f'Use the screen of the window (widget.screen()) at {calls}'
