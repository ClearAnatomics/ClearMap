"""
ClearMap never runs ``pyrcc4``, ``pyrcc5`` or ``pyrcc6``: PyQt6 has no resource compiler, so a boot which runs it fails
there with a FileNotFoundError. The Qt resources are compiled by the ``rcc`` of the loaded Qt (gui/qt_resources).
"""
import ast

from tests.source_scan import maintained_syntax_trees

PYRCC_COMMANDS = ('pyrcc4', 'pyrcc5', 'pyrcc6')


def _is_a_pyrcc_command(node):
    return isinstance(node, ast.Constant) and node.value in PYRCC_COMMANDS


def test_no_maintained_module_runs_pyrcc():
    runs = [f'{path}:{node.lineno}' for path, tree in maintained_syntax_trees(('ClearMap',))
            for node in ast.walk(tree) if _is_a_pyrcc_command(node)]

    assert not runs, f'Compile the resources with ClearMap.gui.qt_resources, not pyrcc, at {runs}'
