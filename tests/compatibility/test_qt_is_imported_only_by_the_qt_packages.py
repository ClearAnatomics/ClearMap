"""
Only two packages of ClearMap import Qt: ClearMap.gui and ClearMap.Visualization.Qt. Their ``__init__`` chooses the Qt
binding (gui/qt_binding) and runs before any of their modules, so every Qt import of ClearMap comes after the choice,
without depending on the order of the imports. The rest of ClearMap reaches Qt through them, and may import Qt names
directly only for type checking (``if TYPE_CHECKING:``), which never runs.
"""
import ast

from tests.qt_imports import QT_MODULES
from tests.source_scan import maintained_syntax_trees

QT_PACKAGES = ('ClearMap/gui/', 'ClearMap/Visualization/Qt/')
QT_IMPORTING_MODULES = QT_MODULES + ('qdarkstyle',)  # qdarkstyle imports qtpy
TYPE_CHECKING_NAME = 'TYPE_CHECKING'


def _is_type_checking(test):
    return (isinstance(test, ast.Name) and test.id == TYPE_CHECKING_NAME) or \
        (isinstance(test, ast.Attribute) and test.attr == TYPE_CHECKING_NAME)


def _nodes_run_only_by_type_checkers(tree):
    return {id(node) for if_node in ast.walk(tree) if isinstance(if_node, ast.If) and _is_type_checking(if_node.test)
            for statement in if_node.body for node in ast.walk(statement)}


def _qt_module_imported_by(node):
    if isinstance(node, ast.Import):
        names = [alias.name for alias in node.names]
    elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module is not None:
        names = [node.module]
    else:
        return None
    return next((name for name in names if name.split('.')[0] in QT_IMPORTING_MODULES), None)


def test_outside_the_qt_packages_qt_is_imported_only_for_type_checking():
    imports = []
    for path, tree in maintained_syntax_trees(('ClearMap',)):
        if path.as_posix().startswith(QT_PACKAGES):
            continue
        type_checking_only = _nodes_run_only_by_type_checkers(tree)
        imports += [f'{path}:{node.lineno} ({_qt_module_imported_by(node)})' for node in ast.walk(tree)
                    if _qt_module_imported_by(node) and id(node) not in type_checking_only]

    assert not imports, f'Reach Qt through ClearMap.gui or ClearMap.Visualization.Qt at {imports}'
