"""
The staged imports of the GUI (pretty_imports) follow the imports of every ClearMap module to import them one by one
behind the splash screen. A relative import in the ``__init__.py`` of a package means the package itself
(``from .x import y`` in ``ClearMap/Visualization/Color/__init__.py`` is ``ClearMap.Visualization.Color.x``), not its
parent: it used to be read as ``ClearMap.Visualization.x``, which does not exist, and the GUI failed to start.
"""
import pytest

from ClearMap.gui import pretty_imports

PACKAGE_WITH_RELATIVE_IMPORT = 'ClearMap.Visualization.Color'
MODULE_IMPORTED_BY_THE_PACKAGE = 'ClearMap.Visualization.Color.orientation_colormap'


def test_a_relative_import_of_a_package_is_a_module_of_that_package():
    dependencies = pretty_imports._ImportGraph().deps(PACKAGE_WITH_RELATIVE_IMPORT)

    assert MODULE_IMPORTED_BY_THE_PACKAGE in dependencies
    assert 'ClearMap.Visualization.orientation_colormap' not in dependencies


@pytest.mark.parametrize('level, module, module_holding_the_import, expected', [
    (1, 'x', 'ClearMap.a.b', 'ClearMap.a.x'),             # b.py is a module of the package ClearMap.a
    (2, 'x', 'ClearMap.a.b', 'ClearMap.x'),
])
def test_a_relative_import_of_a_module_is_relative_to_its_package(level, module, module_holding_the_import, expected):
    resolved = pretty_imports._ImportGraph._resolve_relative(module_holding_the_import, level, module, is_package=False)

    assert resolved == expected


@pytest.mark.parametrize('level, module, expected', [
    (1, 'x', 'ClearMap.a.b.x'),
    (1, None, 'ClearMap.a.b'),
    (2, 'x', 'ClearMap.a.x'),
])
def test_a_relative_import_of_a_package_is_relative_to_the_package_itself(level, module, expected):
    resolved = pretty_imports._ImportGraph._resolve_relative('ClearMap.a.b', level, module, is_package=True)

    assert resolved == expected

