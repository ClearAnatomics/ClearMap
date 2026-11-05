"""
No maintained module uses a NumPy name that NumPy 2 removed or moved.

NumPy 2 is where ClearMap 3.2 runs. A name that is gone is an ``AttributeError`` or an ``ImportError`` at the line
that uses it, often deep in a pipeline or in the GUI. The scan is on the syntax trees, so that a docstring or a
comment that names the old API does not count, and it does not need the code to be imported (Qt, graph_tool...).
"""
import ast
import warnings
from functools import partial
from pathlib import Path

import pytest

PACKAGE_DIRECTORY = Path(__file__).resolve().parents[2] / 'ClearMap'
# Vendored code, code that is not imported, and the scripts of the user (ignored by git)
NOT_MAINTAINED_DIRECTORIES = ('External', '_Old', '_Todo', 'custom_scripts')
NUMPY_MODULE_NAMES = ('np', 'numpy')
# Name of the module that moved: where it is now
MOVED_NUMPY_MODULES = {
    'numpy.core': 'numpy._core is private, import the public name from numpy (e.g. from numpy import memmap)',
}
# Attribute of the numpy module that is gone: what replaces it
REMOVED_NUMPY_ATTRIBUTES = {
    'float_': 'np.float64',
    'in1d': 'np.isin',
}


def _parsed_python_sources():
    for path in sorted(PACKAGE_DIRECTORY.rglob('*.py')):
        if set(path.relative_to(PACKAGE_DIRECTORY).parts) & set(NOT_MAINTAINED_DIRECTORIES):
            continue
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')  # Old files with invalid escape sequences
            try:
                yield path, ast.parse(path.read_text(encoding='utf-8'))
            except SyntaxError:  # Legacy Python 2 files: they cannot be imported
                continue


def _imports_of_a_moved_module(node, module_name):
    if isinstance(node, ast.ImportFrom):
        return node.module is not None and (node.module == module_name or node.module.startswith(module_name + '.'))
    if isinstance(node, ast.Import):
        return any(alias.name == module_name or alias.name.startswith(module_name + '.') for alias in node.names)
    return False


def _uses_a_removed_attribute(node, attribute_name):
    return (isinstance(node, ast.Attribute) and node.attr == attribute_name
            and isinstance(node.value, ast.Name) and node.value.id in NUMPY_MODULE_NAMES)


def _offending_lines(is_offending):
    return [f'{path.relative_to(PACKAGE_DIRECTORY.parent)}:{node.lineno}'
            for path, tree in _parsed_python_sources() for node in ast.walk(tree) if is_offending(node)]


@pytest.mark.parametrize('module_name', sorted(MOVED_NUMPY_MODULES))
def test_no_maintained_module_imports_a_numpy_module_that_moved(module_name):
    offenders = _offending_lines(partial(_imports_of_a_moved_module, module_name=module_name))

    assert not offenders, f'{module_name}: {MOVED_NUMPY_MODULES[module_name]}\n' + '\n'.join(offenders)


@pytest.mark.parametrize('attribute_name', sorted(REMOVED_NUMPY_ATTRIBUTES))
def test_no_maintained_module_uses_a_numpy_attribute_that_was_removed(attribute_name):
    offenders = _offending_lines(partial(_uses_a_removed_attribute, attribute_name=attribute_name))

    replacement = REMOVED_NUMPY_ATTRIBUTES[attribute_name]
    assert not offenders, f'np.{attribute_name}: use {replacement}\n' + '\n'.join(offenders)
