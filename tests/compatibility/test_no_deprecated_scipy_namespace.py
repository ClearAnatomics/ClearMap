"""
No maintained module uses the private submodules of ``scipy.ndimage`` (``scipy.ndimage.filters`` and the like).

SciPy deprecates them (a ``DeprecationWarning`` at each import) and removes them in SciPy 2.0, within the life of
ClearMap 3.2. Every function they hold is in the ``scipy.ndimage`` namespace. The scan is on the syntax trees, so
the modules need not be imported (compiled code, Qt...).
"""
import ast
import warnings
from pathlib import Path

PACKAGE_DIRECTORY = Path(__file__).resolve().parents[2] / 'ClearMap'
# Vendored code, code that is not imported, and the scripts of the user (ignored by git)
NOT_MAINTAINED_DIRECTORIES = ('External', '_Old', '_Todo', 'custom_scripts')
DEPRECATED_NDIMAGE_SUBMODULES = ('filters', 'fourier', 'interpolation', 'measurements', 'morphology')


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


def _is_a_deprecated_module_name(module_name):
    return any(module_name == f'scipy.ndimage.{submodule}' or module_name.startswith(f'scipy.ndimage.{submodule}.')
               for submodule in DEPRECATED_NDIMAGE_SUBMODULES)


def _uses_a_deprecated_submodule(node):
    if isinstance(node, ast.ImportFrom):
        if node.module == 'scipy.ndimage':
            return any(alias.name in DEPRECATED_NDIMAGE_SUBMODULES for alias in node.names)
        return node.module is not None and _is_a_deprecated_module_name(node.module)
    if isinstance(node, ast.Import):
        return any(_is_a_deprecated_module_name(alias.name) for alias in node.names)
    if isinstance(node, ast.Attribute) and node.attr in DEPRECATED_NDIMAGE_SUBMODULES:
        parent = node.value
        return (isinstance(parent, ast.Attribute) and parent.attr == 'ndimage') or \
            (isinstance(parent, ast.Name) and parent.id == 'ndimage')
    return False


def test_no_maintained_module_uses_a_deprecated_scipy_ndimage_submodule():
    uses = [f'{path.relative_to(PACKAGE_DIRECTORY.parent)}:{node.lineno}'
            for path, tree in _parsed_python_sources()
            for node in ast.walk(tree) if _uses_a_deprecated_submodule(node)]

    assert not uses, f'Use the scipy.ndimage namespace (e.g. scipy.ndimage.convolve) at {uses}'
