"""
No maintained module of ClearMap imports the deprecated ``ClearMap.IO.IO`` module.

``ClearMap.IO.IO`` is the compatibility module of the pre-3.1 API: each of its names gives a ``DeprecationWarning``
that points to the current module. ClearMap's own calls to it showed these warnings to users for code they did not
write. The scan is on the syntax trees, so the modules need not be imported (compiled code, Qt...).
"""
import ast
import warnings
from pathlib import Path

PACKAGE_DIRECTORY = Path(__file__).resolve().parents[2] / 'ClearMap'
# Vendored code, code that is not imported, and the scripts of the user (ignored by git)
NOT_MAINTAINED_DIRECTORIES = ('External', '_Old', '_Todo', 'custom_scripts')
ALLOWED_MODULES = (
    'IO/IO.py',  # The compatibility module itself
    'Environment.py',  # The namespace of the user scripts: its `io` is the old API, kept for them
    'ImageProcessing/Topology/Topology3d.py',  # In the profiling block of its _test function
)
DEPRECATED_MODULE = 'ClearMap.IO.IO'


def _parsed_python_sources():
    for path in sorted(PACKAGE_DIRECTORY.rglob('*.py')):
        relative_path = path.relative_to(PACKAGE_DIRECTORY)
        if set(relative_path.parts) & set(NOT_MAINTAINED_DIRECTORIES) or relative_path.as_posix() in ALLOWED_MODULES:
            continue
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')  # Old files with invalid escape sequences
            try:
                yield path, ast.parse(path.read_text(encoding='utf-8'))
            except SyntaxError:  # Legacy Python 2 files: they cannot be imported
                continue


def _imports_the_deprecated_module(node):
    if isinstance(node, ast.Import):
        return any(alias.name == DEPRECATED_MODULE for alias in node.names)
    if isinstance(node, ast.ImportFrom):
        imported_names = {alias.name for alias in node.names}
        if node.level == 0:
            return node.module == DEPRECATED_MODULE or (node.module == 'ClearMap.IO' and 'IO' in imported_names)
        # Relative: from ..IO import IO, from ..IO.IO import x
        return (node.module == 'IO' and 'IO' in imported_names) or node.module == 'IO.IO'
    return False


def test_no_maintained_module_imports_the_deprecated_clearmap_io_module():
    uses = [f'{path.relative_to(PACKAGE_DIRECTORY.parent)}:{node.lineno}'
            for path, tree in _parsed_python_sources()
            for node in ast.walk(tree) if _imports_the_deprecated_module(node)]

    assert not uses, f'Use the module that ClearMap.IO.IO names in its DeprecationWarning at {uses}'
