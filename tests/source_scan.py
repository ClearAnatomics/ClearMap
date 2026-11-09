"""
The syntax trees of the maintained Python modules, for the tests that scan the source instead of importing it (the
modules need not be importable here: compiled code, Qt, graph-tool...).
"""
import ast
import warnings
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
# Vendored code, code that is not imported, and the scripts of the user (ignored by git)
NOT_MAINTAINED_DIRECTORIES = ('External', '_Old', '_Todo', 'custom_scripts')


def maintained_syntax_trees(directories=('ClearMap',), *, allowed_modules=()):
    """
    Yield the path (relative to the repository) and the syntax tree of each maintained module.

    Parameters
    ----------
    directories : tuple of str
        The directories of the repository to scan.
    allowed_modules : tuple of str
        Paths relative to the repository (``'ClearMap/gui/app.py'``) to leave out.
    """
    for directory in directories:
        for path in sorted((REPOSITORY_ROOT / directory).rglob('*.py')):
            relative_path = path.relative_to(REPOSITORY_ROOT)
            if set(relative_path.parts) & set(NOT_MAINTAINED_DIRECTORIES) or \
                    relative_path.as_posix() in allowed_modules:
                continue
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')  # Old files with invalid escape sequences
                try:
                    yield relative_path, ast.parse(path.read_text(encoding='utf-8'))
                except SyntaxError:  # Legacy Python 2 files: they cannot be imported
                    continue
