"""
No maintained module of ClearMap has an invalid escape sequence in a string (``'\\d'`` instead of ``r'\\d'``).

It is a ``SyntaxWarning`` since Python 3.12 (a ``DeprecationWarning`` before) and will be a ``SyntaxError``: the module
could not be imported any more. It also hides a mistake: a backslash at the end of a line of a docstring joins it with
the next one.
"""
import warnings
from pathlib import Path

PACKAGE_DIRECTORY = Path(__file__).resolve().parents[2] / 'ClearMap'
# Vendored code, code that is not imported, and the scripts of the user (ignored by git)
NOT_MAINTAINED_DIRECTORIES = ('External', '_Old', '_Todo', 'custom_scripts')
INVALID_ESCAPE_MESSAGE = 'invalid escape sequence'


def _maintained_python_files():
    return [path for path in sorted(PACKAGE_DIRECTORY.rglob('*.py'))
            if not set(path.relative_to(PACKAGE_DIRECTORY).parts) & set(NOT_MAINTAINED_DIRECTORIES)]


def _lines_with_invalid_escape_sequences(path):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        try:
            compile(path.read_text(encoding='utf-8'), str(path), 'exec')
        except SyntaxError:  # Legacy Python 2 files: they cannot be imported
            return []
    return [warning.lineno for warning in caught if INVALID_ESCAPE_MESSAGE in str(warning.message)]


def test_no_maintained_module_has_an_invalid_escape_sequence():
    offenders = {str(path.relative_to(PACKAGE_DIRECTORY)): lines
                 for path in _maintained_python_files()
                 if (lines := _lines_with_invalid_escape_sequences(path))}

    assert not offenders, f'Write these strings as raw strings (r"..."), the line numbers are: {offenders}'
