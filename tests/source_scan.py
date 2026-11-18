"""
The syntax trees of the maintained Python modules, for the tests that scan the source instead of importing it (the
modules need not be importable here: compiled code, Qt, graph-tool...).
"""
import ast
import subprocess
import warnings
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
# Vendored code, code that is not imported, and the scripts of the user (ignored by git). Outside a git repository only
# these names tell them apart: inside one, the files that git ignores (spikes, the *_rc.py of old versions) are left out
NOT_MAINTAINED_DIRECTORIES = ('External', '_Old', '_Todo', 'custom_scripts')


def _python_files(directory, root):
    """The Python files of ``directory``: the tracked ones and the new ones, not those git ignores (all, outside git)"""
    try:
        listed = subprocess.run(['git', '-C', str(root), 'ls-files', '--cached', '--others', '--exclude-standard', '-z',
                                 '--', directory], capture_output=True, text=True, check=True).stdout
    except (FileNotFoundError, subprocess.CalledProcessError):  # No git, or not a repository (a zip of a commit)
        return sorted((root / directory).rglob('*.py'))
    return sorted(path for path in (root / name for name in listed.split('\0') if name.endswith('.py'))
                  if path.is_file())  # A file removed but not yet staged is still listed


def maintained_syntax_trees(directories=('ClearMap',), *, allowed_modules=(), root=REPOSITORY_ROOT):
    """
    Yield the path (relative to the root) and the syntax tree of each maintained module.

    Parameters
    ----------
    directories : tuple of str
        The directories of the repository to scan.
    allowed_modules : tuple of str
        Paths relative to the repository (``'ClearMap/gui/app.py'``) to leave out.
    root : pathlib.Path
        The repository, by default the one of these tests.
    """
    for directory in directories:
        for path in _python_files(directory, root):
            relative_path = path.relative_to(root)
            if set(relative_path.parts) & set(NOT_MAINTAINED_DIRECTORIES) or \
                    relative_path.as_posix() in allowed_modules:
                continue
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')  # Old files with invalid escape sequences
                try:
                    yield relative_path, ast.parse(path.read_text(encoding='utf-8'))
                except SyntaxError:  # Legacy Python 2 files: they cannot be imported
                    continue
