"""
The version of ClearMap is written once, in ``ClearMap/_version.py``.

Everything else reads it from there: ``setup.py`` (to build the package) and the modules that need to know
the version (the config layer to pick the config files, the GUI to show it). None of them asks the metadata of
an installed package, which only exists once ClearMap is pip installed: from a checkout, or on a machine
where the package is not installed (a cluster, a CI job) it raises ``PackageNotFoundError`` or, worse,
a fallback value silently disagrees with ``setup.py``.
"""
import ast
from pathlib import Path

from packaging.version import Version

import ClearMap

REPOSITORY_ROOT = Path(ClearMap.__file__).resolve().parent.parent
PACKAGE_DIRECTORY = REPOSITORY_ROOT / 'ClearMap'
SETUP_SCRIPT = REPOSITORY_ROOT / 'setup.py'

PACKAGE_NAME = 'ClearMap'
METADATA_MODULES = ('importlib.metadata', 'importlib_metadata')
VENDORED_DIRECTORY_NAME = 'External'


def _is_python_file_of_clearmap(path: Path) -> bool:
    return VENDORED_DIRECTORY_NAME not in path.relative_to(PACKAGE_DIRECTORY).parts


def _names_of_metadata_version_function(tree: ast.AST) -> set[str]:
    """The names under which ``version`` of the package metadata is imported in this module"""
    return {alias.asname or alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module in METADATA_MODULES
            for alias in node.names if alias.name == 'version'}


def _asks_the_installed_metadata_for_clearmap(tree: ast.AST) -> list[int]:
    """The lines where ``version('ClearMap')`` of the package metadata is called"""
    names = _names_of_metadata_version_function(tree)
    return [node.lineno for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in names
            and node.args and isinstance(node.args[0], ast.Constant) and node.args[0].value == PACKAGE_NAME]


def test_no_module_asks_the_installed_package_for_the_version_of_clearmap():
    offenders = {}
    for path in sorted(PACKAGE_DIRECTORY.rglob('*.py')):
        if not _is_python_file_of_clearmap(path):
            continue
        try:
            tree = ast.parse(path.read_text(encoding='utf-8'))
        except SyntaxError:  # Legacy Python 2 files: they cannot be imported, so they cannot ask anything
            continue
        lines = _asks_the_installed_metadata_for_clearmap(tree)
        if lines:
            offenders[str(path.relative_to(REPOSITORY_ROOT))] = lines
    assert not offenders, f'Read ClearMap.__version__ instead of the metadata of the installed package: {offenders}'


def test_setup_derives_its_version_instead_of_writing_it():
    setup_call = next(node for node in ast.walk(ast.parse(SETUP_SCRIPT.read_text(encoding='utf-8')))
                      if isinstance(node, ast.Call) and getattr(node.func, 'id', None) == 'setup')
    version_argument = next(keyword.value for keyword in setup_call.keywords if keyword.arg == 'version')
    assert not isinstance(version_argument, ast.Constant), 'setup.py must read ClearMap/_version.py'


def test_the_package_reports_the_version_of_its_source():
    from ClearMap import _version
    assert ClearMap.__version__ == _version.__version__


def test_the_version_is_a_release_number_the_config_layer_can_use():
    release = Version(ClearMap.__version__)  # The config layer builds the names of its folders from major.minor
    assert (release.major, release.minor) >= (3, 1)
