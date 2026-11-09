"""
The About dialog shows the commit that ``get_commit_info`` reports (the stamp of the installation, or git), not a
``commit_info.py`` written into the source tree by the installer.

``ClearMap.gui.app`` builds the application when it is imported, so its source is read instead.
"""
import ast
from pathlib import Path

APP_SOURCE = Path(__file__).resolve().parents[2] / 'ClearMap' / 'gui' / 'app.py'
STALE_COMMIT_MODULE = 'commit_info'


def _imported_names(tree):
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            yield node.module or ''
            yield from (alias.name for alias in node.names)
        elif isinstance(node, ast.Import):
            yield from (alias.name for alias in node.names)


def test_the_about_dialog_does_not_import_a_file_written_at_install_time():
    tree = ast.parse(APP_SOURCE.read_text(encoding='utf-8'))

    stale = [name for name in _imported_names(tree) if name.split('.')[-1] == STALE_COMMIT_MODULE]

    assert not stale, f'app.py imports {stale}: ask ClearMap.Utils.git_info.get_commit_info instead'


def test_the_about_dialog_asks_get_commit_info_for_the_commit():
    tree = ast.parse(APP_SOURCE.read_text(encoding='utf-8'))

    imported = [node for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
                and node.module == 'ClearMap.Utils.git_info' and any(a.name == 'get_commit_info' for a in node.names)]
    called = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
              and getattr(node.func, 'id', None) == 'get_commit_info']

    assert imported and called
