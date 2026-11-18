"""The scans of the source read what the repository maintains: not the files git ignores, which an old clone may hold"""
import subprocess

from tests.source_scan import maintained_syntax_trees


def _scanned(root):
    return sorted(path.as_posix() for path, _ in maintained_syntax_trees(('ClearMap',), root=root))


def _write(root, relative_path, text='x = 1\n'):
    path = root / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def _repository_with_ignored_files(root):
    _write(root, '.gitignore', 'spikes/\n*_rc.py\n')
    _write(root, 'ClearMap/tracked.py')
    _write(root, 'ClearMap/gui/spikes/ignored_by_folder.py')
    _write(root, 'ClearMap/gui/creator/icons_rc.py')
    subprocess.run(['git', '-C', str(root), 'init', '-q'], check=True)
    subprocess.run(['git', '-C', str(root), 'add', '.gitignore', 'ClearMap/tracked.py'], check=True)
    _write(root, 'ClearMap/new_and_not_added.py')  # A new module is scanned before it is added


def test_the_files_which_git_ignores_are_not_scanned(tmp_path):
    _repository_with_ignored_files(tmp_path)

    assert _scanned(tmp_path) == ['ClearMap/new_and_not_added.py', 'ClearMap/tracked.py']


def test_outside_a_git_repository_only_the_not_maintained_folders_are_left_out(tmp_path):
    _write(tmp_path, 'ClearMap/one.py')
    _write(tmp_path, 'ClearMap/gui/two_rc.py')
    _write(tmp_path, 'ClearMap/custom_scripts/mine.py')

    assert _scanned(tmp_path) == ['ClearMap/gui/two_rc.py', 'ClearMap/one.py']
