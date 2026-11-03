"""Renaming a batch of files and folders: checked as a whole, moved in two steps, put back on failure"""
from pathlib import Path

import pytest

from ClearMap.IO.FileUtils import FileRenameError, Rename, apply_renames, check_renames


def make(root: Path, *names: str) -> list[Path]:
    paths = []
    for name in names:
        path = root / name
        path.write_text(name)
        paths.append(path)
    return paths


def contents(root: Path) -> dict[str, str]:
    return {path.name: path.read_text() for path in sorted(root.iterdir()) if path.is_file()}


def test_a_file_is_renamed(tmp_path):
    source, = make(tmp_path, 'a.txt')

    apply_renames([Rename(source, tmp_path / 'b.txt')])

    assert contents(tmp_path) == {'b.txt': 'a.txt'}


def test_a_folder_is_renamed_with_what_it_holds(tmp_path):
    (tmp_path / 'old').mkdir()
    (tmp_path / 'old' / 'x.txt').write_text('x')

    apply_renames([Rename(tmp_path / 'old', tmp_path / 'new')])

    assert (tmp_path / 'new' / 'x.txt').read_text() == 'x'
    assert not (tmp_path / 'old').exists()


def test_two_files_swap_their_names(tmp_path):
    a, b = make(tmp_path, 'a.txt', 'b.txt')

    apply_renames([Rename(a, b), Rename(b, a)])

    assert contents(tmp_path) == {'a.txt': 'b.txt', 'b.txt': 'a.txt'}


def test_a_case_only_rename_ends_with_the_new_case(tmp_path):
    source, = make(tmp_path, 'fos_data.txt')

    apply_renames([Rename(source, tmp_path / 'Fos_data.txt')])

    assert [path.name for path in tmp_path.iterdir()] == ['Fos_data.txt']


def test_no_temporary_name_is_left_behind(tmp_path):
    a, b = make(tmp_path, 'a.txt', 'b.txt')

    apply_renames([Rename(a, tmp_path / 'c.txt'), Rename(b, tmp_path / 'd.txt')])

    assert sorted(path.name for path in tmp_path.iterdir()) == ['c.txt', 'd.txt']


def test_a_rename_which_changes_nothing_is_left_out(tmp_path):
    a, = make(tmp_path, 'a.txt')

    assert apply_renames([Rename(a, a)]) == []
    assert contents(tmp_path) == {'a.txt': 'a.txt'}


def test_a_target_which_exists_refuses_the_whole_batch_and_nothing_moves(tmp_path):
    a, b, c = make(tmp_path, 'a.txt', 'b.txt', 'c.txt')

    with pytest.raises(FileRenameError, match='onto what exists'):
        apply_renames([Rename(a, tmp_path / 'z.txt'), Rename(b, c)])

    assert contents(tmp_path) == {'a.txt': 'a.txt', 'b.txt': 'b.txt', 'c.txt': 'c.txt'}


def test_a_target_which_is_moved_away_by_the_batch_is_no_obstacle(tmp_path):
    a, b = make(tmp_path, 'a.txt', 'b.txt')

    apply_renames([Rename(a, tmp_path / 'c.txt'), Rename(b, a)])  # a is free once it is moved to c

    assert contents(tmp_path) == {'c.txt': 'a.txt', 'a.txt': 'b.txt'}


def test_two_renames_to_the_same_target_refuse_the_batch(tmp_path):
    a, b = make(tmp_path, 'a.txt', 'b.txt')

    with pytest.raises(FileRenameError, match='same path'):
        check_renames([Rename(a, tmp_path / 'c.txt'), Rename(b, tmp_path / 'C.txt')])  # Same file on some systems


def test_two_renames_from_the_same_source_refuse_the_batch(tmp_path):
    a, = make(tmp_path, 'a.txt')

    with pytest.raises(FileRenameError, match='same path'):
        check_renames([Rename(a, tmp_path / 'b.txt'), Rename(a, tmp_path / 'c.txt')])


def test_a_source_which_does_not_exist_refuses_the_batch(tmp_path):
    with pytest.raises(FileRenameError, match='does not exist'):
        check_renames([Rename(tmp_path / 'ghost.txt', tmp_path / 'b.txt')])


def test_checking_the_batch_moves_nothing(tmp_path):
    a, = make(tmp_path, 'a.txt')

    assert check_renames([Rename(a, tmp_path / 'b.txt')]) == [Rename(a, tmp_path / 'b.txt')]
    assert contents(tmp_path) == {'a.txt': 'a.txt'}


@pytest.mark.parametrize('failing_move', [1, 2, 3, 4])
def test_a_move_which_fails_puts_everything_back(tmp_path, monkeypatch, failing_move):
    a, b = make(tmp_path, 'a.txt', 'b.txt')
    original_rename = Path.rename
    calls = []

    def failing_rename(self, target):
        calls.append(self)
        if len(calls) == failing_move:
            raise PermissionError('locked')
        return original_rename(self, target)

    monkeypatch.setattr(Path, 'rename', failing_rename)

    with pytest.raises(FileRenameError, match='put back'):
        apply_renames([Rename(a, tmp_path / 'c.txt'), Rename(b, tmp_path / 'd.txt')])

    monkeypatch.undo()
    assert contents(tmp_path) == {'a.txt': 'a.txt', 'b.txt': 'b.txt'}


def test_what_cannot_be_put_back_is_reported(tmp_path, monkeypatch):
    a, b = make(tmp_path, 'a.txt', 'b.txt')
    original_rename = Path.rename
    calls = []

    def failing_rename(self, target):
        calls.append(self)
        if len(calls) >= 3:  # The first move of the second step fails, and so does every put back
            raise PermissionError('locked')
        return original_rename(self, target)

    monkeypatch.setattr(Path, 'rename', failing_rename)

    with pytest.raises(FileRenameError, match='could not put everything back'):
        apply_renames([Rename(a, tmp_path / 'c.txt'), Rename(b, tmp_path / 'd.txt')])


def test_what_is_inside_a_renamed_folder_is_renamed_before_the_folder(tmp_path):
    folder = tmp_path / 'a_to_b'
    folder.mkdir()
    (folder / 'a_landmarks.pts').write_text('a')
    (folder / 'b_landmarks.pts').write_text('b')

    apply_renames([Rename(folder, tmp_path / 'x_to_b'),
                   Rename(folder / 'a_landmarks.pts', folder / 'x_landmarks.pts')])

    assert sorted(path.name for path in (tmp_path / 'x_to_b').iterdir()) == ['b_landmarks.pts', 'x_landmarks.pts']
    assert not folder.exists()


def test_a_failure_while_renaming_inside_a_folder_puts_everything_back(tmp_path, monkeypatch):
    folder = tmp_path / 'a_to_b'
    folder.mkdir()
    (folder / 'a_landmarks.pts').write_text('a')
    original_rename = Path.rename
    calls = []

    def failing_rename(self, target):
        calls.append(self)
        if len(calls) == 3:  # The folder itself, after what is inside it moved
            raise PermissionError('locked')
        return original_rename(self, target)

    monkeypatch.setattr(Path, 'rename', failing_rename)

    with pytest.raises(FileRenameError, match='put back'):
        apply_renames([Rename(folder, tmp_path / 'x_to_b'),
                       Rename(folder / 'a_landmarks.pts', folder / 'x_landmarks.pts')])

    monkeypatch.undo()
    assert [path.name for path in folder.iterdir()] == ['a_landmarks.pts']
    assert not (tmp_path / 'x_to_b').exists()


def test_renames_nested_on_several_levels_move_the_innermost_first(tmp_path):
    folder, subfolder = tmp_path / 'a', tmp_path / 'a' / 'b'
    subfolder.mkdir(parents=True)
    (subfolder / 'c.txt').write_text('c')

    apply_renames([Rename(folder, tmp_path / 'x'),
                   Rename(subfolder, folder / 'y'),
                   Rename(subfolder / 'c.txt', subfolder / 'z.txt')])

    assert [str(path.relative_to(tmp_path)) for path in tmp_path.rglob('*') if path.is_file()] == ['x/y/z.txt']
