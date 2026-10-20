"""The renames the user vets in the dialog: the ones kept, to the names typed, checked as a whole"""
import pytest

from ClearMap.IO.FileUtils import FileRenameError, Rename
from ClearMap.gui.dialogs import vetted_renames


@pytest.fixture()
def two_files(tmp_path):
    first, second = tmp_path / 'cfos_cells.npy', tmp_path / 'cfos_density.tif'
    first.touch()
    second.touch()
    return [Rename(first, tmp_path / 'fos_cells.npy'), Rename(second, tmp_path / 'fos_density.tif')]


def test_the_renames_left_as_proposed_are_applied_as_proposed(two_files):
    assert vetted_renames(two_files, [(True, 'fos_cells.npy'), (True, 'fos_density.tif')]) == two_files


def test_an_unticked_rename_is_left_out(two_files):
    assert vetted_renames(two_files, [(False, 'fos_cells.npy'), (True, 'fos_density.tif')]) == two_files[1:]


def test_a_typed_name_renames_in_the_same_folder(two_files, tmp_path):
    vetted = vetted_renames(two_files, [(True, ' cfos_v2_cells.npy '), (False, '')])

    assert vetted == [Rename(tmp_path / 'cfos_cells.npy', tmp_path / 'cfos_v2_cells.npy')]


@pytest.mark.parametrize('typed_name', ['', '  ', '.', '..', 'sub/fos_cells.npy', 'sub\\fos_cells.npy'])
def test_a_typed_name_which_is_not_a_file_name_is_refused(two_files, typed_name):
    with pytest.raises(FileRenameError, match='is not a file name'):
        vetted_renames(two_files, [(True, typed_name), (False, '')])


def test_two_files_typed_to_the_same_name_are_refused(two_files):
    with pytest.raises(FileRenameError, match='same path'):
        vetted_renames(two_files, [(True, 'fos.npy'), (True, 'fos.npy')])


def test_a_typed_name_onto_a_file_which_stays_is_refused(two_files):
    with pytest.raises(FileRenameError, match='onto what exists'):
        vetted_renames(two_files, [(True, 'cfos_density.tif'), (False, '')])


def test_typing_back_the_current_name_renames_nothing(two_files):
    assert vetted_renames(two_files, [(True, 'cfos_cells.npy'), (False, '')]) == []
