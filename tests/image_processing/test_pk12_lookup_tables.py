"""
The skeletonization look-up tables have 2**26 entries. They are made when first used, in a cache folder, so that
importing the module neither takes long nor writes in the (possibly read-only) folder of the program.
"""
import importlib
import zipfile

import numpy as np
import pytest

from ClearMap import Settings as settings
from ClearMap.ImageProcessing.Skeletonization import PK12

SMALL_TABLE = np.array([True, False, False, True])


def small_table(function):
    return SMALL_TABLE


def never_called(*args, **kwargs):
    raise AssertionError('a look-up table was read or written while importing the module')


@pytest.fixture
def folders(tmp_path, monkeypatch):
    program_dir, cache_dir = tmp_path / 'program', tmp_path / 'cache'
    program_dir.mkdir()
    monkeypatch.setattr(PK12, 'PK12_FOLDER', program_dir)
    monkeypatch.setattr(settings, 'clearmap_cache_dir', cache_dir)
    return program_dir, cache_dir


def write_zipped_table(program_dir, table):
    npy_path = program_dir / 'source.npy'
    np.save(npy_path, table)
    with zipfile.ZipFile(program_dir / 'PK12.npy.zip', 'w') as archive:
        archive.write(npy_path, arcname='PK12.npy')
    npy_path.unlink()


def test_importing_the_module_does_not_read_nor_write_a_table(monkeypatch):
    monkeypatch.setattr(np, 'load', never_called)
    monkeypatch.setattr(np, 'save', never_called)

    importlib.reload(PK12)


def test_the_shipped_zip_is_decompressed_in_the_cache_and_not_next_to_the_module(folders):
    program_dir, cache_dir = folders
    write_zipped_table(program_dir, SMALL_TABLE)

    table = PK12.initialize_lookup_table()

    assert np.array_equal(table, SMALL_TABLE)
    assert (cache_dir / 'PK12.npy').exists()
    assert not (program_dir / 'PK12.npy').exists()


def test_a_table_prepared_next_to_the_module_is_used_and_the_cache_stays_empty(folders):
    program_dir, cache_dir = folders
    np.save(program_dir / 'PK12.npy', SMALL_TABLE)

    table = PK12.initialize_lookup_table()

    assert np.array_equal(table, SMALL_TABLE)
    assert not cache_dir.exists()


def test_a_table_without_a_shipped_zip_is_computed_into_the_cache(folders, monkeypatch):
    program_dir, cache_dir = folders
    monkeypatch.setattr(PK12, 'generate_lookup_table', small_table)

    table = PK12.initialize_lookup_table()

    assert np.array_equal(table, SMALL_TABLE)
    assert (cache_dir / 'PK12.npy').exists()
    assert list(program_dir.iterdir()) == []


def test_the_former_module_level_tables_are_still_there_and_loaded_once(monkeypatch):
    calls = []

    def count_calls(function=None, filename=None):
        calls.append(filename)
        return SMALL_TABLE

    monkeypatch.setattr(PK12, 'initialize_lookup_table', count_calls)
    PK12.get_delete_table.cache_clear()
    try:
        assert np.array_equal(PK12.delete, SMALL_TABLE)
        assert np.array_equal(PK12.keep, ~SMALL_TABLE)
        assert len(calls) == 1
    finally:
        PK12.get_delete_table.cache_clear()


def test_an_unknown_module_attribute_is_still_an_attribute_error():
    with pytest.raises(AttributeError, match='no_such_table'):
        PK12.no_such_table
