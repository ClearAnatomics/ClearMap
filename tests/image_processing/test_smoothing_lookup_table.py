"""
The smoothing look-up table has 2**27 entries. It is used where the installation prepared it, else made once in the
cache folder: the folder of the program is often read-only (a cluster).
"""
import zipfile

import numpy as np
import pytest

from ClearMap import Settings as settings
from ClearMap.ImageProcessing.Binary import Smoothing

SMALL_TABLE = np.array([False, True, True, False])


def small_table(function, verbose, processes):
    return SMALL_TABLE


@pytest.fixture
def folders(tmp_path, monkeypatch):
    program_dir, cache_dir = tmp_path / 'program', tmp_path / 'cache'
    program_dir.mkdir()
    monkeypatch.setattr(Smoothing, 'SMOOTHING_FOLDER', program_dir)
    monkeypatch.setattr(settings, 'clearmap_cache_dir', cache_dir)
    return program_dir, cache_dir


def test_the_shipped_zip_is_decompressed_in_the_cache_and_not_next_to_the_module(folders):
    program_dir, cache_dir = folders
    np.save(program_dir / 'source.npy', SMALL_TABLE)
    with zipfile.ZipFile(program_dir / 'Smoothing.npy.zip', 'w') as archive:
        archive.write(program_dir / 'source.npy', arcname='Smoothing.npy')
    (program_dir / 'source.npy').unlink()

    table = Smoothing.initialize_lookup_table(verbose=False)

    assert np.array_equal(table, SMALL_TABLE)
    assert (cache_dir / 'Smoothing.npy').exists()
    assert not (program_dir / 'Smoothing.npy').exists()


def test_a_table_prepared_next_to_the_module_is_used_and_the_cache_stays_empty(folders):
    program_dir, cache_dir = folders
    np.save(program_dir / 'Smoothing.npy', SMALL_TABLE)

    table = Smoothing.initialize_lookup_table(verbose=False)

    assert np.array_equal(table, SMALL_TABLE)
    assert not cache_dir.exists()


def test_a_table_without_a_shipped_zip_is_computed_into_the_cache(folders, monkeypatch):
    program_dir, cache_dir = folders
    monkeypatch.setattr(Smoothing, 'generate_lookup_table', small_table)

    table = Smoothing.initialize_lookup_table(verbose=False)

    assert np.array_equal(table, SMALL_TABLE)
    assert (cache_dir / 'Smoothing.npy').exists()
    assert list(program_dir.iterdir()) == []
