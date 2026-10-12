"""Tests for the tif backend"""
import gc
import warnings

import numpy as np
import pytest
import tifffile

from ClearMap.IO.source.backends import tif_backend

STACK_SHAPE = (3, 4, 5)  # z, y, x


@pytest.fixture
def tif_path(tmp_path):
    path = tmp_path / 'stack.tif'
    tifffile.imwrite(path, np.arange(np.prod(STACK_SHAPE), dtype=np.uint16).reshape(STACK_SHAPE),
                     photometric='minisblack')
    return path


@pytest.mark.parametrize('slicing', [None, (slice(None), slice(None), 0)], ids=['whole', 'sliced'])
def test_read_from_a_path_closes_the_file(tif_path, slicing):
    """read() opens the file itself when given a path, so it must close it (it used to leak the handle)"""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always', ResourceWarning)
        data = tif_backend.read(str(tif_path), slicing=slicing)
        gc.collect()  # An unclosed file handle is reported when it is collected

    assert [str(w.message) for w in caught if issubclass(w.category, ResourceWarning)] == []
    assert np.asarray(data).size > 0  # The data read stays valid after the file is closed


def test_read_from_a_source_leaves_it_open(tif_path):
    """A source given by the caller belongs to the caller: read() does not close it"""
    source = tif_backend.TifSource(str(tif_path))
    first = tif_backend.read(source)
    second = tif_backend.read(source)  # Would fail on a closed file

    np.testing.assert_array_equal(first, second)
    source.close()
