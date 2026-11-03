"""Converting tiles (tif -> npy) must write the pixel data, not a wrapper around the source"""
import numpy as np
import pytest
import tifffile

from ClearMap.IO import conversion, io_ops


@pytest.fixture
def tile(tmp_path):
    data = np.arange(3 * 4 * 5, dtype=np.uint16).reshape(3, 4, 5)
    path = tmp_path / '13-20-34_4x-fastmode_Blaze[00 x 00]_C01.ome.tif'
    tifffile.imwrite(path, data, ome=True, metadata={'axes': 'ZYX'})
    return path, data


@pytest.mark.parametrize('processes', ['serial', 1], ids=['serial', 'pool'])
def test_convert_tile_writes_numeric_npy(tile, processes):
    path, data = tile
    sinks = conversion.convert_files([str(path)], extension='.npy', processes=processes)

    assert len(sinks) == 1
    sink = io_ops.read(sinks[0])
    assert sink.dtype == data.dtype
    assert sink.shape == io_ops.open_ro(str(path)).shape
    assert sink.sum() == data.sum()
