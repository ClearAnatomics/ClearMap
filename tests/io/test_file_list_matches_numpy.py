"""
A file list of tiles is sliced like the array of its tiles. The reference is that array, built in memory:
whatever the numbering of the files (with gaps or not) and the order of the axes, source[slicing] equals array[slicing].
"""
import itertools

import numpy as np
import pytest

from ClearMap.IO.source.backends.file_list_backend import FileListSource

TILE_SHAPE = (4, 3)
# Numbering of the tags in the names, with a gap in each: positions 0, 1, (2) are not 1, 2, 3
NUMBERINGS = {'contiguous': {'X': (1, 2), 'Y': (1, 2, 3)}, 'with gaps': {'X': (1, 3), 'Y': (2, 5, 6)}}
ORDERS_OF_THE_AXES = {'X then Y': ['X', 'Y'], 'Y then X': ['Y', 'X']}
ALL = slice(None)
# Slicings of the two axes of the file list; a list is combined with a slice only, as numpy pairs two lists
SLICINGS_OF_THE_LIST_AXES = [
    (ALL, ALL), (0, ALL), (ALL, 1), (-1, -1), (1, 0), (slice(1, None), ALL), (ALL, slice(None, None, 2)),
    (ALL, slice(-2, None)), ([1, 0], ALL), (ALL, [1, 0]), ([0], ALL), (ALL, [-1, 0, -2]), (np.array([1, 0]), ALL),
]


def _tile_value(x, y):
    return 10 * x + y


@pytest.fixture(params=NUMBERINGS, ids=list(NUMBERINGS))
def numbering(request):
    return NUMBERINGS[request.param]


@pytest.fixture(params=ORDERS_OF_THE_AXES, ids=list(ORDERS_OF_THE_AXES))
def axes_order(request):
    return ORDERS_OF_THE_AXES[request.param]


@pytest.fixture
def source_and_array(tmp_path, numbering, axes_order):
    """The source of the tiles written to files, and the same data as an array of axes (rows, columns, *axes_order)"""
    pixels = np.arange(np.prod(TILE_SHAPE)).reshape(TILE_SHAPE)
    tiles = {}
    for x, y in itertools.product(numbering['X'], numbering['Y']):
        tiles[x, y] = (pixels + 100 * _tile_value(x, y)).astype(np.uint16)
        np.save(tmp_path / f'tile_{x:02d}_{y:02d}.npy', tiles[x, y])
    values_of_the_axes = [numbering[name] for name in axes_order]
    array = np.empty(TILE_SHAPE + tuple(len(values) for values in values_of_the_axes), dtype=np.uint16)
    for positions in itertools.product(*(range(len(values)) for values in values_of_the_axes)):
        values = dict(zip(axes_order, (values[p] for values, p in zip(values_of_the_axes, positions))))
        array[(ALL, ALL) + positions] = tiles[values['X'], values['Y']]
    source = FileListSource(expression=str(tmp_path / 'tile_<X,2>_<Y,2>.npy'), axes_order=axes_order)
    return source, array


@pytest.mark.parametrize('slicing_of_the_list_axes', SLICINGS_OF_THE_LIST_AXES, ids=str)
def test_a_slicing_of_the_tiles_gives_what_it_gives_on_the_array(source_and_array, slicing_of_the_list_axes):
    source, array = source_and_array
    slicing = (ALL, ALL) + slicing_of_the_list_axes

    np.testing.assert_array_equal(np.asarray(source[slicing]), array[slicing])


def test_a_slicing_of_the_pixels_of_each_tile_gives_what_it_gives_on_the_array(source_and_array):
    source, array = source_and_array
    slicing = (slice(1, 3), slice(None, None, 2), ALL, 0)

    np.testing.assert_array_equal(np.asarray(source[slicing]), array[slicing])
