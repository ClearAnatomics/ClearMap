"""
A memory map created with a shape of numpy integers (a shape computed with numpy) can be opened again.

``np.lib.format.open_memmap`` writes the repr of the shape into the npy header. With NumPy 2 the repr of a numpy
integer is ``np.int64(3)``, which the header parser refuses: the file could not be opened again (the wobbly stitching
wrote such a ``main_stitched.npy``).
"""
import numpy as np

import ClearMap.IO.source.backends.mmp_backend as mmp_backend

SHAPE = (np.int64(3), np.int64(4), 2)


def test_a_memmap_created_with_a_shape_of_numpy_integers_can_be_opened_again(tmp_path):
    location = tmp_path / 'stitched.npy'
    mmp_backend.create(location=str(location), shape=SHAPE, dtype='uint16', order='C', as_source=False)

    reopened = np.load(location, mmap_mode='r')

    assert reopened.shape == (3, 4, 2)
