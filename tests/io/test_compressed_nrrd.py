"""
A gzip or a bz2 nrrd file reads back the array that was written, as an array that can be modified.

The data was encoded with ``ndarray.tostring()`` and decoded with the binary mode of ``np.fromstring()``, which
NumPy 2 removed (AttributeError and ValueError). The bz2 file was opened with ``bz2.BZ2File(fileobj=...)``, the
Python 2 signature: in Python 3 the file object is the first argument and a bz2 nrrd could not be read at all.
"""
import numpy as np
import pytest

import ClearMap.IO.source.backends.nrrd_backend as nrrd_backend

SHAPE = (4, 3, 2)  # Three different sizes, so that a transposed result does not compare equal
ENCODINGS = ('gzip', 'bz2')


def _write_nrrd(location, encoding):
    array = np.arange(np.prod(SHAPE), dtype=np.uint16).reshape(SHAPE)
    nrrd_backend._write(location, array, options={'encoding': encoding})
    return array


@pytest.mark.parametrize('encoding', ENCODINGS)
def test_a_compressed_nrrd_file_reads_back_the_array_written(tmp_path, encoding):
    location = str(tmp_path / 'volume.nrrd')

    array = _write_nrrd(location, encoding)

    np.testing.assert_array_equal(nrrd_backend.read(location), array)


@pytest.mark.parametrize('encoding', ENCODINGS)
def test_a_compressed_nrrd_file_reads_back_a_writable_array(tmp_path, encoding):
    # np.frombuffer() over bytes is read-only; np.fromstring() returned a copy
    location = str(tmp_path / 'volume.nrrd')
    _write_nrrd(location, encoding)

    assert nrrd_backend.read(location).flags.writeable
