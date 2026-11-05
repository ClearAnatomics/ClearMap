"""
A compressed mhd file (zlib, ``CompressedData = True``) reads back the array that was written.

The raw data was encoded with ``ndarray.tostring()``, which NumPy 2 removed (AttributeError).
"""
import numpy as np

import ClearMap.IO.source.backends.mhd_backend as mhd_backend

SHAPE = (4, 3, 2)  # Three different sizes, so that a transposed result does not compare equal


def test_a_compressed_mhd_file_reads_back_the_array_written(tmp_path):
    array = np.arange(np.prod(SHAPE), dtype=np.uint16).reshape(SHAPE)
    header, header_file, raw_file = mhd_backend._header(location=str(tmp_path / 'volume.mhd'), array=array,
                                                        compression=True, return_header_and_raw_file=True)
    mhd_backend._write_header(header_file, header)

    mhd_backend._write_raw(raw_file, array, compression=True)

    np.testing.assert_array_equal(mhd_backend.MhdSource(header_file).array, array)
