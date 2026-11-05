"""
The shared memory arrays (:mod:`ClearMap.ParallelProcessing.SharedMemoryArray`) are numpy arrays over a
``multiprocessing.RawArray`` of the c type of their dtype.

The c type was looked up in private tables of ``numpy.ctypeslib`` (``_typecodes``, then
``_ctype_from_dtype_scalar``), which NumPy 2 moved: creating any shared array raised an AttributeError.
"""
import ctypes

import numpy as np
import pytest

import ClearMap.ParallelProcessing.SharedMemoryArray as sma

SHAPE = (3, 4)
DTYPES = ('uint8', 'uint16', 'int32', 'int64', 'float32', 'float64', bool)


@pytest.mark.parametrize('dtype', DTYPES)
def test_a_shared_array_has_the_shape_and_dtype_it_is_created_with(dtype):
    array = sma.zeros(SHAPE, dtype=dtype)

    assert array.shape == SHAPE
    assert array.dtype == np.dtype(dtype)
    assert not array.any()


def test_a_shared_array_is_a_view_of_its_shared_memory():
    array = sma.zeros(SHAPE, dtype='float32')

    array[1, 2] = 7

    assert np.frombuffer(sma.base(array), dtype='float32')[1 * SHAPE[1] + 2] == 7


def test_the_c_type_of_a_bool_array_is_a_byte():
    # The numpy wrapper views the bytes of the shared memory as bool
    assert sma.ctype(np.zeros(1, dtype=bool)) is ctypes.c_uint8
