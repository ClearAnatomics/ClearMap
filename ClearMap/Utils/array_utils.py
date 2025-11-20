"""
This module provides functions to remap the values of an array from one range to another.
It is essentially a fronted to optimized ``Cython`` functions.

The function remap_array_ranges is the main function of this module.
It takes as input the source array, the new array, the source ranges, and the new ranges and
returns the remapped array.
"""
import os
import warnings

import numpy as np
from pyximport import pyximport

pyximport.install(setup_args={"include_dirs": [np.get_include(), os.path.dirname(os.path.abspath(__file__))]},
                  reload_support=True)

from ClearMap.Utils.array_utils_cy import remap_array_ranges_1_d, remap_array_ranges_2_d


def remap_array_ranges(src_array, new_array, src_ranges, new_ranges):
    if not np.all(src_ranges[:, 1] >= src_ranges[:, 0]):
        raise ValueError("src_ranges not sorted, please flip edges if necessary")
    if not np.all(new_ranges[:, 1] >= new_ranges[:, 0]):
        raise ValueError("new_ranges not sorted, please flip edges if necessary")

    if src_ranges.max() > src_array.shape[0]:
        raise ValueError(f'src_ranges {src_ranges.max()} is larger than the source array shape {src_array.shape[0]}')
    if new_ranges.max() > new_array.shape[0]:
        raise ValueError(f'new_ranges {new_ranges.max()} is larger than the new array shape {new_array.shape[0]}')

    src_len = np.diff(src_ranges, axis=1).ravel()
    dst_len = np.diff(new_ranges, axis=1).ravel()

    assert src_len.shape == dst_len.shape, "Source and destination ranges do not have the same length."

    functions_map = {
        1: remap_array_ranges_1_d,
        2: remap_array_ranges_2_d
    }
    if src_array.ndim not in functions_map.keys():
        raise ValueError(f'remap_array_ranges only supports arrays with dimensions in {list(functions_map.keys())}')
    else:
        f = functions_map[src_array.ndim]
        result = f(src_array, new_array, src_ranges, new_ranges)
        return np.asarray(result, dtype=src_array.dtype)

    # warnings.warn(f'Source and destination ranges do not have the same length, defaulting to np.take.')
    #
    # new_n_rows = int(new_ranges[-1, 1])
    #
    # out_shape = (new_n_rows,) + src_array.shape[1:]
    # out = np.zeros(out_shape, dtype=src_array.dtype)
    #
    # for (src_start, src_end), (new_start, new_end) in zip(src_ranges, new_ranges):
    #     out[new_start:new_end] = src_array[src_start:src_end]
    #
    # return out


def dtype_range(arr):
    """
    Return (min,max) that the dtype *could* hold.

    Parameters
    ----------
    arr : array_like
        The array whose dtype is to be checked.

    Returns
    -------
    min, max : tuple
        The minimum and maximum values that the dtype could hold.
    """
    if np.issubdtype(arr.dtype, np.integer):
        info = np.iinfo(arr.dtype)
        return info.min, info.max
    elif np.issubdtype(arr.dtype, np.floating):
        info = np.finfo(arr.dtype)
        return info.min, info.max
    else:                   # bool, complex, etc.
        return 0.0, 1.0
