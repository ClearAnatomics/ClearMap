import os

import numpy as np
from pyximport import pyximport

from ClearMap.Utils.utilities import patch_distutils_get_extension

patch_distutils_get_extension()
pyximport.install(setup_args={"include_dirs": [np.get_include(), os.path.dirname(os.path.abspath(__file__))]},
                  reload_support=True)

from ClearMap.Utils.array_utils_cy import remap_array_ranges_1_d, remap_array_ranges_2_d


def remap_array_ranges(src_array, new_array, src_ranges, new_ranges):
    functions_map = {
        1: remap_array_ranges_1_d,
        2: remap_array_ranges_2_d
    }
    if src_array.ndim not in functions_map.keys():
        raise ValueError(f"Only arrays with dimensions {' or '.join([str(k) for k in functions_map.keys()])}"
                         f" are supported")
    else:
        result = functions_map[src_array.ndim](src_array, new_array, src_ranges, new_ranges)
        return np.asarray(result, dtype=src_array.dtype)
