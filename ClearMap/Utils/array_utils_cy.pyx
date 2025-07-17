#cython: language_level=3, boundscheck=False, wraparound=False, nonecheck=False, initializedcheck=False

import numpy as np
cimport numpy as np

ctypedef fused index_t:
    int
    long
    np.int32_t
    np.int64_t

ctypedef fused source_t:
    int
    long
    float
    np.int16_t
    np.int32_t
    np.int64_t
    np.uint8_t  # used for bool
    np.uint16_t
    np.uint32_t
    np.uint64_t
    np.float32_t
    np.float64_t


cpdef source_t[:] remap_array_ranges_1_d(source_t[:] src_array, source_t[:] new_array,
                                        index_t[:, :] src_ranges, index_t[:, :] new_ranges):
    cdef size_t i
    for i in range(src_ranges.shape[0]):
        new_array[new_ranges[i, 0]:new_ranges[i, 1]] = src_array[src_ranges[i, 0]:src_ranges[i, 1]]
    return new_array


cpdef source_t[:, :] remap_array_ranges_2_d(source_t[:, :] src_array, source_t[:, :] new_array,
                                        index_t[:, :] src_ranges, index_t[:, :] new_ranges):
    cdef size_t i
    for i in range(src_ranges.shape[0]):
        new_array[new_ranges[i, 0]:new_ranges[i, 1], :] = src_array[src_ranges[i, 0]:src_ranges[i, 1], :]
    return new_array
