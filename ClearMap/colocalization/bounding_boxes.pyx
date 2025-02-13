# Copyright Charly Rousseau & Gaël Cousin

# cython: language="c++", language_level=3
# distutils: language = c++
cimport cython
import numpy as np
cimport numpy as np
from libcpp.map cimport map as cpp_map
from libcpp.vector cimport vector as cpp_vector
# from libcpp.array cimport array as cpp_array


ctypedef fused source_int_t:
  np.int32_t
  np.int64_t
  np.uint8_t
  np.uint16_t
  np.uint32_t

ctypedef Py_ssize_t index_t

# ctypedef unsigned long long ull


# cdef struct Point:
#     ull x
#     ull y
#     ull z


# cpdef find_values_coordinates(np.ndarray[source_int_t, ndim=3] arr):
#     cdef cpp_map[source_int_t, cpp_vector[Point]] result
#     cdef index_t i, j, k
#     cdef source_int_t value

#     cdef index_t dim1 = arr.shape[0]
#     cdef index_t dim2 = arr.shape[1]
#     cdef index_t dim3 = arr.shape[2]

#     with cython.boundscheck(False), cython.wraparound(False):
#         for i in range(dim1):
#             for j in range(dim2):
#                 for k in range(dim3):
#                     value = arr[i, j, k]
#                     if not result.count(value):
#                         result[value] = cpp_vector[Point]()
#                     result[value].push_back(Point(i, j, k))
#     return result


cpdef bbox_3d(np.ndarray[source_int_t, ndim=3] img):
    cdef source_int_t n_labels = img.max() + 1  # WARNING: because +1 if max, could overflow
    # what we do for zero could be done for any axis
    cdef np.ndarray[np.uint32_t, ndim=3] res = np.empty((n_labels, 3, 2), dtype='uint32')
    cdef index_t i, j, k

    cdef index_t shape[3]
    cdef index_t max_shape = 0
    for i in range(3):
        shape[i] = img.shape[i]
        if shape[i] > max_shape:
            max_shape = shape[i]
    res[:, :, 0] = max_shape + 1
    res[:, :, 1] = 0

    cdef source_int_t val
    cdef index_t coordinates[3]
    with cython.boundscheck(False), cython.wraparound(False):
        for i in range(shape[0]):
            for j in range(shape[1]):
                for k in range(shape[2]):
                    coordinates[0] = i
                    coordinates[1] = j
                    coordinates[2] = k   # TODO: check if we could do a one liner without loosing performance
                    val = img[i, j, k]
                    for axis in range(3):
                        res[val, axis, 0] = min(coordinates[axis], res[val, axis, 0])
                        res[val, axis, 1] = max(coordinates[axis], res[val, axis, 1])
    return res


cpdef bbox_2d(np.ndarray[source_int_t, ndim=2] img):
    cdef source_int_t n_labels = img.max() + 1  # WARNING: because +1 if max, could overflow
    # what we do for zero could be done for any axis
    cdef np.ndarray[np.uint32_t, ndim=3] res = np.empty((n_labels, 2, 2), dtype='uint32')
    cdef index_t i, j

    cdef index_t shape[2]
    cdef index_t max_shape = 0
    for i in range(2):
        shape[i] = img.shape[i]
        if shape[i] > max_shape:
            max_shape = shape[i]
    res[:, :, 0] = max_shape + 1
    res[:, :, 1] = 0

    cdef source_int_t val
    cdef index_t coordinates[2]
    with cython.boundscheck(False), cython.wraparound(False):
        for i in range(shape[0]):
            for j in range(shape[1]):
                coordinates[0] = i
                coordinates[1] = j
                val = img[i, j]
                for axis in range(2):
                    res[val, axis, 0] = min(coordinates[axis], res[val, axis, 0])
                    res[val, axis, 1] = max(coordinates[axis], res[val, axis, 1])
    return res



cpdef bbox_1d(np.ndarray[source_int_t, ndim=1] img):
    cdef source_int_t n_labels = img.max() + 1  # WARNING: because +1 if max, could overflow
    # what we do for zero could be done for any axis
    cdef np.ndarray[np.uint32_t, ndim=3] res = np.empty((n_labels, 1, 2), dtype='uint32')
    cdef index_t i, j

    cdef index_t shape[1]
    cdef index_t max_shape = 0
    for i in range(1):
        shape[i] = img.shape[i]
        if shape[i] > max_shape:
            max_shape = shape[i]
    res[:, :, 0] = max_shape + 1
    res[:, :, 1] = 0

    cdef source_int_t val
    cdef index_t coordinates[1]
    with cython.boundscheck(False), cython.wraparound(False):
        for i in range(shape[0]):
                coordinates[0] = i
                val = img[i]
                for axis in range(2):
                    res[val, axis, 0] = min(coordinates[axis], res[val, axis, 0])
                    res[val, axis, 1] = max(coordinates[axis], res[val, axis, 1])
    return res