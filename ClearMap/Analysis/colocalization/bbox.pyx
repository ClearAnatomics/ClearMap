# Copyright Charly Rousseau & Gaël Cousin

# cython: language="c++", language_level=3
# distutils: language = c++
cimport cython
import numpy as np
cimport numpy as cnp


ctypedef fused source_int_t:
  cnp.int32_t
  cnp.int64_t
  cnp.uint8_t
  cnp.uint16_t
  cnp.uint32_t

ctypedef Py_ssize_t index_t


# Define inline C-level min and max functions. These will be inlined by the C compiler for maximum speed.
cdef inline source_int_t c_min(source_int_t a, source_int_t b):
    return a if a < b else b

cdef inline source_int_t c_max(source_int_t a, source_int_t b):
    return a if a > b else b


cpdef bbox_3d(cnp.ndarray[source_int_t, ndim=3] img_):
    print(img_.dtype, img_.max(), img_.shape[0], img_.shape[1], img_.shape[2])
    cdef source_int_t[:, :, :]img = img_
    cdef source_int_t n_labels = img_.max() + 1  # WARNING: because +1 if max, could overflow
    # what we do for zero could be done for any axis
    cdef cnp.ndarray[cnp.uint32_t, ndim=3] res_ = np.empty((n_labels, 3, 2), dtype='uint32')
    cdef cnp.uint32_t[:,:,:] res = res_
    cdef index_t i, j, k, axis

    cdef index_t shape[3]
    cdef index_t max_shape = 0
    for i in range(3):
        shape[i] = img_.shape[i]
        if shape[i] > max_shape:
            max_shape = shape[i]
    res[:, :, 0] = max_shape + 1
    res[:, :, 1] = 0

    cdef source_int_t val
    cdef index_t coordinates[3]
    cdef index_t coord_val = 0
    with cython.boundscheck(False), cython.wraparound(False):
        for i in range(shape[0]):
            for j in range(shape[1]):
                for k in range(shape[2]):
                    coordinates[0] = i
                    coordinates[1] = j
                    coordinates[2] = k   # TODO: check if we could do a one liner without loosing performance
                    val = img[i, j, k]
                    for axis in range(3):
                        coord_val = coordinates[axis]
                        res[val, axis, 0] = c_min(coord_val, res[val, axis, 0])
                        res[val, axis, 1] = c_max(coord_val, res[val, axis, 1])
    return res_


cpdef bbox_2d(cnp.ndarray[source_int_t, ndim=2] img_):
    cdef source_int_t[:, :] img = img_
    cdef source_int_t n_labels = img_.max() + 1  # WARNING: because +1 if max, could overflow
    # what we do for zero could be done for any axis
    cdef cnp.ndarray[cnp.uint32_t, ndim=3] res_ = np.empty((n_labels, 2, 2), dtype='uint32')
    cdef cnp.uint32_t[:,:,:] res = res_
    cdef index_t i, j, axis

    cdef index_t shape[2]
    cdef index_t max_shape = 0
    for i in range(2):
        shape[i] = img_.shape[i]
        if shape[i] > max_shape:
            max_shape = shape[i]
    res[:, :, 0] = max_shape + 1
    res[:, :, 1] = 0

    cdef source_int_t val
    cdef index_t coordinates[2]
    cdef index_t coord_val = 0
    with cython.boundscheck(False), cython.wraparound(False):
        for i in range(shape[0]):
            for j in range(shape[1]):
                coordinates[0] = i
                coordinates[1] = j
                val = img[i, j]
                for axis in range(2):
                    coord_val = coordinates[axis]
                    res[val, axis, 0] = c_min(coord_val, res[val, axis, 0])
                    res[val, axis, 1] = c_max(coord_val, res[val, axis, 1])
    return res_


cpdef bbox_1d(cnp.ndarray[source_int_t, ndim=1] img_):
    cdef source_int_t[:] img = img_
    cdef source_int_t n_labels = img_.max() + 1  # WARNING: because +1 if max, could overflow
    # what we do for zero could be done for any axis
    cdef cnp.ndarray[cnp.uint32_t, ndim=3] res_ = np.empty((n_labels, 1, 2), dtype='uint32')
    cdef cnp.uint32_t[:,:,:] res = res_
    cdef index_t i, j, axis

    cdef index_t shape[1]
    cdef index_t max_shape = 0
    for i in range(1):
        shape[i] = img_.shape[i]
        if shape[i] > max_shape:
            max_shape = shape[i]
    res[:, :, 0] = max_shape + 1
    res[:, :, 1] = 0

    cdef source_int_t val
    cdef index_t coordinates[1]
    cdef index_t coord_val = 0
    with cython.boundscheck(False), cython.wraparound(False):
        for i in range(shape[0]):
            coordinates[0] = i
            val = img[i]
            axis = 0
            coord_val = coordinates[axis]
            res[val, axis, 0] = c_min(coord_val, res[val, axis, 0])
            res[val, axis, 1] = c_max(coord_val, res[val, axis, 1])
    return res_
