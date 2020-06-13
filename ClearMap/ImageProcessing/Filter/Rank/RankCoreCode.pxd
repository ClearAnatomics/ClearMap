#cython: language_level=3, boundscheck=False, wraparound=False, nonecheck=False, initializedcheck=False, cdivision=True

"""
RankCordeCode
=============

Cython definitions for the core rank filter code.
"""
__author__    = 'Christoph Kirst <christoph.kirst.ck@gmail.com>'
__license__   = 'GPLv3 - GNU General Pulic License v3 (see LICENSE.txt)'
__copyright__ = 'Copyright © 2020 by Christoph Kirst'
__webpage__   = 'http://idisco.info'
__download__  = 'http://www.github.com/ChristophKirst/ClearMap2'

from numpy cimport uint8_t, uint16_t, double_t, int64_t

ctypedef fused source_t:
    uint8_t
    uint16_t
    int64_t

ctypedef fused sink_t:
    uint8_t
    uint16_t
    int64_t
    double_t

ctypedef fused index_t:
    Py_ssize_t


cdef void rank_core(void kernel(sink_t*, index_t*, index_t, source_t, index_t, index_t*, double*) nogil,
                    source_t[:, :, :] image, char[:, :, :] selem,
                    sink_t[:, :, :, :] sink, index_t max_bin,
                    index_t[:] parameter_index, double[:] parameter_double) except *


cdef void rank_core_masked(void kernel(sink_t*, index_t*, index_t, source_t, index_t, index_t*, double*) nogil,
                           source_t[:, :, :] source, char[:, :, :] selem, char[:,:,:] mask,
                           sink_t[:, :, :, :] sink, index_t max_bin,
                           index_t[:] parameter_index, double[:] parameter_double) except *