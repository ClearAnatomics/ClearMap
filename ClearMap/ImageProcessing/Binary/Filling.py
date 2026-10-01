"""
Filling
=======

Parallel binary filling on arbitraily sized images.
"""
__author__ = 'Christoph Kirst <christoph.kirst.ck@gmail.com>'
__license__ = 'GPLv3 - GNU General Public License v3 (see LICENSE.txt)'
__copyright__ = 'Copyright © 2020 by Christoph Kirst'
__webpage__ = 'https://idisco.info'
__download__ = 'https://github.com/ClearAnatomics/ClearMap'

import os
import gc

import numpy as np

import pyximport

pyximport.install(setup_args={"include_dirs": [np.get_include(), os.path.dirname(os.path.abspath(__file__))]},
                  reload_support=True)

from . import FillingCode as code

from ClearMap.IO import source_geometry
from ClearMap.Utils.utilities import sanitize_n_processes
import ClearMap.Utils.Timer as tmr
import ClearMap.Utils.array_checks as ac


#%%############################################################################
# ## Binarization
###############################################################################

def fill(source, sink=None, seeds=None, processes=None, verbose=False):
    """Fill binary holes.

    Arguments
    ---------
    source: np.ndarray
        Input source.
    sink: np.ndarray or None
        If None, a new array is allocated in memory.
    seeds : array or None
        An array of seed points for the fill operation.
        If None, the border indices of the source are used as seeds.
    processes: int or None
        How many CPU cores to use, None defaults to max
    verbose: bool
        Whether to print verbose output

    Returns
    -------
    sink : array
        Binary image with filled holes.
    """
    if source is sink:
        raise NotImplementedError("Cannot perform operation in place.")

    if verbose:
        print('Binary filling: initialized!', flush=True)
        timer = tmr.Timer()

    # The flat source, temp and sink arrays must share the same memory layout, as flat indices and
    # strides computed from one are used on the others: work on a contiguous source.
    order = source_geometry.order(source)  # keep the layout of contiguous sources
    if order not in ('C', 'F'):  # non contiguous: copy to ClearMap's default (Fortran) order
        order = 'F'
    source = np.asarray(source, order=order)  # copy only if not contiguous (read-only is fine)

    # create temporary shared array
    temp = np.empty(source.shape, dtype='int8', order=order)

    source_flat = ac.as_uint8_flags(source.reshape(-1, order=order), name='source')  # non zero is foreground
    temp_flat = temp.reshape(-1, order=order)

    processes = sanitize_n_processes(processes)

    if verbose:
        print(f'\tBinary filling: Using {processes} processes', flush=True)
        print('\tBinary filling: temporary arrays created', flush=True)

    # prepare flood fill
    code.prepare_temp(source_flat, temp_flat, processes=processes)
    if verbose:
        print('\tBinary filling: flood fill prepared', flush=True)

    # flood fill in parallel using mp
    if seeds is None:
        seeds = border_indices(source)
    else:
        seeds = np.asarray(seeds)
        if seeds.shape != source.shape:
            raise ValueError(f'The seeds shape {seeds.shape!r} does not match the source shape {source.shape!r}!')
        (seeds, ) = np.where(seeds.reshape(-1, order=order))
    seeds = ac.as_index_array(seeds, name='seeds', ndim=1)

    strides = ac.as_index_array(source_geometry.element_strides(temp), name='strides', ndim=1)

    code.label_temp(temp_flat, strides, seeds, processes=processes)
    if verbose:
        print('\tBinary filling: temporary array labeled!', flush=True)

    if sink is None:
        sink = np.empty(source.shape, dtype=bool, order=order)
    if sink.shape != source.shape:
        raise ValueError(f'The sink shape {sink.shape!r} does not match the source shape {source.shape!r}!')
    sink_flat = ac.check_dtype(ac.bool_as_uint8(sink.reshape(-1, order=order)), (np.uint8,), name='sink')
    if not np.shares_memory(sink_flat, sink):  # the result would be written to a copy
        raise ValueError(f'The sink must be {order}-contiguous like the source!')

    code.fill(source_flat, temp_flat, sink_flat, processes=processes, verbose=verbose)

    if verbose:
        timer.print_elapsed_time('Binary filling')

    del temp, temp_flat
    gc.collect()

    return sink


def border_indices(source):
    """Returns the flat indices of the border pixels in source"""

    ndim = source.ndim
    shape = source.shape
    strides = source_geometry.element_strides(source)

    border = []
    for d in range(ndim):
        offsets = tuple(0 if i > d else 1 for i in range(ndim))
        for c in [0, shape[d]-1]:
            sl = tuple(slice(o, None if o == 0 else -o) if i != d else c for i, o in enumerate(offsets))
            where = np.where(np.logical_not(source[sl]))
            n = len(where[0])
            if n > 0:
                indices = np.zeros(n, dtype=np.intp)
                l = 0
                for k in range(ndim):
                    if k == d:
                        indices += strides[k] * c
                    else:
                        indices += strides[k] * (where[l] + offsets[k])
                        l += 1
                border.append(indices)
    empty_array = np.zeros(0, dtype=np.intp)
    return np.concatenate(border) if border else empty_array


def _test():
    """Tests."""
    import numpy as np
    import ClearMap.Visualization.Plot3d as p3d
    import ClearMap.ImageProcessing.Binary.Filling as bf

    from importlib import reload
    reload(bf)

    test = np.zeros((50, 50, 50), dtype=bool)
    test[20:30, 30:40, 25:35] = True
    test[25:27, 34: 38, 27: 32] = False
    test[5:15, 5:15, 23:35] = True
    test[8:12, 8:12, 27:32] = False

    filled = bf.fill(test, sink=None, processes=10)
    p3d.plot([test, filled])
