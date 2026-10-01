"""Tests of the Python wrappers of the Cython modules, with the compiled modules replaced by stubs.

The stubs record the arguments the wrappers pass (so dtypes/shapes/layouts can be asserted) and,
for Filling, re-implement the kernels in Python so that the flat index / strides logic is tested
end to end. Run from the repository root with the shim in sys.path (see run_tests.sh).
"""
import sys
import warnings

import numpy as np
import pytest

import stubenv
from stubenv import CALLS

CODE = {name: stubenv.stub(name) for name in (
    'ClearMap.ParallelProcessing.DataProcessing.ArrayProcessingCode',
    'ClearMap.ParallelProcessing.DataProcessing.MeasurePointListCode',
    'ClearMap.ParallelProcessing.DataProcessing.devolve_point_list_code',
    'ClearMap.ParallelProcessing.DataProcessing.ConvolvePointListCode',
    'ClearMap.ImageProcessing.Clipping.ClippingCode',
    'ClearMap.ImageProcessing.Binary.FillingCode',
    'ClearMap.ImageProcessing.Differentiation.HessianCode',
    'ClearMap.ImageProcessing.Tracing.TraceCode',
    'ClearMap.ImageProcessing.Thresholding.ThresholdingCode',
)}

# --- Python re-implementation of the FillingCode kernels (same flat index semantics) ---------------
_fill = CODE['ClearMap.ImageProcessing.Binary.FillingCode']
NOT_CHECKED, FILLED = -1, 1  # values used by FillingCode.pyx (EMPTY = 0 unused here)


def _prepare_temp(source, temp, processes):
    temp[:] = np.where(source != 0, FILLED, NOT_CHECKED)


def _label_temp(temp, strides, seeds, processes):
    size = temp.shape[0]
    for seed in seeds:
        queue = [seed]
        while queue:
            i = queue.pop()
            if temp[i] == NOT_CHECKED:
                temp[i] = FILLED
                for st in strides:
                    for j in (i - st, i + st):
                        if 0 <= j < size and temp[j] == NOT_CHECKED:
                            queue.append(j)


def _fill_kernel(source, temp, sink, processes, verbose=False):
    sink[:] = (source != 0) | (temp == NOT_CHECKED)


_fill.prepare_temp, _fill.label_temp, _fill.fill = _prepare_temp, _label_temp, _fill_kernel

# block sums / where for ArrayProcessing.where (count > 0, as the Cython code)
_apc = CODE['ClearMap.ParallelProcessing.DataProcessing.ArrayProcessingCode']
_apc.block_sums_1d = lambda source, blocks, processes: np.array([np.sum(source > 0)], dtype=np.intp)


def _where_1d(source, where, sums, blocks, processes):
    CALLS.append(('where_1d', (source, where, sums), {}))
    where[:] = np.flatnonzero(source > 0)


_apc.where_1d = _where_1d

import ClearMap.ParallelProcessing.DataProcessing.ArrayProcessing as ap
import ClearMap.ImageProcessing.Clipping.Clipping as clp
import ClearMap.ImageProcessing.Binary.Filling as bf
import ClearMap.ImageProcessing.Differentiation.Hessian as hes
import ClearMap.ImageProcessing.Tracing.Trace as trc
import ClearMap.ImageProcessing.Thresholding.Thresholding as th
import ClearMap.ParallelProcessing.DataProcessing.MeasurePointList as mpl
import ClearMap.ParallelProcessing.DataProcessing.DevolvePointList as dpl


def last(name):
    for n, a, k in reversed(CALLS):
        if n == name:
            return a, k
    raise AssertionError(f'{name} was not called')


# --- ArrayProcessing ---------------------------------------------------------------------------------
def test_initialize_sink_bool_is_viewed_as_uint8():
    sink = np.zeros((3, 4), dtype=bool)
    _, buffer = ap.initialize_sink(sink=sink)
    assert buffer.dtype == np.uint8 and np.shares_memory(buffer, sink)


def test_initialize_sink_refuses_non_contiguous_1d():
    base = np.zeros((4, 6), dtype=np.uint16)
    with pytest.raises(ValueError, match='contiguous'):
        ap.initialize_sink(sink=base[:, :3], as_1d=True)  # rows of 3 out of 6: no uniform flat stride


def test_initialize_source_shape_strides_are_intp():
    _, _, shape, strides = ap.initialize_source(np.zeros((2, 3), dtype=np.uint8), return_shape=True, return_strides=True)
    assert shape.dtype == np.intp and strides.dtype == np.intp


def test_where_unsupported_dtype_matches_kernel_semantics():
    src = np.array([0, 3, -2, 0, 5], dtype=np.int16)  # int16 is not compiled
    res = ap.where(src, cutoff=0)
    a, _ = last('where_1d')
    assert a[0].dtype == np.uint8
    assert list(np.asarray(res.array if hasattr(res, 'array') else res)) == [1, 4]  # entries > 0


def test_apply_lut_checks():
    src = np.array([0, 1, 2], dtype=np.uint8)
    ap.apply_lut(src, np.array([5, 6, 7], dtype=np.uint16))
    a, _ = last('apply_lut')
    assert a[1].dtype == a[2].dtype == np.uint16
    with pytest.raises(TypeError, match='source'):
        ap.apply_lut(src.astype(np.float32), np.array([5, 6, 7], dtype=np.uint16))
    with pytest.raises(Exception):  # rejected by initialize_sink (IncompatibleSource) before the explicit check
        ap.apply_lut(src, np.array([5, 6, 7], dtype=np.uint16), sink=np.zeros(3, dtype=np.float64))


def test_neighbours_indices_coerced():
    ap.neighbours(np.array([1, 2, 3], dtype=np.int32), offset=1)
    a, _ = last('neighbours')
    assert a[0].dtype == np.intp


# --- Clipping ----------------------------------------------------------------------------------------
def test_clip_defaults_and_checks():
    src = np.zeros((2, 3, 4), dtype=np.uint16)
    clp.clip(src)  # used ap.io.min_value, which does not exist
    a, _ = last('clip')
    assert (a[2], a[3]) == (0, 65535)
    with pytest.raises(ValueError, match='larger'):
        clp.clip(src, clip_min=3, clip_max=3)
    with pytest.raises(TypeError, match='source'):
        clp.clip(src.astype(np.int16))


# --- Filling (kernels re-implemented above) --------------------------------------------------------
def _hollow_cube():
    a = np.zeros((7, 8, 9), dtype=bool)
    a[1:6, 1:7, 1:8] = True
    a[2:5, 2:6, 2:7] = False  # hole
    expected = np.zeros_like(a)
    expected[1:6, 1:7, 1:8] = True
    return a, expected


@pytest.mark.parametrize('layout', ['C', 'F', 'view', 'int'])
def test_fill_layouts(layout):
    a, expected = _hollow_cube()
    if layout == 'F':
        a = np.asfortranarray(a)
    elif layout == 'view':
        big = np.zeros((7, 8, 18), dtype=bool)
        big[:, :, ::2] = a
        a = big[:, :, ::2]  # non contiguous
    elif layout == 'int':
        a = a.astype(np.int32) * 7
    result = bf.fill(a)
    assert result.dtype == bool and np.array_equal(result, expected)


def test_fill_sink_checks():
    a, expected = _hollow_cube()
    sink = np.zeros(a.shape, dtype=bool)
    assert bf.fill(a, sink=sink) is sink and np.array_equal(sink, expected)
    with pytest.raises(ValueError, match='contiguous'):
        bf.fill(a, sink=np.zeros(a.shape, dtype=bool, order='F'))
    with pytest.raises(TypeError, match='sink'):
        bf.fill(a, sink=np.zeros(a.shape, dtype=np.int32))


def test_fill_all_foreground_border():
    a = np.ones((4, 4, 4), dtype=bool)  # no background on the border: border_indices used to raise
    assert bf.fill(a).all()


# --- Hessian -----------------------------------------------------------------------------------------
def test_hessian_parameter_never_empty_and_sink_checked():
    src = np.zeros((3, 4, 5), dtype=np.uint16)
    hes.tubeness(src)
    _, k = last('tubeness')
    assert k['parameter'].size >= 1 and k['parameter'].flags.writeable and k['source'].dtype == np.float64
    with pytest.raises(TypeError, match='sink'):
        hes.tubeness(src, sink=np.zeros(src.shape, dtype=np.int32))


# --- Trace -------------------------------------------------------------------------------------------
def test_trace_layout_and_names():
    src = np.zeros((4, 5, 6))
    score = np.asfortranarray(np.ones((4, 5, 6), dtype=np.float32))
    mask = np.asfortranarray(np.zeros((4, 5, 6), dtype=bool))
    trc.trace_to_mask(src, score, (1, 2, 3), mask)  # was calling code.traceToMask
    a, _ = last('trace_to_mask')
    assert all(x.flags.c_contiguous for x in (a[0], a[1], a[3]))
    assert a[1].dtype == np.float64 and a[3].dtype == np.uint8 and a[2].dtype == np.intp
    with pytest.raises(ValueError, match='inside'):
        trc.trace(src, src, (1, 2, 3), (4, 0, 0))
    with pytest.raises(ValueError, match='score'):
        trc.trace(src, np.zeros((4, 5, 7)), (1, 2, 3), (0, 0, 0))


# --- Thresholding ------------------------------------------------------------------------------------
def test_threshold_background_is_converted_not_reinterpreted():
    src = np.random.default_rng(0).random((4, 5, 6))
    bg = np.zeros(src.shape, dtype=np.int64)
    bg[0] = 3
    th.threshold(src, threshold=0.9, hysteresis_threshold=0.5, background=bg)
    a, _ = last('threshold_to_background')
    assert a[2].dtype == np.uint8 and a[2].size == src.size and a[2].sum() == 30


# --- MeasurePointList --------------------------------------------------------------------------------
def test_measure_checks():
    src = np.zeros((10, 11, 12), dtype=np.uint16)
    pts = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
    search = np.array([[0, 0, 0], [1, 0, 0]], dtype=np.int32)
    mpl.measure_mean(src, pts, search, [2, 1])  # was broken: initialize_source tuple used as buffer
    a, _ = last('measure_mean')
    assert a[4].dtype == np.intp and a[5].dtype == np.intp and a[6].dtype == np.float64
    mpl.find_smaller_than_values(src, pts, search, [1.0, 2.0])  # called find_smaller_than_fraction
    a, _ = last('find_smaller_than_values')
    assert a[5].dtype == np.float64 and a[6].dtype == np.intp and a[6].shape == (2,)
    with pytest.raises(ValueError, match='coordinates'):
        mpl.measure_max(src, np.array([5, 6]), search, [1, 1])  # linear indices on a 3d source
    with pytest.raises(ValueError, match='outside'):
        mpl.measure_max(src, np.array([[1, 2, 12]]), search, [1])
    with pytest.raises(ValueError, match='max_search'):
        mpl.measure_max(src, pts, search, [3, 1])


# --- DevolvePointList --------------------------------------------------------------------------------
def test_devolve_checks_and_name():
    pts = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float64)
    idx = np.array([[0, 0, 0], [1, 0, 0]], dtype=np.int32)
    dpl.devolve(pts, shape=(8, 8, 8), indices=idx, weights=[1.0, 2.0], kernel=[1.0, 0.5])
    a, _ = last('devolve_weights_kernel')
    assert a[2].dtype == np.intp and a[3].dtype == np.float64
    with pytest.raises(ValueError, match='weights'):
        dpl.devolve(pts, shape=(8, 8, 8), indices=idx, weights=[1.0])
    with pytest.raises(ValueError, match='points'):
        dpl.devolve(pts, shape=(8, 8), indices=idx[:, :2])


def test_threshold_flat_arrays_share_the_source_order():
    src = np.random.default_rng(1).random((4, 5, 6))  # C order
    bg = np.asfortranarray(np.random.default_rng(2).random(src.shape) > 0.5)
    seeds = np.asfortranarray(src > 0.8)
    th.threshold(src, seeds=seeds, hysteresis_threshold=0.5, background=bg)
    a, _ = last('threshold_to_background')
    assert np.array_equal(a[2], bg.reshape(-1, order='C').view(np.uint8))   # was flattened in F order
    assert np.array_equal(a[4], np.flatnonzero(seeds.reshape(-1, order='C')))
    assert a[1].dtype == np.int8 and a[4].dtype == np.intp
    with pytest.raises(ValueError, match='contiguous'):
        th.threshold(src, threshold=0.5, sink=np.zeros(src.shape, dtype=np.int8, order='F'))
