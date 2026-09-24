"""Tests for GraphSource and the graph-tool backend."""
import os
import pickle
import warnings

import numpy as np
import pytest

pytest.importorskip('graph_tool')

from ClearMap.Analysis.graphs import graph_gt
from ClearMap.IO import io_ops
from ClearMap.IO.source.backends import gt_backend
from ClearMap.IO.source.backends.gt_backend import GraphGtSource
from ClearMap.Utils.exceptions import ClearMapValueError, ClearMapNotImplementedError, SourceNotFoundError


def _graph(n=10, shape=(10, 20, 30)):
    graph = graph_gt.Graph(n_vertices=n)
    graph.shape = shape
    return graph


@pytest.fixture
def path(tmp_path):
    location = str(tmp_path / 'graph.gt')
    gt_backend.write(location, _graph())
    return location


def test_round_trip(tmp_path):
    location = str(tmp_path / 'g.gt')
    assert io_ops.write(location, _graph(7)) == location
    graph = io_ops.read(location)
    assert graph.n_vertices == 7 and tuple(graph.shape) == (10, 20, 30)


def test_open_ro_and_repr_do_not_load(path):
    source = io_ops.open_ro(path)
    str(source)
    assert not source.is_loaded
    assert source.n_vertices == 10 and source.is_loaded


def test_read_hands_over_the_cached_graph(path):
    source = io_ops.open_ro(path)
    cached = source.graph
    assert source.read() is cached and not source.is_loaded  # no copy, no second load
    assert io_ops.read(source).n_vertices == 10              # reloads from disk


def test_partial_load_is_forwarded(path):
    graph = io_ops.read(path, exclude_edge_geometry_properties=True)
    assert graph.n_vertices == 10


def test_source_pickles_as_a_handle(path):
    source = io_ops.open_ro(path)
    source.graph  # load
    clone = pickle.loads(pickle.dumps(source))
    assert not clone.is_loaded and source.is_loaded
    assert source.as_virtual() is source and clone.n_vertices == 10
    with pytest.raises(ClearMapValueError):
        GraphGtSource(graph=_graph()).as_virtual()  # nothing to reload from


def test_write_is_atomic(path):
    original = os.path.getsize(path)

    def failing_dump(graph, location, **kwargs):
        with open(location, 'wb') as f:
            f.write(b'partial')
        raise RuntimeError('disk full')

    real_dump = GraphGtSource._dump
    GraphGtSource._dump = classmethod(lambda cls, graph, location, **kw: failing_dump(graph, location, **kw))
    try:
        with pytest.raises(RuntimeError):
            gt_backend.write(path, _graph(3))
    finally:
        GraphGtSource._dump = real_dump
    assert os.path.getsize(path) == original and io_ops.read(path).n_vertices == 10
    assert not [f for f in os.listdir(os.path.dirname(path)) if f.startswith('.tmp-')]


def test_write_through_source(path):
    source = io_ops.open_ro(path)
    source.graph
    source.write(_graph(4))
    assert not source.is_loaded and source.n_vertices == 4


def test_legacy_forms(tmp_path, path):
    source = GraphGtSource(_graph(5))  # graph passed as first argument
    assert source.location is None and source.n_vertices == 5
    target = str(tmp_path / 'legacy.gt')
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        source.write(target)  # old write(location)
        opened = gt_backend.read(path, as_source=True)
    assert io_ops.read(target).n_vertices == 5 and isinstance(opened, GraphGtSource)
    assert sum(issubclass(w.category, DeprecationWarning) for w in caught) == 2


def test_refusals(tmp_path, path):
    source = io_ops.open_ro(path)
    with pytest.raises(ClearMapNotImplementedError):
        io_ops.read(path, slicing=slice(0, 2))
    with pytest.raises(ClearMapNotImplementedError):
        gt_backend.write(path, _graph(), slicing=slice(0, 2))
    with pytest.raises(ClearMapNotImplementedError):
        gt_backend.edit(path)
    with pytest.raises(ClearMapNotImplementedError):
        source['x'] = 1
    with pytest.raises(ClearMapValueError):
        io_ops.write(path, np.zeros(3))  # not a graph
    with pytest.raises(ClearMapNotImplementedError):
        gt_backend.create(str(tmp_path / 'blank.gt'))
    with pytest.raises(FileExistsError):
        gt_backend.write(path, _graph(), overwrite=False)
    with pytest.raises(SourceNotFoundError):
        io_ops.read(str(tmp_path / 'missing.gt'))
