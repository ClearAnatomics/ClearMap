"""
Graph.vertex_properties_table, run on a stand-in that has what it uses of the Graph (the real one needs graph_tool):
the method is called unbound.
"""
import numpy as np
import pytest

from ClearMap.Analysis.graphs.graph_gt import Graph

COORDINATES = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
ATLAS_COORDINATES = np.array([[10, 20, 30], [40, 50, 60], [70, 80, 90]])
RADII = np.array([0.5, 1.5, 2.5])
DEGREES = np.array([1, 3, 1])


class FakeGraph:
    def __init__(self):
        self.properties = {'coordinates': COORDINATES, 'coordinates_atlas': ATLAS_COORDINATES, 'radii': RADII,
                           'offsets': np.array([[1, 2], [3, 4], [5, 6]])}

    def vertex_property(self, name):
        return self.properties[name]

    def vertex_indices(self):
        return np.arange(len(COORDINATES))

    def vertex_degrees(self):
        return DEGREES


def _table(*args, **kwargs):
    return Graph.vertex_properties_table(FakeGraph(), *args, **kwargs)


def test_coordinates_are_one_column_per_axis_after_the_vertex_id():
    table = _table(['coordinates'])

    assert list(table.columns) == ['vertex_id', 'x', 'y', 'z']
    assert table[['x', 'y', 'z']].to_numpy().tolist() == COORDINATES.tolist()
    assert table['vertex_id'].tolist() == [0, 1, 2]


def test_other_coordinates_are_prefixed_with_their_name():
    assert list(_table(['coordinates_atlas']).columns) == ['vertex_id', 'atlas_x', 'atlas_y', 'atlas_z']


def test_scalar_properties_are_one_column_and_other_vectors_one_per_dimension():
    table = _table(['radii', 'offsets'])

    assert list(table.columns) == ['vertex_id', 'radii', 'offsets_dim0', 'offsets_dim1']
    assert table['radii'].tolist() == RADII.tolist()
    assert table['offsets_dim1'].tolist() == [2, 4, 6]


def test_degrees_are_the_last_column_when_asked():
    assert list(_table(['coordinates'], with_degrees=True).columns) == ['vertex_id', 'x', 'y', 'z', 'degree']
    assert _table(['coordinates'], with_degrees=True)['degree'].tolist() == DEGREES.tolist()
    assert 'degree' not in _table(['coordinates']).columns
