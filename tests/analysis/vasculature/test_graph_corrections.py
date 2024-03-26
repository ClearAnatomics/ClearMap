import graph_tool.topology
import pytest

import numpy as np

from ClearMap.Analysis.Graphs import GraphGt
from ClearMap.Analysis.vasculature.graph_corrections import remove_spurious_branches, join_neighbouring_degrees_1
from ClearMap.Analysis.vasculature.graph_corrections import remove_auto_loops


def create_graph(edges, vertices=None):
    n_vertices = len(np.unique(np.array(list(edges.keys()))))
    graph = GraphGt.Graph(n_vertices=n_vertices, edges=list(edges.keys()))
    graph.add_edge_property('length', np.array(list(edges.values())))
    if vertices is not None:
        graph.add_vertex_property('coordinates', np.array(list(vertices.values()), dtype=float))
    return graph


def assert_graphs_equal(expected_result, filtered_graph):
    assert graph_tool.topology.isomorphism(filtered_graph._base, expected_result._base)
    assert graph_tool.topology.similarity(filtered_graph._base, expected_result._base) == 1.0


@pytest.fixture()
def test_graph():
    edges = {
        (0, 1): 1,
        (1, 2): 12,
        (1, 5): 7,
        (2, 3): 8,
        (2, 6): 3,
        (2, 7): 4,
        (3, 4): 1,
        (4, 5): 2,
        (4, 8): 10,
        (5, 5): 2,
        (8, 4): 8,
        (8, 9): 6,
        (8, 11): 2,
        (9, 9): 6,
        (9, 10): 2
     }
    vertices = {
        0: (0, 3, 0),
        1: (4, 3, 0),
        2: (5, 5, 0),
        3: (9, 5, 0),
        4: (10, 3, 0),
        5: (7, 0, 0),
        6: (4, 6, 0),
        7: (5, 6, 0),
        8: (13, 3, 0),
        9: (15, 1, 0),
        10: (16, 2, 0),
        11: (16, 4, 0),
    }
    return create_graph(edges, vertices)


def test_remove_spurious_branches(test_graph):
    expected_result_edges = {
        (1, 2): 12,
        (1, 5): 7,
        (2, 3): 8,
        (3, 4): 1,
        (4, 5): 2,
        (4, 8): 10,
        (8, 4): 8,
        (5, 5): 2,
        (8, 9): 6,
        (9, 9): 6
        }

    expected_result = create_graph(expected_result_edges)
    cleaned = remove_spurious_branches(test_graph, min_length=5)
    print('Result')
    print(cleaned.edge_connectivity())
    # assert_graphs_equal(expected_result, cleaned)
    assert np.all(expected_result.edge_connectivity() == cleaned.edge_connectivity())


def test_remove_auto_loops(test_graph):
    expected_result_edges = {
         (0, 1): 1,
         (1, 2): 12,
         (1, 5): 7,
         (2, 3): 8,
         (2, 6): 3,
         (2, 7): 4,
         (3, 4): 1,
         (4, 5): 2,
         (4, 8): 10,
         (8, 4): 8,
         (8, 9): 6,
         (8, 11): 2,
         (9, 9): 6,
         (9, 10): 2
        }
    expected_result = create_graph(expected_result_edges)
    result = remove_auto_loops(test_graph, min_length=5)
    print()
    print('Result')
    print(result.edge_connectivity())
    print("Expected_result")
    print(expected_result.edge_connectivity())
    assert_graphs_equal(expected_result, result)


class Edge:
    def __init__(self, v1, v2, l):
        self.v1 = v1
        self.v2 = v2
        self.l = l

    @property
    def connectivity(self):
        return self.v1, self.v2

    @property
    def length(self):
        return self.l

def test_join_degree_1_neighbours(test_graph):
    expected_result_edges = {
        (0, 1): 1,
        (1, 2): 12,
        (1, 5): 7,
        (2, 3): 8,
        (2, 6): 3,
        (2, 7): 4,
        (3, 4): 1,
        (4, 5): 2,
        (4, 8): 10,
        (8, 4): 8,
        (5, 5): 2,
        (8, 9): 6,
        (8, 11): 2,
        (9, 9): 6,
        (9, 10): 2,
        (6, 7): 1,
        (10, 11): 2,
        }
    vertices = {
        0: (0, 3, 0),
        1: (4, 3, 0),
        2: (5, 5, 0),
        3: (9, 5, 0),
        4: (10, 3, 0),
        5: (7, 0, 0),
        6: (4, 6, 0),
        7: (5, 6, 0),
        8: (13, 3, 0),
        9: (15, 1, 0),
        10: (16, 2, 0),
        11: (16, 4, 0),
    }
    expected_result = create_graph(expected_result_edges, vertices=vertices)
    print(test_graph)
    assert_graphs_equal(expected_result, join_neighbouring_degrees_1(test_graph, min_radius=2.5))
