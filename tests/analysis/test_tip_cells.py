"""The graph side of the classification of the degree 1 vertices (no model, no torch)"""
import numpy as np
import pandas as pd
import pytest

from ClearMap.Analysis.graphs.tip_cells import (add_tip_cell_properties, NOT_SCORED,
                                                TIP_CELL_SCORE_PROPERTY_NAME, TIP_CELL_PREDICTION_PROPERTY_NAME)

DEGREES = np.array([1, 3, 1, 1])


class FakeGraph:
    """What the functions use of ClearMap.Analysis.graphs.graph_gt.Graph (which needs graph_tool)"""
    def __init__(self):
        self.properties = {}

    def vertex_degrees(self):
        return DEGREES

    def define_vertex_property(self, name, source):  # Adds or replaces, like the Graph
        self.properties[name] = np.asarray(source)


def _scores(**score_by_vertex) -> pd.DataFrame:
    ids = [int(name.removeprefix('v')) for name in score_by_vertex]
    return pd.DataFrame({'vertex_id': ids, TIP_CELL_SCORE_PROPERTY_NAME: list(score_by_vertex.values())})


def test_scores_land_on_their_vertex_and_the_others_are_not_scored():
    graph = FakeGraph()

    add_tip_cell_properties(graph, _scores(v2=0.9, v0=0.1), threshold=0.5)

    assert graph.properties[TIP_CELL_SCORE_PROPERTY_NAME].tolist() == pytest.approx([0.1, NOT_SCORED, 0.9, NOT_SCORED])
    assert graph.properties[TIP_CELL_PREDICTION_PROPERTY_NAME].tolist() == [0, 0, 1, 0]


def test_a_score_at_the_threshold_is_not_a_tip_cell():
    graph = FakeGraph()

    add_tip_cell_properties(graph, _scores(v0=0.5, v2=0.51), threshold=0.5)

    assert graph.properties[TIP_CELL_PREDICTION_PROPERTY_NAME].tolist() == [0, 0, 1, 0]


def test_running_again_replaces_the_properties():
    graph = FakeGraph()
    add_tip_cell_properties(graph, _scores(v0=0.9), threshold=0.5)

    add_tip_cell_properties(graph, _scores(v3=0.9), threshold=0.5)

    assert graph.properties[TIP_CELL_PREDICTION_PROPERTY_NAME].tolist() == [0, 0, 0, 1]


def test_no_classified_vertex_leaves_all_not_scored():
    """E.g. all the degree 1 vertices are too close to the border of the image: the scores table is empty"""
    graph = FakeGraph()

    add_tip_cell_properties(graph, pd.DataFrame({'vertex_id': [], TIP_CELL_SCORE_PROPERTY_NAME: []}), threshold=0.5)

    assert graph.properties[TIP_CELL_SCORE_PROPERTY_NAME].tolist() == [NOT_SCORED] * 4
    assert graph.properties[TIP_CELL_PREDICTION_PROPERTY_NAME].tolist() == [0] * 4
