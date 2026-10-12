"""
tip_cells
=========

The graph side of the classification of the degree 1 vertices of a vascular graph: the properties that store
the score and prediction that the model gives back
(see :mod:`ClearMap.ImageProcessing.machine_learning.vertices_classification`).

A degree 1 vertex is either a *tip cell* (a vessel that grows or retracts) or the end of a bad tracing
(an interrupted vessel, to be reconnected later).

No heavy import here (torch...): the graph and the table are all this module needs.
"""
import numpy as np
import pandas as pd

TIP_CELL_SCORE_PROPERTY_NAME = 'tip_cell_score'
TIP_CELL_PREDICTION_PROPERTY_NAME = 'tip_cell_prediction'

NOT_SCORED = -1.0
"""The score of the vertices that were not classified (the ones that are not of degree 1 or too close to the image border)"""


def add_tip_cell_properties(graph, scores_df: pd.DataFrame, threshold: float) -> None:
    """
    Store the tip cell score and prediction of the classified vertices in the graph, in place.
    The properties are replaced if they exist (the classification can be run again).

    Parameters
    ----------
    graph : ClearMap.Analysis.graphs.graph_gt.Graph
        The graph to update.
    scores_df : pd.DataFrame
        The ``vertex_id`` and ``tip_cell_score`` of the classified vertices.
    threshold : float
        A vertex whose score is above this threshold is predicted to be a tip cell.

    Notes
    -----
    The vertices that are not in ``scores_df`` have the score :data:`NOT_SCORED` and the prediction 0.
    """
    n_vertices = len(graph.vertex_degrees())

    mapped_scores = np.full(n_vertices, NOT_SCORED, dtype=np.float32)
    predictions = np.zeros(n_vertices, dtype=np.int32)

    scores_df = scores_df.set_index('vertex_id')
    valid_ids = scores_df.index.to_numpy(dtype=np.int64)  # Not float if there is none (no vertex could be classified)

    mapped_scores[valid_ids] = scores_df[TIP_CELL_SCORE_PROPERTY_NAME].to_numpy()
    predictions[valid_ids] = (scores_df[TIP_CELL_SCORE_PROPERTY_NAME].to_numpy() > threshold).astype(np.int32)

    graph.define_vertex_property(TIP_CELL_SCORE_PROPERTY_NAME, mapped_scores)
    graph.define_vertex_property(TIP_CELL_PREDICTION_PROPERTY_NAME, predictions)
