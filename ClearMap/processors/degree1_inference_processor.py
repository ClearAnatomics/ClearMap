"""
This module provides functionality to run inference on the nature of degree=1 vertices in a vascular graph,
detecting tip cells and interrupted vessels.

The main user function is `degree1_verification`, which performs inference on the graph vertices
and updates the graph in place with predictions based on a pre-trained model.
"""

# FIXME: this is not a processor. It does not follow the processor API. Convert or move to another location.

import numpy as np
import pandas as pd

from ClearMap.Analysis.graphs import graph_gt
from ClearMap.Analysis.graphs.graph_gt import Graph
from ClearMap.ImageProcessing.machine_learning.vertices_classification.degree1_inference_utils import (run_inference, update_graph_properties,
                                                                                                       get_default_model_path)
from ClearMap.processors.tube_map import BinaryVesselProcessor


def export_vertex_coordinates(graph: Graph) -> pd.DataFrame:
    """
    Extract coordinates and degrees of all vertices from a graph into a DataFrame.

    Parameters
    ----------

    graph : ClearMap.Analysis.graphs.graph_gt.Graph
        The graph from which to extract vertex properties.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing vertex IDs, coordinates (x, y, z), and degrees.
    """

    coords = np.array(graph.vertex_property("coordinates"))

    # vertex_ids = np.array([int(v) for v in graph.vertices])

    df = pd.DataFrame({
        "vertex_id": graph.vertex_indices(),
        "x": coords[:, 0],
        "y": coords[:, 1],
        "z": coords[:, 2],
        "degree": np.array(graph.vertex_degrees())
    })

    return df


def degree1_verification(sample_manager, model_weights_path=None, threshold=0.8, stitched_img_channel='vasc',
                         graph_channel=('vasc'), graph_step=None):
    """
    Perform tip cells / interrupted vessels detection and update the graph with predictions.
    Called externally from the pipeline (e.g. tube_map_new_api.py) with no additional logic inside.

    Parameters
    ----------

    sample_manager : ClearMap.processors.sample_preparation.SampleManager
        The SampleManager instance managing the sample data.
    model_weights_path : str | Path | None
        Path to the model weights file for inference. Default is None, which uses the default model.
    threshold : float
        Threshold for tip cell prediction. Default is 0.8.
    stitched_img_channel : str
        The channel of the stitched image to use for inference.
        This is the channel which contains the grayscale of the tip cells.
        Default is 'vasc'.
    graph_channel: str or tuple or list
        The channel(s) of the graph.
        This can be a single channel or a tuple/list of channels.
        Default is ('vasc',).
    graph_step: str or None
        The step in the graph processing pipeline to use for inference.
        Typically, one of ['raw', 'cleaned', 'reduced', 'annotated']
        If None, the default graph step will be used.
    """
    print("[INFO] Degree=1 Inference Processor launched...")

    if model_weights_path is None:
        model_weights_path = get_default_model_path()

    if not isinstance(graph_channel, tuple):
        if isinstance(graph_channel, list):
            graph_channel = tuple(graph_channel)
        else:
            graph_channel = (graph_channel,)

    # Force the workspace to recognise the compound channel of the graph
    _ = BinaryVesselProcessor(sample_manager=sample_manager)

    stitched_asset = sample_manager.get('stitched', channel=stitched_img_channel)
    graph_asset = sample_manager.get('graph', channel=graph_channel, asset_sub_type=graph_step)

    if not stitched_asset.exists or not graph_asset.exists:
        print(f"[WARNING] Required files missing: {stitched_asset.path} or {graph_asset.path}")
        return

    # Load image and graph
    image = stitched_asset.as_source(mode='r')
    # image = np.load(stitched_path, mmap_mode="r").swapaxes(0, 2)
    graph = graph_asset.read()

    # Prepare vertex coordinates (DataFrame)
    coords_df = export_vertex_coordinates(graph)

    # Run model inference
    scores_df = run_inference(
        image=image,
        vertices_df=coords_df,
        model_weights=model_weights_path,
        patch_shape=(30, 30, 30),
        batch_size=16
    )

    # Update graph and save
    updated_graph_path = sample_manager.get_path('graph', channel=graph_channel, asset_sub_type='updated')
    update_graph_properties(graph=graph, scores_df=scores_df, output_graph_path=updated_graph_path, threshold=threshold)

    print("Degree=1 Inference Processor completed.")