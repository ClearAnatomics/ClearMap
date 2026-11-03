"""
This module provides functionality to run inference on 3D patches extracted from a 3D image,

The main user functions are `run_inference`, which extracts patches centered on degree=1 graph vertices,
and `update_graph_properties`, which updates the graph with new properties based on the inference results.
"""
from pathlib import Path

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as Fnn
from torch.utils.data import Dataset, DataLoader
from torchvision.models.video import swin3d_s


MAX_WORKERS_LOADER = 8


def get_default_model_path(weights_file_name="swin3d_s_weights.pth"):
    """
    Get the default path to the pre-trained Swin3D model weights.

    Parameters
    ----------
    weights_file_name : str
        Name of the weights file. Default is "swin3d_s_weights.pth".

    Returns
    -------
    str
        Path to the pre-trained model weights.
    """
    return Path(__file__).parent / "weights" / weights_file_name


class Swin3DModel(nn.Module):
    """
    Swin3D-based binary classifier for degree=1 tip cells and interrupted vessels.
    """
    def __init__(self, num_classes=2):
        super().__init__()
        self.model = swin3d_s(weights=None)
        in_features = self.model.head.in_features
        self.model.head = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        return self.model(x)


def load_model(weights_path):
    """
    Load the Swin3D model with pre-trained weights.

    Parameters
    ----------
    weights_path: str or Path
        Path to the model weights file.

    Returns
    -------
    Swin3DModel
        An instance of the Swin3DModel with loaded weights.
    """
    model = Swin3DModel(num_classes=2).cuda()
    state_dict = torch.load(weights_path, map_location="cuda", weights_only=True)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


class PatchDataset(Dataset):
    """
    Dataset for extracting 3D patches centered on degree=1 graph vertices.
    """
    def __init__(self, vertices_df, image, shape=(30, 30, 30)):
        self.image = image
        self.dimensions = np.array(shape)
        self.half_dim = self.dimensions // 2
        self.image_shape = np.array(image.shape)

        self.vertices_df = vertices_df.copy() # deepcopy not needed ?
        self.vertices_df = self.vertices_df[self.vertices_df["degree"] == 1].reset_index(drop=True)
        self.filter_valid_vertices()

    def __len__(self):
        return len(self.vertices_df)

    def __getitem__(self, idx):
        vertex = self.vertices_df.iloc[idx]
        vertex_id = int(vertex["vertex_id"])
        x, y, z = map(int, (vertex["x"], vertex["y"], vertex["z"]))
        x_min, x_max = x - self.half_dim[0], x + self.half_dim[0]
        y_min, y_max = y - self.half_dim[1], y + self.half_dim[1]
        z_min, z_max = z - self.half_dim[2], z + self.half_dim[2]
        patch = self.image[x_min:x_max, y_min:y_max, z_min:z_max]
        patch = torch.tensor(patch, dtype=torch.float32).unsqueeze(0).repeat(3, 1, 1, 1)
        return vertex_id, patch

    def filter_valid_vertices(self):
        valid_mask = (
            (self.vertices_df["x"] - self.half_dim[0] >= 0) & (self.vertices_df["x"] + self.half_dim[0] < self.image_shape[0]) &
            (self.vertices_df["y"] - self.half_dim[1] >= 0) & (self.vertices_df["y"] + self.half_dim[1] < self.image_shape[1]) &
            (self.vertices_df["z"] - self.half_dim[2] >= 0) & (self.vertices_df["z"] + self.half_dim[2] < self.image_shape[2])
        )
        self.vertices_df = self.vertices_df[valid_mask].reset_index(drop=True)


def run_inference(image, vertices_df, model_weights, patch_shape=(30, 30, 30), batch_size=16):
    """
    Run inference on patches centered on degree=1 graph vertices.

    Parameters
    ----------
    image : np.ndarray
        The 3D image from which to extract patches.
    vertices_df : pd.DataFrame
        DataFrame containing vertex id, coordinates and degrees.
    model_weights : str or Path
        Path to the model weights file.
    patch_shape : tuple
        Shape of the patches to extract around each vertex.
    batch_size : int
        Batch size for inference.
    """
    model = load_model(model_weights)
    dataset = PatchDataset(vertices_df, image, shape=patch_shape)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=MAX_WORKERS_LOADER)

    all_ids, all_scores = [], []
    with torch.no_grad():
        for ids, patches in dataloader:
            patches = patches.cuda()
            probs = Fnn.softmax(model(patches), dim=1)[:, 1]
            all_ids.extend(ids.cpu().numpy())
            all_scores.extend(probs.cpu().numpy())

    return pd.DataFrame({"vertex_id": all_ids, "tip_cell_score": all_scores})


def update_graph_properties(graph, scores_df, output_graph_path, threshold=0.8):
    """
    Updating the graph with two new properties: tip_cell_score and tip_cell_prediction

    Parameters
    ----------
    graph : ClearMap.Analysis.Graphs.GraphGt
        The graph to update with new properties.
    scores_df : pd.DataFrame
        DataFrame containing vertex IDs and their corresponding tip cell scores.  # FIXME: check other columns
    output_graph_path: str or Path
        Path to save the updated graph with new properties.
    threshold : float
        Threshold for classifying a vertex as a tip cell. Default is 0.8.
    """
    vertex_degrees = np.array(graph.vertex_degrees())
    total_vertices = len(vertex_degrees)

    mapped_scores = np.full(total_vertices, -1.0, dtype=np.float32)
    predictions = np.zeros(total_vertices, dtype=np.int32)

    scores_df = scores_df.set_index("vertex_id")
    valid_ids = scores_df.index.values

    mapped_scores[valid_ids] = scores_df.tip_cell_score.values
    predictions[valid_ids] = (scores_df.tip_cell_score.values > threshold).astype(int)

    graph.add_vertex_property("tip_cell_score", mapped_scores)
    graph.add_vertex_property("tip_cell_prediction", predictions)
    graph.save(output_graph_path)
    print(f"Updated graph saved to: {output_graph_path}")