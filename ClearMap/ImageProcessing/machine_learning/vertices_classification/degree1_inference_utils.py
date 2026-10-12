"""
This module provides functionality to run inference on 3D patches extracted from a 3D image.

The main user function is `run_inference`, which extracts patches centered on degree=1 graph vertices.
The graph side (the vertices to classify, the properties to store the results) is in
:mod:`ClearMap.Analysis.graphs.tip_cells`.
"""
from pathlib import Path

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as Fnn
from torch.utils.data import Dataset, DataLoader
from torchvision.models.video import swin3d_s

from ClearMap.Analysis.graphs.tip_cells import TIP_CELL_SCORE_PROPERTY


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

    return pd.DataFrame({"vertex_id": all_ids, TIP_CELL_SCORE_PROPERTY: all_scores})
