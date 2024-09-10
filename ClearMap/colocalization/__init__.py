"""
Colocalization
----------------

This module allows to compare signal between different channels

The basic idea is to exploit the data of 
- a binary mask per channel
- a dataframe with a representative point per connected component, given in pixel coords
    with possibly extra measurment information for the nucleus in the respective channel,
    eg some intensity measurment.

The idea is to be able to determine if two  detected nuclei in distinct channels
correspond indeed to the same nucleus.

We have to rely on well aligned channel images for this sake.

The simplest method is to break symmetry and have a reference channel that
marks all nuclei of the studied cell catagory (e.g. all cell type or neurons)

Then, for each reference channel detected nucleus, decide if it is to be considered 
positive for the the other channels.

This can be made relying on the following capabilities, given two channels
- Compute the overlap of each connected component of a given channel with the 
other channel, componentwise or globally or, more generally
- compute a matching score based on the information we have

Ideally, all of this is to be made efficiently both with quick sequential code
and parallelization techniques.
"""

from IO.Source import Source
import pandas as pd


class Channel:
    """ In out context a channel is represented by a binary mask and some voxel coordinates in a dataframe.
    The voxel coordinates should be a set of representatives for the partition of the True voxels in the binary mask
    in connected components for the euclidean radius 1 closed ball footprint. We admit the situation
    where not all connected components have a representative but the points must belong to distinct components.
    """
    def __init__(self, binary_img: Source, dataframe: pd.DataFrame, coord_names=['x','y','z']) -> None:
        self.binary_img = binary_img
        self.dataframe = dataframe
        self.coord_names = coord_names
    @property
    def representative_points(self):
        return self.dataframe[self.coord_names]
    
    def labels(self):
        """Return the labeled image from the representative points and the binary mask
        """


