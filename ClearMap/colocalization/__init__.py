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

WARNING: The current version of this module deals only with 3d images.
"""

from __future__ import annotations
from functools import cached_property

import numpy as np
import pandas as pd
import scipy.ndimage as ndi


from ..IO.Source import Source


# batch distance computation
def distances(points_1: np.ndarray, points_2: np.ndarray) -> np.ndarray:
    """Compute the distances between the points in points_1 and those of points_2.

    Parameters
    ----------
    points_1 : np.ndarray
        array of points of shape (n_points_1,dim)
    points_2 : np.ndarray
        array of points of shape (n_points_2,dim)
    Returns
    -------
    np.ndarray
        distance matrix A: A[i,j] is the distance between the ith point in
        points_1 and the jth point in points_2.
    """
    return (
        np.sum(
            (points_1[:, np.newaxis, :] - points_2[np.newaxis, :, :]) ** 2,
            axis=-1,
        )
    ) ** 0.5


# a cool auxiliary function
def bilabel_bincount(labels_1: np.array, labels_2: np.array) -> np.array:
    """Count the number of occurences for all the conjunctions of labels.

    Parameters
    ----------
    labels_1 : np.array
        an array of labels
    labels_2 : np.array
        another array of labels, with the same shape as labels_1

    Returns
    -------
    np.array
        The array counts such that counts[i,j]==np.count_nonzero((labels_1==i)*(labels_2==j)).

    """

    # inspired by the join segmentation of skimage
    factor = labels_1.max() + 1
    max_val = labels_2.max() * factor + factor - 1

    # determine the cheapest dtype to hold the join
    dtypes = ["uint8", "uint16", "uint32", "uint64", "argh"]
    for i, dtype in enumerate(dtypes):
        if 2 ** (2 ** (3 + i)) > max_val:
            break
    if dtype == "argh":
        raise RuntimeError("This software is not able to store the join labels.")

    joined_labels = factor * labels_2.astype(dtype) + labels_1.astype(dtype)

    counts = np.bincount(
        joined_labels.flatten(), minlength=factor * (max_val // factor + 1)
    )

    # We return the counts recast in 2d
    return counts.reshape((factor, -1)).transpose()


class Channel:
    """In our context a channel is represented by a binary mask and some voxel coordinates in a dataframe.
    The voxel coordinates should be a set of representatives for the partition of the True voxels in the binary mask
    in connected components for the 1-connectivity (also termed as 6 connectivity in 3d). We admit the situation
    where not all connected components have a representative but the points must belong to distinct components.
    """

    def __init__(
        self, binary_img: Source, dataframe: pd.DataFrame, coord_names=["x", "y", "z"]
    ) -> None:
        self.binary_img = binary_img.array
        self.dataframe = dataframe
        self.coord_names = coord_names
        self.ndim = len(self.coord_names)

    @property
    def representative_points(self):
        return self.dataframe[self.coord_names]

    @cached_property
    def labels(self):
        """Return the labeled image from the binary mask"""
        labels, _ = ndi.label(self.binary_img)
        return labels

    @cached_property
    def index_label_correspondance(self):
        return [
            self.labels[tuple(self.dataframe[self.coord_names].iloc[index])]
            for index in range(len(self.dataframe))
        ]

    def nucleus(self, index):
        """Return the nucleus associated to a given index, as a mask.

        Parameters
        ----------
        index : int
        index for the chosen representative point in self.dataframe

        """
        return self.index_label_correspondance[index] == self.labels

    def bounding_box(self, i) -> tuple[slice, ...]:
        """Return the bounding box of the ith nucleus.

        Parameters
        ----------
        i : int
            the index of the considered nucleus.

        Returns
        -------
        tuple[slice,...]
            the tuple of slices (slice_0, slice_1, slice_2,...) that defines
            the bounding box
        """

        return tuple(
            slice(*[self._bounding_boxes_array[i, axis, j] for j in range(2)])
            for axis in self.ndim
        )

    @cached_property
    def _bounding_boxes_array(self):
        stack = []
        for index in range(len(self.dataframe)):
            nuke = self.nucleus(index)
            locus = np.where(nuke)
            stack += [[[locus[i].min(), locus[i].max()] for i in range(self.ndim)]]
        return np.array(stack)

    @cached_property
    def centers(self):
        starts = self._bounding_boxes_array[:, :, 0]
        stops = self._bounding_boxes_array[:, :, 1]
        return ((starts + stops) // 2).reshape(self._bounding_boxes_array.shape[:2])

    def center(self, i) -> tuple[int, ...]:
        """Return the center of the bounding box of the ith nucleus.

        Parameters
        ----------
        i : int
            the index of the considered nucleus.

        Returns
        -------
        tuple[int,...]
            the tuple of coords for the central voxel
        """
        return tuple(self.centers[i])

    @cached_property
    def sizes(self):
        """Return the list of the sizes of nuclei in the order of our dataframe.

        Returns
        -------
        list[int]
            The list of sizes of nuclei in the order of our dataframe.
        """
        return np.bincount(self.labels.flatten())[self.index_label_correspondance]

    def masked_sizes(self, mask):
        """Return the list of the sizes of the non-masked part of each nucleus

        Returns
        -------
        list[int]
            The list of counts of True pixels in mask for each nucleus, in the
            order of our dataframe.
        """
        return np.bincount((mask * self.labels).flatten(), minlength=self.sizes.size)[
            self.index_label_correspondance
        ]

    def overlap_rates(self, other_channel: Channel):
        """Return the rate of positive pixels for other_channel in each nucleus

        Parameters
        ----------
        other_channel : Channel
            The Channel to compare with self

        Returns
        -------
        np.array
            The array of rates, in the order of self.dataframe
        """

        return self.sizes / self.masked_sizes(other_channel.binary_img)

    def single_blob_overlap_rates(self, other_channel: Channel) -> np.ndarray:
        """For each nucleus of self, compute the max overlap rate with a single nucleus of other_channel.

        Parameters
        ----------
        other_channel : Channel
            channel to compare with self

        Returns
        -------
        np.ndarray
            the max overlap rate for each nucleus in the order of self.dataframe
        """
        counts = bilabel_bincount(self.labels, other_channel.labels)
        return np.max(counts, axis=1)[self.index_label_correspondance] / self.sizes

    def centers_distances(self, other_channel: Channel) -> np.array:
        """Return array of distances between centers across the two channels.

        Parameters
        ----------
        other_channel : Channel
            Channel to compare with self

        Returns
        -------
        np.array
            distance matrix A: A[index_1,index_2] is the distance between the centers
            of the nucleus indexed by index_1 in self.dataframe and  the nucleus
            indexed by index_2 in other_channel.dataframe.
        """

        return distances(self.centers, other_channel.centers)
