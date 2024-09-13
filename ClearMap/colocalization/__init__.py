# Copyright Gaël Cousin & Charly Rousseau

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
            axis=2,
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
    max_1 = labels_1.max()
    max_2 = labels_2.max()
    factor = max_2 + 1
    # factor will also be the heigth of the output
    max_val = max_1 * factor + max_2
    # determine the cheapest dtype to hold the join
    dtypes = ["uint8", "uint16", "uint32", "uint64", "argh"]
    for i, dtype in enumerate(dtypes):
        if 2 ** (2 ** (3 + i)) > max_val:
            break
    if dtype == "argh":
        raise RuntimeError("This software is not able to store the join labels.")

    joined_labels = factor * labels_1.astype(dtype) + labels_2.astype(dtype)

    counts = np.bincount(joined_labels.flatten(), minlength=(max_1 + 1) * (max_2 + 1))

    # We return the counts recast in 2d
    return counts.reshape((max_1 + 1, max_2 + 1))


# for comparison/test purposes
def _naive_bilabel_bincount(A, B):
    m = A.max() + 1
    n = B.max() + 1
    res = np.zeros((m, n), dtype="uint8")
    for i in range(m):
        for j in range(n):
            res[i, j] = np.count_nonzero((A == i) * (B == j))
    return res


class Channel:
    """In our context a channel is represented by a binary mask and some voxel coordinates in a dataframe.
    The voxel coordinates should be a set of representatives for the partition of the True voxels in the binary mask
    in connected components for the 1-connectivity (also termed as 6 connectivity in 3d). We admit the situation
    where not all connected components have a representative but the points must belong to distinct components.
    """

    channel_count = 0
    channel_names = []

    def __init__(
        self,
        binary_img: np.ndarray,
        dataframe: pd.DataFrame,
        coord_names=["x", "y", "z"],
        channel_name="",
    ) -> None:
        self.binary_img = binary_img
        self.dataframe = dataframe
        self.coord_names = coord_names
        self.ndim = len(self.coord_names)

        # we identify the channel by a name fo caching management
        if not channel_name:
            channel_name = Channel.channel_count
            Channel.channel_count += 1
        if channel_name in Channel.channel_names:
            raise ValueError("The passed Channel name is already in use.")

        self.name = channel_name
        Channel.channel_names.append(channel_name)

        self._overlaps_dic = {}

    @property
    def representative_points(self):
        return self.dataframe[self.coord_names]

    @cached_property
    def labels(self):
        """Return the labeled image from the binary mask"""
        labels, _ = ndi.label(self.binary_img)
        return labels

    @cached_property
    def index_label_correspondance(self) -> np.array:
        """Return the labels of nuclei in the nuclei index order

        Returns
        -------
        np.array
            the flat array of the labels in the nuclei index order
        """
        return np.array(
            [
                self.labels[tuple(self.dataframe[self.coord_names].iloc[index])]
                for index in range(len(self.dataframe))
            ]
        )

    @cached_property
    def label_index_correspondance(self):
        # beware that not all label corresponds to an index,
        # if we do not have a complete set of reps in the dataframe.
        # We set arbitrarily the index value 0 for the labels that will not be used.
        max_label = self.index_label_correspondance.max()
        return np.array(
            [
                (
                    np.where(self.index_label_correspondance == label)[0][0]
                    if label in self.index_label_correspondance
                    else 0
                )
                for label in range(max_label + 1)
            ]
        )

    @cached_property
    def _shifted_index_label_correspondance(self):
        """Return the index_label_correspondance shifted by 1

        Returns
        -------
        list[int]
            the list L such that L[i]=index_label_correspondance[i]-1
        """
        return np.array(self.index_label_correspondance) - 1

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
        starts = self._bounding_boxes_array[:, :, 0].astype("float32")
        stops = self._bounding_boxes_array[:, :, 1].astype("float32")
        return ((starts + stops) / 2).reshape(self._bounding_boxes_array.shape[:2])

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
        return np.bincount(
            (mask * self.labels).flatten(), minlength=self.sizes.size + 1
        )[self.index_label_correspondance]

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

        return self.masked_sizes(other_channel.binary_img) / self.sizes

    def _set_blobwise_overlaps(self, other_channel: Channel):
        """
        Compute and store the matrix M of overlaps wher M[i,j] is the overlap between blob
        of index i for self and the blob of index j for other_channel.
        """
        counts = bilabel_bincount(self.labels, other_channel.labels)
        # we reorder the lines and cols from label orders to index orders
        col_selector = np.zeros(
            (counts.shape[1], len(other_channel.index_label_correspondance))
        )
        col_selector[
            other_channel.index_label_correspondance, np.arange(col_selector.shape[1])
        ] = 1

        line_selector = np.zeros(
            (len(self.index_label_correspondance), counts.shape[0])
        )
        line_selector[
            np.arange(line_selector.shape[0]), self.index_label_correspondance
        ] = 1

        self._overlaps_dic[other_channel.name] = line_selector @ counts @ col_selector

    def blobwise_overlaps(self, other_channel: Channel) -> np.ndarray:
        """Return the matrix of overlaps.

        Parameters
        ----------
        other_channel : Channel
            The channel to compare with self.

        Returns
        -------
        np.ndarray
            The matrix M such that M[i,j] is the overlap count in voxels for blob
            of index i for self and the blob of index j for other_channel.
        """

        if other_channel.name in self._overlaps_dic:
            return self._overlaps_dic[other_channel.name]

        elif self.name in other_channel._overlaps_dic:
            return other_channel._overlaps_dic[self.name].transpose()
        else:
            self._set_blobwise_overlaps(other_channel)
            return self._overlaps_dic[other_channel.name]

    def max_blobwise_overlaps(
        self, other_channel: Channel, return_max_indices=True
    ) -> np.ndarray:
        """For each nucleus of self, compute the max overlap with a single nucleus of other_channel.
            if return_max_indices is True, nuclei indices that realize the max are also returned.

        Parameters
        ----------
        other_channel : Channel
            channel to compare with self

        Returns
        -------
        np.ndarray
            the max overlap (in voxels) for each nucleus in the order of self.dataframe

        np.ndarray
            the array of nucleus indices of other_channel that realize the maxima.
        """

        counts = self.blobwise_overlaps(other_channel)

        if return_max_indices:
            argmax = np.argmax(counts, axis=1)
            maxima = counts[np.arange(counts.shape[0]), argmax]

            return maxima, argmax

        else:
            return np.max(counts, axis=1)

    def max_blobwise_overlap_rates(
        self, other_channel: Channel, return_max_indices=True
    ) -> np.ndarray:
        """For each nucleus of self, compute the max overlap rate with a single nucleus of other_channel.
            if return_max_indices is True, nuclei indices that realize the max are also returned.

        Parameters
        ----------
        other_channel : Channel
            channel to compare with self

        Returns
        -------
        np.ndarray
            the max overlap rate for each nucleus in the order of self.dataframe

        np.ndarray
            the array of nucleus indices of other_channel that realize the maxima.
        """

        if return_max_indices:
            maxima, argmax = self.max_blobwise_overlaps(
                other_channel, return_max_indices=True
            )
            return maxima / self.sizes, argmax
        else:
            maxima = self.max_blobwise_overlaps(other_channel, return_max_indices=False)
            return maxima / self.sizes

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
