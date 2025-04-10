# Copyright Gaël Cousin & Charly Rousseau

"""
Colocalization
--------------

This module allows to compare signal between different channels

The basic idea is to exploit the data of
- a binary mask per channel
- a dataframe with a representative point per connected component, given in pixel coords
    with possibly extra measurement information for the nucleus in the respective channel,
    eg some intensity measurement.

The idea is to be able to determine if two detected nuclei in distinct channels
correspond indeed to the same nucleus.

.. note::
    We have to rely on well aligned channel images for this module to work.

The simplest method is to break symmetry and have a reference channel that
marks all nuclei of the studied cell category (e.g. all cell type or neurons)

Then, for each reference channel detected nucleus, decide if it is to be considered
positive for the the other channels.

This can be made relying on the following capabilities, given two channels
- Compute the overlap of each connected component of a given channel with the
other channel, component-wise or globally or, more generally
- compute a matching score based on the information we have

Ideally, all of this is to be made efficiently both with quick sequential code
and parallelization techniques.

.. warning::
    The current version of this module deals only with 3d images.
    See shape_detection.label_pixels_from_centers commented version for a 2d version.
"""

from __future__ import annotations
from functools import cached_property
import warnings

import numpy as np
import pandas as pd
import scipy.ndimage as ndi
from scipy.spatial.transform import Rotation
from sklearn import neighbors
import skimage.morphology

from ClearMap.Analysis.colocalization import bbox as bounding_boxes
from ClearMap.Analysis.colocalization.parallelism import compare
from ClearMap.Utils.exceptions import ClearMapValueError


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
    """Count the number of occurrences for all the conjunctions of labels.

    Parameters
    ----------
    labels_1 : np.array
        an array of labels
    labels_2 : np.array
        another array of labels, with the same shape as labels_1

    Returns
    -------
    np.array
        The array counts such that counts[i,j] == np.count_nonzero((labels_1==i) * (labels_2==j)).

    """

    # inspired by the join segmentation of skimage
    max_1 = labels_1.max()
    max_2 = labels_2.max()
    factor = max_2 + 1
    # factor will also be the height of the output
    max_val = max_1 * factor + max_2
    # determine the cheapest dtype to hold the join
    for bit_width in (8, 16, 32, 64):
        if 2**(bit_width -1) > max_val:  # -1 because signed
            dtype = f'int{bit_width}'
            break
    else:
        raise RuntimeError("This software is not able to store the join labels.")

    joined_labels = factor * labels_1.astype(dtype) + labels_2.astype(dtype)

    counts = np.bincount(joined_labels.flatten(), minlength=(max_1 + 1) * (max_2 + 1))

    # We return the counts recast in 2d
    return counts.reshape((max_1 + 1, max_2 + 1))


# for comparison/test purposes
def _naive_bilabel_bincount(arr_a, arr_b):
    m = arr_a.max() + 1
    n = arr_b.max() + 1
    res = np.zeros((m, n), dtype="uint8")
    for i in range(m):
        for j in range(n):
            res[i, j] = np.count_nonzero((arr_a == i) * (arr_b == j))
    return res


def _data_frame_ok(df: pd.DataFrame):
    return (
        isinstance(df.index, pd.RangeIndex)
        and df.index.start == 0
        and df.index.stop == len(df)
        and df.index.step == 1
        and "index" not in df.columns
    )


def contiguous_labels(labels):  # TODO: try optimization with return_index argument of np.unique
    vals = np.unique(labels)
    labels = np.searchsorted(vals, labels)
    return labels


def cleanup(binary_img: np.ndarray, df: pd.DataFrame, coord_names):
    """Return the labeled image giving exactly the connected components of binary_img that correspond to a representative in dataframe.
    Parameters
    ----------
    binary_img : np.ndarray
        _description_
    df : pd.DataFrame
        _description_
    coord_names : _type_
        _description_
    """
    labels, _ = ndi.label(binary_img)
    coords = df[coord_names]
    relevant_labels = labels[*coords.values.transpose()]
    cleaned_labels = np.where(np.isin(labels, relevant_labels), labels, 0)
    cleaned_labels = contiguous_labels(cleaned_labels)
    return cleaned_labels

def get_minimum_dtype(max_val, signed=True):
    for bit_width in (8, 16, 32, 64):
        if 2 ** (bit_width - int(signed)) > max_val:  # -1 because signed
            dtype = f'{"" if signed else "u"}int{bit_width}'
            break
    else:
        raise RuntimeError(f"No integer dtype is suitable to store values up to {max_val}.")
    return dtype

# an efficie,t alternative to cleanup, yet it seems that it does not work properly if binary_img is read from a file.
# replacing binary_img by binary_img.copy() in the function call fixes this strange bug.
def watershed_label(binary_img: np.ndarray, representatives:tuple[np.ndarray,...], label_values:np.ndarray, connectivity=1, dtype=None):
    """Label the connected components of binary image, assigning the values for a set of representatives.

    Any connected component without a reprenstative is labeled 0.


    Parameters
    ----------
    binary_img: np.ndaray
        n-dimensional binary image (0= background,1=foreground)
    representatives : tuple[np.ndarray,...]
        n-tuple of flat integer arrays of the same size, say k. They give the n coordinates of the k representatives.
    label_values : np.ndarray
        flat size k array specifying the sought label for each representative in the output
    connectivity : int | np.ndarray optional
        a specification of the connectivity notion, as in skimage label or watershed
    dtype : _type_, optional
        dtype of the output. must be a uint or float dtype if specified. Defaults to None.
        If None, the smallest suitable uint dtype is used.

    Returns
    -------
    np.ndarray :
        The labeling of binary_img with the specified labels, where all the connected components without any element
        in repesentatives ares erased.

    Raises
    ------
    ValueError
        An error is raised if negative label_values are passed.
    """
    
    values = label_values
    if values.min() < 0:
        raise ValueError("No negative label value accepted.")
    if dtype is None:
        dtype = get_minimum_dtype(values.max(), signed=False)
    seeds = np.zeros_like(binary_img, dtype=dtype)
    seeds[representatives] = values
    res=skimage.segmentation.watershed(
        np.zeros_like(seeds), markers=seeds, mask=binary_img, connectivity=connectivity
    )
    return  res




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
        coord_names=("x", "y", "z"),
        voxel_dims=None,
        physical_origin=None,
        channel_name="",
        clean_image: bool = False,
        already_clean: bool = False,
    ) -> None:
        """Instantiate an object

        Parameters
        ----------
        binary_img : np.ndarray
            _description_
        dataframe : pd.DataFrame
            the index must correspond to the row rank, no funky index names or subsets.
        coord_names : tuple, optional
            _description_, by default ("x", "y", "z")
        voxel_dims : _type_, optional
            _description_, by default None
        physical_origin : _type_, optional
            _description_, by default None
        channel_name : str, optional
            _description_, by default ""
        clean_image: bool
            If connected components of binary_img corresponding to no representative in dataframe should be erased. Defaults to False.
            This is a costly operation designed to be used only in blocks.
        already_clean: bool
            Bypass prior cleaning for the blocks if True. Defaults to False.
        Raises
        ------
        ValueError
            _description_
        ValueError
            _description_
        """
        if _data_frame_ok(dataframe):
            self.dataframe = dataframe
        else:
            raise ValueError(
                "Either the passed dataframe indexing is not a step 1 integer range starting from 0, or the 'index' column name is used."
            )
        self.already_clean = already_clean
        if clean_image and not self.already_clean:
            labels = cleanup(binary_img, dataframe, coord_names)
            self._labels = labels
            self.binary_img = labels > 0
        else:
            self.binary_img = binary_img

        self.coord_names = coord_names
        self.ndim = len(self.coord_names)
        if voxel_dims is None:
            voxel_dims = [1] * self.ndim
        if physical_origin is None:
            physical_origin = [0] * self.ndim
        self.voxel_dims = np.array(voxel_dims)
        self.physical_origin = np.array(physical_origin)

        if self.voxel_dims.size != self.ndim or self.physical_origin.size != self.ndim:
            raise ValueError("The lengths of voxel_dims, physical_origin and coord_names do not match.")

        # we identify the channel by a name fo caching management
        if not channel_name:
            channel_name = Channel.channel_count
            Channel.channel_count += 1
        if channel_name in Channel.channel_names:
            raise ValueError("The passed Channel name is already in use.")

        self.name = channel_name
        Channel.channel_names.append(channel_name)

        self._overlaps_dic = {}

        self._labels = None
        self.__bounding_boxes_array = None
        self.__centers = None

    @property
    def representative_points(self):
        return self.dataframe[self.coord_names]

    # @cached_property
    def labels(
        self,
    ):  # TODO optimize relabeling from representatives, eg with watershedding
        """Return the labeled image from the binary mask"""
        if self._labels is None:
            self._labels, _ = ndi.label(self.binary_img)
        return self._labels

    @cached_property
    def index_label_correspondence(self) -> np.array:
        """Return the labels of nuclei in the nuclei index order

        Returns
        -------
        np.array
            the flat array of the labels in the nuclei index order
        """
        coords = self.dataframe[self.coord_names]
        return np.array([self.labels()[tuple(coords.iloc[index])] for index in range(len(self.dataframe))])

    @cached_property
    def label_index_correspondence(self):
        # beware that not all label corresponds to an index,
        # if we do not have a complete set of reps in the dataframe.
        # We set arbitrarily the index value 0 for the labels that will not be used.
        max_label = self.index_label_correspondence.max()
        return np.array(
            [
                (
                    np.where(self.index_label_correspondence == label)[0][0]
                    if label in self.index_label_correspondence
                    else 0
                )
                for label in range(max_label + 1)
            ]
        )

    @cached_property
    def _shifted_index_label_correspondence(self):
        """Return the index_label_correspondence shifted by 1

        Returns
        -------
        list[int]
            the list L such that L[i]=index_label_correspondence[i]-1
        """
        return np.array(self.index_label_correspondence) - 1

    def nucleus(self, index):
        """Return the nucleus associated to a given index, as a mask.

        Parameters
        ----------
        index : int
        index for the chosen representative point in self.dataframe

        """
        return self.index_label_correspondence[index] == self.labels()

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
            slice(*[self._bounding_boxes_array()[i, axis, j] + j for j in range(2)]) for axis in range(self.ndim)
        )

    # kept mainly for comparison and for the generic case of function below
    def _naive_bounding_boxes_array(self):
        stack = []
        for index in range(len(self.dataframe)):
            nuke = self.nucleus(index)
            locus = np.where(nuke)
            stack += [[[locus[i].min(), locus[i].max()] for i in range(self.ndim)]]
        return np.array(stack)

    # @cached_property
    def _bounding_boxes_array(self):
        """Return an array that specifies the bounding boxes for self.labels()

        Returns
        -------
        np.ndarray
            The array A of size len(self.dataframe) such that A[i] specifies the
            bounding box of the blob of line i in self.dataframe.
            For any axis, A[i][axis_index][0] is the least index value along axis
            with index axis_index for a pixel of the ith nucleus and A[i][axis_index][1]
            is the maximal reached index value for the same nucleus.
            Beware that from this max, 1 must be added to obtain the stop attribute
            for a slice defining the corresponding bounding box.
        """
        if self.labels().ndim not in [1, 2, 3]:
            if self.__bounding_boxes_array is None:
                self.__bounding_boxes_array = self._naive_bounding_boxes_array()
            return self.__bounding_boxes_array
        # optimized bounding_boxes, to avoid looping on all labels
        if self.__bounding_boxes_array is None:
            if self.labels().ndim == 1:
                res = bounding_boxes.bbox_1d(self.labels())
            elif self.labels().ndim == 2:
                res = bounding_boxes.bbox_2d(self.labels())
            else:
                res = bounding_boxes.bbox_3d(self.labels())
            self.__bounding_boxes_array = np.ascontiguousarray(res[self.index_label_correspondence])
        return self.__bounding_boxes_array

    # @cached_property
    def centers(self):
        if self.__centers is None:
            starts = self._bounding_boxes_array()[:, :, 0].astype("float32")
            stops = self._bounding_boxes_array()[:, :, 1].astype("float32")
            self.__centers = (starts + stops) / 2
        return self.__centers

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
        return tuple(self.centers()[i])

    def centers_df(self, description="center of bounding box"):
        cols = [description + " " + coord_name for coord_name in self.coord_names]
        return pd.DataFrame(self.centers(), columns=cols)

    @cached_property
    def sizes(self):
        """Return the list of the sizes of nuclei in the order of our dataframe.

        Returns
        -------
        np.ndarray
            The 1D array of sizes of nuclei (ints) in the order of our dataframe.
        """
        return np.bincount(self.labels().flatten())[self.index_label_correspondence]

    def masked_sizes(self, mask):
        """Return the list of the sizes of the non-masked part of each nucleus

        Returns
        -------
        list[int]
            The list of counts of True pixels in mask for each nucleus, in the
            order of our dataframe.
        """
        return np.bincount((mask * self.labels()).flatten(), minlength=self.sizes.size + 1)[
            self.index_label_correspondence
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

        return self.masked_sizes(other_channel.binary_img) / self.sizes

    def _set_blobwise_overlaps(self, other_channel: Channel):
        """
        Compute and store the matrix M of overlaps where M[i,j] is the overlap between blob
        of index i for self and the blob of index j for other_channel.
        """
        counts = bilabel_bincount(self.labels(), other_channel.labels())
        self._overlaps_dic[other_channel.name] = self._reorder_counts(other_channel, counts)

    def _reorder_counts(self, other_channel, counts):
        return counts[self.index_label_correspondence, :][:, other_channel.index_label_correspondence]

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
        if np.any(self.voxel_dims != other_channel.voxel_dims) or np.any(
            self.physical_origin != other_channel.physical_origin
        ):
            raise ValueError(
                "The blobwise overlap is implemented only for two Channels with same voxel_dims and physical_origin."
            )

        if other_channel.name in self._overlaps_dic:
            return self._overlaps_dic[other_channel.name]

        elif self.name in other_channel._overlaps_dic:
            return other_channel._overlaps_dic[self.name].transpose()
        else:
            self._set_blobwise_overlaps(other_channel)
            return self._overlaps_dic[other_channel.name]

    def max_blobwise_overlaps(
        self, other_channel: Channel, return_max_indices: bool = True
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """For each nucleus of self, compute the max overlap with a single nucleus of other_channel.
            if return_max_indices is True, nuclei indices that realize the max are also returned.

        Parameters
        ----------
        other_channel: Channel
            channel to compare with self
        return_max_indices: bool
            whether to return the indices of the nuclei of other_channel that realize the maxima (i.e. argmax).
             (default True)

        Returns
        -------
        np.ndarray
            the max overlap (in voxels) for each nucleus in the order of self.dataframe

        np.ndarray
            the array of nucleus indices of other_channel that realize the maxima.
        """

        counts = self.blobwise_overlaps(other_channel)
        # we handle the case of no blob for other_channel by adding a virtual empty blob
        counts = np.hstack([counts, np.zeros((counts.shape[0], 1))])

        if return_max_indices:
            argmax = np.argmax(counts, axis=1)
            maxima = counts[np.arange(counts.shape[0]), argmax]
            if (counts.shape[1] - 1) in argmax:
                argmax = np.array([np.nan] * counts.shape[0])
                warnings.warn("Returning nan argmax due to empty other_channel blob list.")
            return maxima, argmax

        else:
            return np.max(counts, axis=1)

    def max_blobwise_overlap_rates(
        self, other_channel: Channel, return_max_indices: bool = True
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """For each nucleus of self, compute the max overlap rate with a single nucleus of other_channel.
            if return_max_indices is True, nuclei indices that realize the max are also returned.

        Parameters
        ----------
        other_channel: Channel
            channel to compare with self
        return_max_indices: bool
            Whether to return the indices of the nuclei of other_channel that realize the maxima (i.e. argmax).
                (default True)

        Returns
        -------
        np.ndarray
            the max overlap rate for each nucleus in the order of self.dataframe

        np.ndarray
            the array of nucleus indices of other_channel that realize the maxima.
        """

        if return_max_indices:
            maxima, argmax = self.max_blobwise_overlaps(other_channel, return_max_indices=True)
            return maxima / self.sizes, argmax
        else:
            maxima = self.max_blobwise_overlaps(other_channel, return_max_indices=False)
            return maxima / self.sizes

    def centers_distances(self, other_channel: Channel) -> np.ndarray:
        """Return array of distances between centers across the two channels.

        Parameters
        ----------
        other_channel : Channel
            Channel to compare with self

        Returns
        -------
        np.ndarray
            distance matrix A: A[index_1,index_2] is the distance between the centers
            of the nucleus indexed by index_1 in self.dataframe and  the nucleus
            indexed by index_2 in other_channel.dataframe.
        """
        physical_centers_1 = self.centers() * (self.voxel_dims.reshape((1, -1))) + self.physical_origin
        physical_centers_2 = (
            other_channel.centers() * (other_channel.voxel_dims.reshape((1, -1))) + other_channel.physical_origin
        )

        return distances(physical_centers_1, physical_centers_2)

    def closest_center_distances(
        self, other_channel: Channel, return_min_indices=True
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """
        Return the distances to the closest centers in `other_channel` for each center in `self`.

        Parameters
        ----------
        other_channel : Channel
            _description_
        return_min_indices : bool, optional.
          Defaults to True.


        Returns
        -------
            similar to max_blobwise_overlap
        """
        distances_ = self.centers_distances(other_channel)  # TODO cache distances as overlaps
        argmin = np.argmin(distances_, axis=1)
        minima = distances_[:, argmin]

        if return_min_indices:
            return minima, argmin
        else:
            return minima

    def compare(
        self, other_channel: Channel, blob_diameter: int, size_min: int, size_max: int, processes: int | None = None
    ) -> pd.DataFrame:
        """
        Return a final colocalization report


        Parameters
        ----------
        other_channel : Channel
            _description_
        blob_diameter : int
            an upper bound for the sought blobs diameters, in PHYSICAL units
        size_min : int
            min size in voxels for a block dimension
        size_max : int
            max size in voxels for a block dimension
        processes : int | None, optional
            positive integer or None, the number of processes to use for block processing.
            Defaults to None, if None the number of processes will equal the computer's number
            of processors.

        Returns
        -------
        pd.DataFrame
            A colocalization report in pandas dataframe format
        """

        if np.any(self.voxel_dims != other_channel.voxel_dims):
            raise ValueError("The two considered channels must have the same voxel_dims.")
        scale = self.voxel_dims

        if self.coord_names != other_channel.coord_names:
            raise ValueError("The comparison of Channels with different coord_names is not implemented.")
        return compare(
            self.binary_img,
            self.dataframe,
            other_channel.binary_img,
            other_channel.dataframe,
            scale=scale,
            coord_names=self.coord_names,
            blob_diameter=blob_diameter,
            processes=processes,
            size_min=size_min,
            size_max=size_max,
            are_already_clean=(self.already_clean, other_channel.already_clean),
        )

    # for comparison/testing purposes
    def _naive_compare(
        self,
        other_channel,
    ):
        max_overlaps, max_overlaps_indices = self.max_blobwise_overlaps(other_channel, return_max_indices=True)
        centers_df_0 = self.centers_df()
        c0_result = centers_df_0
        blobwise_overlap_df = pd.DataFrame(
            {
                "max blobwise overlap (in voxels)": max_overlaps,
                "max relative blobwise overlap": self.max_blobwise_overlap_rates(
                    other_channel, return_max_indices=False
                ),
                "index of maximizing overlap blob": max_overlaps_indices,
            },
        )
        c0_result = c0_result.join(blobwise_overlap_df, validate="1:1")

        # cook the max overlap center dataframe
        centers_df_1 = other_channel.centers_df(description="maximizing blob bbox center")
        max_overlap_coords = centers_df_1.iloc[max_overlaps_indices].set_index(self.dataframe.index)

        c0_result = c0_result.join(max_overlap_coords, validate="1:1")
        c1_result = other_channel.centers_df()
        # compute closest point
        description = "center of bounding box"
        cols = [description + " " + coord_name for coord_name in self.coord_names]
        points_0 = c0_result[cols].to_numpy() * np.array(self.voxel_dims).reshape((1, -1))
        points_1 = c1_result[cols].to_numpy() * np.array(self.voxel_dims).reshape((1, -1))
        learner = neighbors.NearestNeighbors(n_neighbors=1, algorithm="brute", n_jobs=-1)

        learner.fit(points_1)
        if len(points_1) > 0:
            distances_, indices = learner.kneighbors(points_0)
            c0_result["closest blob distance"] = distances_
            c0_result["closest blob bbox center index"] = indices.flatten()
            c0_cols = ["closest blob center " + coord for coord in self.coord_names]
            c0_result[c0_cols] = c1_result[cols].iloc[indices.flatten()].to_numpy()
        else:
            c0_result["closest blob distance"] = np.nan
            c0_result["closest blob bbox center index"] = np.nan
            cols = ["closest blob center " + coord for coord in self.coord_names]
            c0_result[cols] = np.nan

        return c0_result


# random Channel generation


def _random_shape(radius):
    dice = np.random.randint(3)
    if dice == 0:
        shape = skimage.morphology.cube(2 * int(radius / 1.8) + 1)
    elif dice == 1:
        shape = skimage.morphology.octahedron(radius)
    elif dice == 2:
        shape = skimage.morphology.ball(radius)
    else:
        raise ClearMapValueError(f"The dice should not have rolled that way. ({dice})")

    locus = np.where(shape)
    points = np.vstack(locus).transpose()

    rand_rot = Rotation.random()
    new_width = np.ceil((2 * radius + 1) * 1.8).astype("uint8") + 3
    new_center = np.array([new_width // 2] * 3)

    centered = points - np.array([radius] * 3)
    new_centered = rand_rot.apply(centered)
    new_locus = (np.round(new_centered + new_center).astype("uint64")).transpose()
    xs, ys, zs = new_locus[0], new_locus[1], new_locus[2]

    result = np.zeros((new_width,) * 3, dtype="bool")
    result[xs, ys, zs] = 1

    return ndi.binary_closing(result, structure=np.ones((3,) * 3)), new_center


def _random_blobs(shape, num_points, min_radius, max_radius):
    if len(shape) != 3:
        raise ValueError("Only 3dim shapes are accepted")
    radii = np.random.randint(min_radius, max_radius, size=(num_points,))
    centers = np.vstack([np.random.randint(shape[i] - 1, size=(num_points,)) for i in range(3)]).transpose()
    margin = 2 * max_radius
    centers += margin
    result = np.zeros(tuple([size + 2 * margin for size in shape]), dtype="bool")
    for i in range(num_points):
        footprint, fp_center = _random_shape(radii[i])
        locus = np.vstack(np.where(footprint)).transpose()
        locus = locus - np.array(fp_center) + centers[i]
        locus = locus.transpose()
        xs, ys, zs = locus[0], locus[1], locus[2]
        # we make sure the added blob will not get connected to previous blobs (for 1-connectivity)
        candidate = np.zeros(tuple([size + 2 * margin for size in shape]), dtype="bool")
        candidate[xs, ys, zs] = 1
        fattened = ndi.binary_dilation(candidate)
        fattened_locus = np.where(fattened)
        xs, ys, zs = fattened_locus[0], fattened_locus[1], fattened_locus[2]
        if 1 not in result[xs, ys, zs]:
            result += candidate

    return result[margin:-margin, margin:-margin, margin:-margin]


def _random_channel_data(shape, num_points, min_radius, max_radius):
    img = _random_blobs(shape, num_points, min_radius, max_radius)
    labels, _ = ndi.label(img)
    vals, indices = np.unique(labels, return_index=True)
    indices = indices[np.where(vals)]
    representatives = np.unravel_index(indices, labels.shape)
    return labels, representatives


def random_channel(shape, num_points, min_radius, max_radius):
    """Return a random Channel of given shape.
    The underlying image is obtained by successive attempts of blob additions.
    If the candidate blob would be connected with existing blobs, it is rejected.


    Parameters
    ----------
    shape : tuple
        3d shape for the Channel binary_image attribute
    num_points : int
        number of attempts to add a blob
    min_radius : _type_
        min radius for candidate blob, if on the boundary
        the resulting blob could have smaller radius
    max_radius : _type_
        max radius for any blob in the final Channel,
        as detected as 1-connected components
    """
    labels, reps = _random_channel_data(shape, num_points, min_radius, max_radius)
    df = pd.DataFrame({c: reps[i] for i, c in enumerate("xyz")})
    return Channel(labels > 0, df)
