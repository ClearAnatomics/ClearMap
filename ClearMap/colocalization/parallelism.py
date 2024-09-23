# Copyright Gaël Cousin & Charly Rousseau
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn import neighbors

from . import channel


from ..IO import IO as io

import ClearMap.ParallelProcessing.BlockProcessing as blockprocessing
import ClearMap.ParallelProcessing.Block as block


def compare(
    img_0: str | np.ndarray,
    df_0: str | pd.DataFrame,
    img_1: str | np.ndarray,
    df_1: str | pd.DataFrame,
    scale,
    coord_names: list[str],
    blob_diameter: int,
    processes: int | None,
):
    """Make a report on colocalization between two channels
    Parameters
    ----------
    img_0 : str | np.ndarray
        binary image or path to binary image for channel 0, as outputed by cell_map
    df_0 : str | pd.DataFrame
        dataframe or path to feather file for channel 0, as outputed by cell_map
    img_1 : str | np.ndarray
        binary image or path to binary image for channel 1, as outputed by cell_map
    df_1 : str
        dataframe or path to feather file for channel 1, as outputed by cell_map

    scale : array like
        list of values specifying voxel dimensions. Caution: this scale
        must be shared by both channels.
    coord_names : list[str]
        list of string specifying the representative points coords columns
        in the dataframes
    blob_diameter : an upper bound for the sought blobs diameters, in PHYSICAL units

    processes : int | None
        positive integer or None, the number of processes to use for block processing.
        Defaults to None, if None the number of processes will equal the computer's number
        of processors.
    """

    scale = np.array(scale)
    voxel_blob_diameters = np.array(blob_diameter) / scale
    source_0 = io.as_source(img_0)
    source_1 = io.as_source(img_1)
    if not isinstance(df_0, pd.DataFrame):
        df_0 = pd.read_feather(df_0)
    if not isinstance(df_1, pd.DataFrame):
        df_1 = pd.read_feather(df_1)

    if not ((processes is None) or (isinstance(processes, int) and processes >= 1)):
        raise ValueError("The passed processes argument must be a positive integer or None.")

    axes = blockprocessing.block_axes(source_0)
    axis_wise_overlap = 2 * voxel_blob_diameters  # we
    # be careful the filtration of overlap below is needed for proper def of overlap in bp.process call
    overlap = axis_wise_overlap[axes]
    results = blockprocessing.process(
        local_report,
        [source_0, source_1],
        function_type="block",
        axes=axes,
        overlap=overlap,
        processes=processes,
        return_result=True,
    )
    c0_results = [result[0] for result in results]
    c1_results = [result[1] for result in results]

    # join the c0_results
    c0_result = pd.concat(c0_results)
    if len(c0_result) != len(df_0):
        raise RuntimeError("some channel 0 blob has been found twice, there is something wrong !")
    # join the c1_results, which might have twice the same center
    c1_result = pd.concat(c1_results).groupby(level=0).first()

    # compute closest point
    description = "center of bounding box"
    cols = [description + " " + coord_name for coord_name in coord_names]
    points_0 = c0_result[cols].to_numpy() * np.array(scale).reshape((-1, 1))
    points_1 = c1_result[cols].to_numpy() * np.array(scale).reshape((-1, 1))
    learner = neighbors.NearestNeighbors(n_neighbors=1)

    learner.fit(points_1)
    if len(points_1) > 0:
        distances, indices = learner.kneighbors(points_0)
        c0_result["closest blob distance"] = distances
        c0_result["closest blob bbox center index"] = indices
        cols = ["closest blob center" + coord for coord in coord_names]
        c0_result[cols] = c1_result[coord_names].iloc[indices]
    else:
        c0_result["closest blob distance"] = np.nan
        c0_result["closest blob bbox center index"] = np.nan
        cols = ["closest blob center" + coord for coord in coord_names]
        c0_result[cols] = np.nan

    return c0_result


def local_report(
    block_0: block.Block,
    df_0: str,
    block_1: block.Block,
    df_1: str,
    scale,
    coord_names,
):
    """Return a report dataframe for the locally computable information.

    Parameters
    ----------
    block_0 : block.Block
        _description_
    df_0 : str
        _description_
    block_1 : block.Block
        _description_
    df_1 : str
        _description_
    scale : _type_
        _description_
    coord_names : _type_
        _description_
    blob_diameter : _type_
        _description_
    """
    df_0 = pd.read_feather(df_0)
    df_1 = pd.read_feather(df_1)
    ndim = len(coord_names)
    # compute valid_indices, the ones of nuclei for which we can compute everything in this block
    # and contained_indices, the ones of nuclei whose representative is contained in the block
    # funky query to be pandas version agnostic
    data_array_0 = df_0[coord_names].to_numpy()
    data_array_1 = df_1[coord_names].to_numpy()

    slices = block_0.valid.base_slicing
    start_array = np.array([slices[i].start for i in range(ndim)])
    stop_array = np.array([slices[i].stop for i in range(ndim)])

    valid_indices_0 = np.where((data_array_0 < stop_array) & (data_array_0 >= start_array))[0]

    slices = block_0.slicing
    start_array = np.array([slices[i].start for i in range(channel_0.ndim)])
    stop_array = np.array([slices[i].stop for i in range(channel_0.ndim)])

    contained_indices_1 = np.where((data_array_1 < stop_array) & (data_array_1 >= start_array))

    sub_df_0 = df_0.iloc[valid_indices_0].reset_index()
    sub_df_1 = df_1.iloc[contained_indices_1].reset_index()
    channel_0 = channel.Channel(block_0.array, sub_df_0, voxel_dims=scale, coord_names=coord_names)
    channel_1 = channel.Channel(block_1.array, sub_df_1, voxel_dims=scale, coord_names=coord_names)
    max_overlaps, max_overlaps_indices = channel_0.max_blobwise_overlaps(channel_1, return_max_indices=True)
    centers_df_0 = channel_0.centers_df()
    c0_result = centers_df_0.set_index(sub_df_0["index"])

    blobwise_overlap_df = pd.DataFrame(
        {
            "max blobwise overlap (in voxels)": max_overlaps,
            "max relative blobwise overlap": channel_0.max_blobwise_overlap_rates(channel_1, return_max_indices=False),
            "index of maximizing overlap blob": sub_df_1.iloc[max_overlaps_indices]["index"],
        },
        index=sub_df_0["index"],
    )
    c0_result = c0_result.join(blobwise_overlap_df, validate="1:1")

    # cook the max overlap center dataframe
    centers_df_1 = channel_1.centers_df(description="maximizing blob bbox center")
    max_overlap_coords = centers_df_1.iloc[max_overlaps_indices].set_index(sub_df_0["index"])

    c0_result = c0_result.join(max_overlap_coords, validate="1:1")

    centers_df_1 = channel_1.centers_df()
    c1_result = centers_df_1.set_index(sub_df_1["index"])

    # the distances will be computed from the final joined dataframes.
    return c0_result, c1_result
