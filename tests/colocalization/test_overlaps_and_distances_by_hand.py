"""
Overlaps, centers and distances of two small 2D channels, computed by hand.

The first channel is one blob (rows 3 to 18, columns 2 to 17). The second has two blobs: the top left corner
(rows 0 to 9, columns 0 to 10) and the top right corner (rows 0 to 9, columns 14 to 19).
"""
import numpy as np
import pandas as pd
import pytest

from ClearMap.Analysis.colocalization.colocalization_channel_mask_components import (
    ColocalizationChannelMaskComponents as Channel)

COORDINATE_NAMES = ['x', 'y']
SHAPE = (20, 20)
ABSOLUTE_TOLERANCE = 1e-5

# By hand: the blob of the first channel covers (19 - 3) * (18 - 2) voxels
BLOB_SIZE = (19 - 3) * (18 - 2)
INTERSECTION_WITH_THE_TOP_LEFT_BLOB = (10 - 3) * (11 - 2)
INTERSECTION_WITH_THE_TOP_RIGHT_BLOB = (10 - 3) * (18 - 14)
RATE_OF_THE_TOP_LEFT_BLOB = INTERSECTION_WITH_THE_TOP_LEFT_BLOB / BLOB_SIZE
RATE_OF_THE_TOP_RIGHT_BLOB = INTERSECTION_WITH_THE_TOP_RIGHT_BLOB / BLOB_SIZE
INDEX_OF_THE_BLOB_OVERLAPPING_THE_MOST = 1  # The blobs of the second channel are numbered in the order of their points

CENTERS_OF_THE_FIRST_CHANNEL = [(10.5, 9.5)]
CENTERS_OF_THE_SECOND_CHANNEL = [(4.5, 16.5), (4.5, 5.0)]
DISTANCES_BETWEEN_THE_CENTERS = np.array([[9.219544, 7.5]])


@pytest.fixture
def channels():
    binary_1 = np.zeros(SHAPE, dtype='uint8')
    binary_1[3:19, 2:18] = 1
    binary_2 = np.zeros(SHAPE, dtype='uint8')
    binary_2[:10, :11] = 1
    binary_2[:10, 14:] = 1

    points_1 = pd.DataFrame([(3, 17)], columns=COORDINATE_NAMES)
    points_2 = pd.DataFrame([(0, 19), (3, 3)], columns=COORDINATE_NAMES)
    return (Channel(binary_1, points_1, coord_names=COORDINATE_NAMES),
            Channel(binary_2, points_2, coord_names=COORDINATE_NAMES))


def test_the_centers_are_the_centers_of_the_bounding_boxes(channels):
    channel_1, channel_2 = channels

    assert [channel_1.center(index) for index in range(len(channel_1.dataframe))] == CENTERS_OF_THE_FIRST_CHANNEL
    assert [channel_2.center(index) for index in range(len(channel_2.dataframe))] == CENTERS_OF_THE_SECOND_CHANNEL


def test_the_overlap_rate_is_the_sum_of_the_rates_of_the_blobs_it_overlaps(channels):
    channel_1, channel_2 = channels

    rates = channel_1.overlap_rates(channel_2)

    assert np.allclose(rates, [RATE_OF_THE_TOP_LEFT_BLOB + RATE_OF_THE_TOP_RIGHT_BLOB], atol=ABSOLUTE_TOLERANCE)


def test_the_maximal_blobwise_overlap_rate_is_the_one_of_the_blob_overlapping_the_most(channels):
    channel_1, channel_2 = channels

    rates = channel_1.max_blobwise_overlap_rates(channel_2, return_max_indices=False)

    assert np.allclose(rates, max(RATE_OF_THE_TOP_LEFT_BLOB, RATE_OF_THE_TOP_RIGHT_BLOB), atol=ABSOLUTE_TOLERANCE)


def test_the_blob_overlapping_the_most_is_returned_with_its_rate(channels):
    channel_1, channel_2 = channels

    rates, indices = channel_1.max_blobwise_overlap_rates(channel_2, return_max_indices=True)

    assert np.allclose(rates, max(RATE_OF_THE_TOP_LEFT_BLOB, RATE_OF_THE_TOP_RIGHT_BLOB), atol=ABSOLUTE_TOLERANCE)
    assert indices == np.array([INDEX_OF_THE_BLOB_OVERLAPPING_THE_MOST])


def test_the_distances_are_the_distances_between_the_centers(channels):
    channel_1, channel_2 = channels

    assert np.allclose(channel_1.centers_distances(channel_2), DISTANCES_BETWEEN_THE_CENTERS, atol=1e-7)
