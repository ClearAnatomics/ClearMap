"""
``compare()`` processes the two channels block by block, in several processes. Its report must be the one of the naive
computation on the whole image (``_naive_compare()``), and the blobs it points to (the one overlapping the most and
the closest one) must really be the ones with these overlaps and distances.

The channels are random blobs, with the same random state for every run.
"""
import numpy as np
import pytest

from ClearMap.Analysis.colocalization.colocalization_channel_mask_components import random_channel

SHAPE = (80, 80, 80)
N_BLOB_ATTEMPTS = 60
MIN_RADIUS, MAX_RADIUS = 2, 4  # The radii are drawn in [MIN_RADIUS, MAX_RADIUS[
BLOB_DIAMETER = 8  # An upper bound for the diameters of the blobs (2 * 3 + 1 voxels at most)
# The blocks overlap by 4 blob diameters, so they must be wider than that: 5 to 6 diameters. The image of 80 voxels
# is processed in several blocks.
SIZE_MIN, SIZE_MAX = 5 * BLOB_DIAMETER, 6 * BLOB_DIAMETER
PROCESSES = 2
RANDOM_SEED = 0
ABSOLUTE_TOLERANCE = 1e-3
COORDINATE_NAMES = ['x', 'y', 'z']


@pytest.fixture(scope='module')
def channels_and_reports():
    random_state = np.random.get_state()
    np.random.seed(RANDOM_SEED)
    try:
        channel_0 = random_channel(SHAPE, N_BLOB_ATTEMPTS, MIN_RADIUS, MAX_RADIUS)
        channel_1 = random_channel(SHAPE, N_BLOB_ATTEMPTS, MIN_RADIUS, MAX_RADIUS)
    finally:
        np.random.set_state(random_state)

    naive = channel_0._naive_compare(channel_1)
    parallel = channel_0.compare(channel_1, blob_diameter=BLOB_DIAMETER, size_min=SIZE_MIN, size_max=SIZE_MAX,
                                 processes=PROCESSES)
    return channel_0, channel_1, naive, parallel


def test_the_parallel_report_has_the_overlaps_and_distances_of_the_naive_report(channels_and_reports):
    _, _, naive, parallel = channels_and_reports
    compared_columns = ['max relative blobwise overlap', 'closest blob distance']

    assert len(naive) > 0
    assert np.allclose(naive[compared_columns].to_numpy(), parallel[compared_columns].to_numpy(),
                       atol=ABSOLUTE_TOLERANCE)


def test_the_closest_blob_of_the_parallel_report_is_at_the_distance_reported(channels_and_reports):
    # Two blobs can be equally close: the index may differ from the naive one, but it has to be a closest blob
    channel_0, channel_1, _, parallel = channels_and_reports
    indices = parallel['closest blob bbox center index'].to_numpy().astype(int)
    coordinates = parallel[[f'closest blob center {name}' for name in COORDINATE_NAMES]].to_numpy()
    distances = parallel['closest blob distance'].to_numpy()

    assert np.allclose(channel_1.centers_df().iloc[indices].to_numpy(), coordinates, atol=ABSOLUTE_TOLERANCE)
    assert np.allclose(channel_0.centers_distances(channel_1)[np.arange(indices.size), indices], distances,
                       atol=ABSOLUTE_TOLERANCE)


def test_the_blob_overlapping_the_most_in_the_parallel_report_has_the_overlap_reported(channels_and_reports):
    channel_0, channel_1, _, parallel = channels_and_reports
    indices = parallel['index of maximizing overlap blob'].to_numpy().astype(int)
    coordinates = parallel[[f'maximizing blob bbox center {name}' for name in COORDINATE_NAMES]].to_numpy()
    overlaps = parallel['max blobwise overlap (in voxels)'].to_numpy()

    assert np.allclose(channel_1.centers_df().iloc[indices].to_numpy(), coordinates, atol=ABSOLUTE_TOLERANCE)
    assert np.allclose(channel_0.blobwise_overlaps(channel_1)[np.arange(indices.size), indices], overlaps,
                       atol=ABSOLUTE_TOLERANCE)
