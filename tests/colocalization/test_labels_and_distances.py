"""
The helpers of the colocalization: the representative of each label, the counts of the pairs of labels of two images
and the distances between two sets of points. Each is compared with a naive computation.

The random inputs come from a generator with a fixed seed, so a failure can be reproduced.
"""
import numpy as np
import pytest

from ClearMap.Analysis.colocalization.colocalization_channel_mask_components import (
    _naive_bilabel_bincount, bilabel_bincount, distances)
from ClearMap.Analysis.Measurements.maxima_detection import label_representatives

RANDOM_SEED = 0
N_RANDOM_CASES = 100


@pytest.fixture
def random_generator():
    return np.random.default_rng(RANDOM_SEED)


def _random_labels(random_generator, *, n_dims, high):
    shape = tuple(random_generator.integers(15, size=(n_dims,)))
    return random_generator.integers(high, size=shape)


@pytest.mark.parametrize('transpose', [False, True])
def test_the_representatives_are_one_voxel_of_each_label_except_the_background(random_generator, transpose):
    for _ in range(N_RANDOM_CASES):
        labeled = _random_labels(random_generator, n_dims=3, high=10)
        labeled = labeled.transpose() if transpose else labeled

        representatives = label_representatives(labeled)

        points = [tuple(axis[k] for axis in representatives) for k in range(len(representatives[0]))]
        labels_of_the_points = np.array([labeled[point] for point in points])
        labels = np.unique(labeled)
        labels = labels[labels != 0]
        assert labels_of_the_points.size == labels.size
        assert np.all(labels_of_the_points == labels)


def test_the_counts_of_the_pairs_of_labels_are_the_naive_counts(random_generator):
    for case in range(N_RANDOM_CASES):
        if case % 2:
            labels_1 = np.asfortranarray(random_generator.integers(5, size=(20, 20)))
            labels_2 = random_generator.integers(6, size=(20, 20))
        else:
            labels_1 = random_generator.integers(6, size=(20, 20))
            labels_2 = random_generator.integers(5, size=(20, 20))

        counts = bilabel_bincount(labels_1, labels_2)

        assert np.all(counts == _naive_bilabel_bincount(labels_1, labels_2))


@pytest.mark.parametrize('dtype, highest_labels', [(np.uint8, (200, 3)), (np.uint16, (300, 400))])
def test_the_counts_of_the_pairs_of_labels_do_not_overflow_in_the_dtype_of_the_labels(random_generator, dtype,
                                                                                        highest_labels):
    # The number of pairs (201 * 4 = 804 and 301 * 401 = 120701) does not fit in the dtype of the labels (255 and
    # 65535). The highest label of each image is on a pixel where the other image has no label.
    labels_1 = random_generator.integers(highest_labels[0] + 1, size=(60, 60)).astype(dtype)
    labels_2 = random_generator.integers(highest_labels[1] + 1, size=(60, 60)).astype(dtype)
    labels_1[0, 0], labels_2[0, 0] = highest_labels[0], 0
    labels_1[1, 1], labels_2[1, 1] = 0, highest_labels[1]

    counts = bilabel_bincount(labels_1, labels_2)

    assert counts.shape == (highest_labels[0] + 1, highest_labels[1] + 1)
    assert np.all(counts == _naive_bilabel_bincount(labels_1, labels_2))


def test_the_distances_are_the_euclidean_distances_between_all_the_pairs_of_points(random_generator):
    for _ in range(N_RANDOM_CASES):
        n_dims = random_generator.integers(1, 10)
        points_1 = random_generator.integers(10, size=(30, n_dims))
        points_2 = random_generator.integers(10, size=(30, n_dims))
        expected = np.sqrt(((points_1[:, None, :] - points_2[None, :, :]) ** 2).sum(axis=-1))

        assert np.allclose(distances(points_1, points_2), expected)
