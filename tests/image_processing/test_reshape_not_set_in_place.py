"""
The neighbourhoods of the topology modules and the result of ``apply_local_function`` are reshaped with ``reshape()``.

They were reshaped by assigning ``array.shape``, which NumPy 2.5 deprecates (``DeprecationWarning: Setting the shape
on a NumPy array has been deprecated``): it will be an error in a later release.
"""
import warnings

import numpy as np

import ClearMap.ImageProcessing.LocalStatistics as local_statistics
import ClearMap.ImageProcessing.Topology.Topology2d as topology_2d
import ClearMap.ImageProcessing.Topology.Topology3d as topology_3d

BLOCK_SHAPE = (2, 2)
IMAGE_SHAPE = (4, 4)


def test_the_2d_neighbourhoods_are_3_by_3_with_the_centre_left_out():
    image = np.zeros((5, 5), dtype=bool)
    image[1:4, 1:4] = True  # A 3 x 3 square inside the zero border
    x, y = np.array([2, 1]), np.array([2, 1])  # The centre of the square and its top left corner

    with warnings.catch_warnings():
        warnings.simplefilter('error', DeprecationWarning)
        neighbourhoods = topology_2d.extract_neighbourhood(image, x, y)

    assert neighbourhoods.shape == (2, 3, 3)
    assert neighbourhoods[0].sum() == 8  # All the square but the centre
    assert neighbourhoods[1].sum() == 3  # The right, bottom and bottom right neighbours of the corner
    assert not neighbourhoods[:, 1, 1].any()


def test_the_3d_neighbourhoods_are_3_by_3_by_3_with_the_centre_left_out():
    image = np.zeros((5, 5, 5), dtype=bool)
    image[1:4, 1:4, 1:4] = True
    x = y = z = np.array([2])

    with warnings.catch_warnings():
        warnings.simplefilter('error', DeprecationWarning)
        neighbourhoods = topology_3d.extract_neighbourhood(image, x, y, z)

    assert neighbourhoods.shape == (1, 3, 3, 3)
    assert neighbourhoods.sum() == 26
    assert not neighbourhoods[:, 1, 1, 1].any()


def test_a_local_scalar_function_gives_one_value_per_block_without_a_trailing_axis():
    source = np.arange(np.prod(IMAGE_SHAPE), dtype=float).reshape(IMAGE_SHAPE)

    with warnings.catch_warnings():
        warnings.simplefilter('error', DeprecationWarning)
        maxima = local_statistics.apply_local_function(source, np.max, selem=BLOCK_SHAPE, interpolate=None)

    np.testing.assert_array_equal(maxima, [[5, 7], [13, 15]])  # The maximum of each 2 x 2 block
