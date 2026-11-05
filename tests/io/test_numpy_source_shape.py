"""
Setting the shape of a numpy source reshapes the array of the source with ``reshape()``.

It assigned ``array.shape``, which NumPy 2.5 deprecates (an error in a later release), and so also reshaped the
array the caller had given (the data viewer adds a z axis to the 2D sources it shows).
"""
import warnings

import numpy as np

from ClearMap.IO.source.backends.npy_backend import NumpySource

PLANE_SHAPE = (3, 4)
VOLUME_SHAPE = PLANE_SHAPE + (1,)


def test_setting_the_shape_of_a_numpy_source_reshapes_its_data():
    plane = np.arange(np.prod(PLANE_SHAPE)).reshape(PLANE_SHAPE)
    source = NumpySource(array=plane)

    with warnings.catch_warnings():
        warnings.simplefilter('error', DeprecationWarning)
        source.shape = VOLUME_SHAPE

    assert source.shape == VOLUME_SHAPE
    np.testing.assert_array_equal(source.array[:, :, 0], plane)


def test_setting_the_shape_of_a_numpy_source_leaves_the_array_of_the_caller_as_it_was():
    plane = np.zeros(PLANE_SHAPE)
    source = NumpySource(array=plane)

    source.shape = VOLUME_SHAPE

    assert plane.shape == PLANE_SHAPE
