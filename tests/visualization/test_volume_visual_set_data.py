"""
``VolumeVisual.set_data()`` sends the volume to the texture as float32, scaled to [0, 1] by the contrast limits,
and leaves the array of the caller as it was.

The volume was converted with ``np.array(vol, dtype='float32', copy=False)`` and then scaled in place: NumPy 2
raises a ValueError when ``copy=False`` needs a copy (any volume that is not float32), and a float32 volume was not
copied, so it was scaled in the array of the caller.
"""
from types import SimpleNamespace

import numpy as np

from ClearMap.Visualization.Vispy.VolumeVisual import VolumeVisual

SHAPE = (2, 3, 4)
MAX_VALUE = 23  # The last value of np.arange over SHAPE


class _Texture:
    def __init__(self):
        self.data = None

    def set_data(self, data):
        self.data = data


def _visual():
    """The attributes set_data() uses, without a vispy canvas"""
    return SimpleNamespace(_clim=None, _tex=_Texture(), shared_program={}, _vol_shape=None,
                           _need_vertex_update=False)


def test_a_uint16_volume_is_sent_to_the_texture_as_float32_scaled_by_the_contrast_limits():
    visual = _visual()
    volume = np.arange(np.prod(SHAPE), dtype=np.uint16).reshape(SHAPE)

    VolumeVisual.set_data(visual, volume)

    assert visual._tex.data.dtype == np.float32
    np.testing.assert_allclose(visual._tex.data, volume / MAX_VALUE, rtol=1e-6)


def test_a_float32_volume_of_the_caller_is_not_scaled_in_place():
    visual = _visual()
    volume = np.arange(np.prod(SHAPE), dtype=np.float32).reshape(SHAPE)
    original = volume.copy()

    VolumeVisual.set_data(visual, volume)

    np.testing.assert_array_equal(volume, original)
