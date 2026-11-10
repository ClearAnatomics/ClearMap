"""
The random blobs of the colocalization tests (``random_channel``) include cubes, made with the scikit-image footprint
that is not deprecated: ``skimage.morphology.cube`` is deprecated since scikit-image 0.25 and removed in 0.27, where
the cubes would raise an AttributeError.
"""
import warnings

import numpy as np
import skimage.morphology

from ClearMap.Analysis.colocalization import colocalization_channel_mask_components as components

CUBE_DICE = 0  # The roll of _random_shape that makes a cube
RADIUS = 4


def _roll_a_cube(*_, **__):
    return CUBE_DICE


def test_a_random_cube_is_made_without_a_deprecated_scikit_image_footprint(monkeypatch):
    monkeypatch.setattr(np.random, 'randint', _roll_a_cube)
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        blob, center = components._random_shape(RADIUS)

    assert blob.ndim == 3 and blob.any()
    assert blob[tuple(center)]


def test_a_random_cube_is_still_made_by_a_scikit_image_older_than_0_25(monkeypatch):
    monkeypatch.setattr(np.random, 'randint', _roll_a_cube)
    monkeypatch.delattr(skimage.morphology, 'footprint_rectangle')  # As in scikit-image < 0.25
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', FutureWarning)  # cube is deprecated where it still exists
        blob, center = components._random_shape(RADIUS)

    assert blob[tuple(center)]
