"""
A landmark is written to the landmark files once it is placed on both images. is_set tested the truth of the
coordinates: NaN (not placed) is true, so a landmark placed on one image only was written as "nan nan nan", and 0 is
false, so a landmark on the first plane of an axis was dropped.
"""
import numpy as np

from ClearMap.gui.widgets import Landmark

NOT_PLACED = (np.nan, np.nan, np.nan)


class FakeButton:
    def setParent(self, parent):
        pass

    def deleteLater(self):
        pass


def landmark_at(fixed_image_coords, moving_image_coords):
    landmark = object.__new__(Landmark)  # Without its dialog
    landmark.button = landmark.color_btn = FakeButton()
    landmark.coords = {'fixed_image': fixed_image_coords, 'moving_image': moving_image_coords}
    return landmark


def test_a_landmark_placed_on_both_images_is_set():
    assert landmark_at((1, 2, 3), (4, 5, 6)).is_set()


def test_a_landmark_placed_on_one_image_only_is_not_set():
    assert not landmark_at((1, 2, 3), NOT_PLACED).is_set()
    assert not landmark_at(NOT_PLACED, (4, 5, 6)).is_set()


def test_a_landmark_on_the_first_plane_of_an_axis_is_set():
    assert landmark_at((0, 2, 3), (4, 0, 6)).is_set()
