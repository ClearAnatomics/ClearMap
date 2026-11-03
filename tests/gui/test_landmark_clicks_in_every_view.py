"""
A click in a data viewer (where the landmarks of the registration are picked) emits the (x, y, z) of the source.
It emitted (horizontal, vertical, slice) of the view, which is (x, y, z) only in the view scrolling through z: in the
other views, the landmarks were saved with permuted coordinates. (Port to 3.1 of PR #195 by ziyin-zhang.)
"""
import numpy as np
import pytest

from ClearMap.Visualization.Qt.DataViewer import DataViewer

SOURCE_SHAPE = (50, 60, 70)
CLICKED_POSITION = (10, 20, 30)  # (x, y, z) of the source
LEFT_BUTTON = 1


class FakeSignal:
    def __init__(self):
        self.emitted = []

    def emit(self, *values):
        self.emitted.append(values)


class FakeMouseEvent:
    def __init__(self, scene_position):
        self.scene_position = scene_position

    def accept(self):
        pass

    def scenePos(self):
        return self.scene_position

    def button(self):
        return LEFT_BUTTON


class FakeDataViewer:
    handleMouseClick = DataViewer.handleMouseClick
    getXYAxes = DataViewer.getXYAxes
    scale_coords = DataViewer.scale_coords

    def __init__(self, scroll_axis):
        self.sources = [np.zeros(SOURCE_SHAPE)]
        self.source_shape = SOURCE_SHAPE
        self.source_scale = (1, 1, 1)
        self.scroll_axis = scroll_axis
        self.color_axis = None
        self.source_index = list(CLICKED_POSITION)  # The slice shown is that of the clicked position
        self.mouse_clicked = FakeSignal()

    def get_coords(self, scene_position):
        """The view shows the 2 axes other than the one it scrolls through: horizontal first"""
        return scene_position


@pytest.mark.parametrize('scroll_axis', [0, 1, 2])
def test_a_click_emits_the_position_in_the_source_whichever_axis_the_view_scrolls_through(scroll_axis):
    viewer = FakeDataViewer(scroll_axis)
    horizontal_axis, vertical_axis = viewer.getXYAxes()

    viewer.handleMouseClick(FakeMouseEvent((CLICKED_POSITION[horizontal_axis], CLICKED_POSITION[vertical_axis])))

    assert viewer.mouse_clicked.emitted == [CLICKED_POSITION]
