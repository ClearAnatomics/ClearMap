"""
A DataViewer draws the markers of a scatter on top of its slices: one item is added to its view, and the markers
are the ones given. The orchestrators and the GUI all go through ``overlay_scatter``, so they never import pyqtgraph.
"""
from types import SimpleNamespace

from ClearMap.Visualization.Qt.DataViewer import DataViewer


class FakeView:
    def __init__(self):
        self.items = []

    def addItem(self, item):
        self.items.append(item)


def test_the_scatter_is_one_item_of_the_view_showing_the_markers_given():
    viewer = SimpleNamespace(view=FakeView(), scatter=None, scatter_coords=None)
    markers = object()

    DataViewer.overlay_scatter(viewer, markers)

    assert viewer.view.items == [viewer.scatter] and viewer.scatter is not None
    assert viewer.scatter_coords is markers
