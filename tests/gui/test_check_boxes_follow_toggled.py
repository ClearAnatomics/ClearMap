"""
The check boxes act on the bool of ``toggled``, not on the int of ``stateChanged``.

With PyQt6, ``state == Qt.Checked`` compares an int with an enum and is always False: the layer boxes of the
DataViewer would hide their layer for good, and "select all" of the graph steps would give setCheckState an int,
which PyQt6 refuses (hypothesis). Qt is faked: the boxes record what is connected to their signals.
"""
from types import SimpleNamespace

import pytest

from ClearMap.Visualization.Qt import DataViewer as data_viewer_module
from ClearMap.gui import tabs

GRAPH_STEP_BOX_NAMES = ('buildGraphSkeletonizeCheckBox', 'buildGraphBuildCheckBox', 'buildGraphCleanCheckBox',
                        'buildGraphReduceCheckBox', 'buildGraphTransformCheckBox', 'buildGraphRegisterCheckBox')


class FakeSignal:
    def __init__(self):
        self.slots = []

    def connect(self, slot):
        self.slots.append(slot)

    def emit(self, *args):
        for slot in self.slots:
            slot(*args)


class FakeCheckBox:
    def __init__(self, *_, **__):
        self.toggled = FakeSignal()
        self.stateChanged = FakeSignal()
        self.checked = None

    def __getattr__(self, name):  # setMaximumWidth...: no effect here
        return _do_nothing

    def setChecked(self, checked):
        self.checked = checked


class FakeLayer:
    visible = True

    def setVisible(self, visible):
        self.visible = visible


def _do_nothing(*_, **__):
    pass


def _fake_widget(*_, **__):
    return SimpleNamespace(setLayout=_do_nothing, addWidget=_do_nothing, setMaximumWidth=_do_nothing,
                           clicked=FakeSignal(), setChecked=_do_nothing)


@pytest.fixture()
def two_layer_viewer(monkeypatch):
    """The axis tools of a DataViewer of two sources, built with fake widgets"""
    for name in ('QGridLayout', 'QWidget', 'QRadioButton'):
        monkeypatch.setattr(data_viewer_module, name, _fake_widget)
    monkeypatch.setattr(data_viewer_module, 'QCheckBox', FakeCheckBox)
    viewer = SimpleNamespace(axis_buttons=[], space_axes=(0, 1, 2), scroll_axis=2, n_sources=2,
                             setSliceAxis=_do_nothing, image_items=[FakeLayer(), FakeLayer()])
    viewer.toggle_layer = data_viewer_module.DataViewer.toggle_layer.__get__(viewer)
    data_viewer_module.DataViewer._DataViewer__setup_axes_controls(viewer)
    return viewer


def test_unticking_a_layer_box_hides_its_layer_and_ticking_it_shows_it(two_layer_viewer):
    second_layer_box = two_layer_viewer.axis_buttons[-1]

    second_layer_box.toggled.emit(False)
    assert [layer.visible for layer in two_layer_viewer.image_items] == [True, False]

    second_layer_box.toggled.emit(True)
    assert [layer.visible for layer in two_layer_viewer.image_items] == [True, True]


@pytest.mark.parametrize('checked', [True, False])
def test_select_all_ticks_or_unticks_every_graph_step(checked):
    ui = SimpleNamespace(**{name: FakeCheckBox() for name in GRAPH_STEP_BOX_NAMES})
    tab = SimpleNamespace(ui=ui)

    tabs.VasculatureTab._VasculatureTab__select_all_graph_steps(tab, checked)

    assert [getattr(ui, name).checked for name in GRAPH_STEP_BOX_NAMES] == [checked] * len(GRAPH_STEP_BOX_NAMES)
