"""
A click on a pipeline step chip toggles the step on its left zone and "keep intermediate" elsewhere, with PyQt5 and
PyQt6 mouse events.

The chip read the click with event.x(): Qt 6 deprecated it, and PyQt6 does not wrap it (AttributeError at each click).
Qt 6 events have position(); Qt 5 events have localPos() and no position(). The events are faked with only what each
Qt version has.
"""
from types import SimpleNamespace

import pytest

from ClearMap.gui import pipeline_widgets
from ClearMap.gui.pipeline_widgets import StepChip

LEFT_ZONE_X = 10
RIGHT_ZONE_X = 120


class Point:
    def __init__(self, x):
        self._x = x

    def x(self):
        return self._x


class Qt6MouseEvent:
    def __init__(self, x):
        self._x = x

    def button(self):
        return pipeline_widgets.Qt.MouseButton.LeftButton

    def position(self):
        return Point(self._x)


class Qt5MouseEvent:
    def __init__(self, x):
        self._x = x

    def button(self):
        return pipeline_widgets.Qt.MouseButton.LeftButton

    def localPos(self):
        return Point(self._x)


class RecordingSignal:
    def __init__(self):
        self.emitted = []

    def emit(self, *args):
        self.emitted.append(args)


def _do_nothing():
    pass


def _chip():
    step = SimpleNamespace(step_id='skeletonize', enabled=True, keep_intermediate=False)
    return SimpleNamespace(_step=step, _ICON_ZONE=StepChip._ICON_ZONE, update=_do_nothing,
                           toggled=RecordingSignal(), keep_changed=RecordingSignal())


@pytest.mark.parametrize('event_class', [Qt6MouseEvent, Qt5MouseEvent], ids=['qt6', 'qt5'])
def test_a_click_on_the_left_zone_toggles_the_step(event_class):
    chip = _chip()

    StepChip.mousePressEvent(chip, event_class(LEFT_ZONE_X))

    assert chip.toggled.emitted == [('skeletonize', False)] and chip.keep_changed.emitted == []


@pytest.mark.parametrize('event_class', [Qt6MouseEvent, Qt5MouseEvent], ids=['qt6', 'qt5'])
def test_a_click_elsewhere_toggles_keep_intermediate(event_class):
    chip = _chip()

    StepChip.mousePressEvent(chip, event_class(RIGHT_ZONE_X))

    assert chip.keep_changed.emitted == [('skeletonize', True)] and chip.toggled.emitted == []
