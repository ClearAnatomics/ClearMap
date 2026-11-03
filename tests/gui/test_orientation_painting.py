"""Painting the orientation from the config moves the three spin boxes: none of their handlers has anything to do"""
import contextlib

import pytest

from ClearMap.gui import params
from ClearMap.gui.params import SampleChannelParameters


class FakeSpinBox:
    def __init__(self):
        self.blocked = False
        self.set_while_blocked = []

    def setValue(self, value):
        self.set_while_blocked.append((value, self.blocked))


@contextlib.contextmanager
def block(spin_box):
    spin_box.blocked = True
    try:
        yield
    finally:
        spin_box.blocked = False


def test_the_orientation_is_painted_with_the_signals_of_the_spin_boxes_blocked(monkeypatch):
    monkeypatch.setattr(params, 'QSignalBlocker', block)
    param = object.__new__(SampleChannelParameters)
    param._channel_name = 'nuclei'
    param.tab = type('Tab', (), {'orientXSpinBox': FakeSpinBox(), 'orientYSpinBox': FakeSpinBox(),
                                 'orientZSpinBox': FakeSpinBox()})()

    param.orientation = (1, 2, 3)

    spins = (param.tab.orientXSpinBox, param.tab.orientYSpinBox, param.tab.orientZSpinBox)
    assert [spin.set_while_blocked for spin in spins] == [[(1, True)], [(2, True)], [(3, True)]]
    assert not any(spin.blocked for spin in spins)
