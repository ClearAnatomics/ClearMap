"""
The reference channel of the registration (registration.reference_channel) is chosen in a combo box of the
registration tab: not chosen yet (the token itself), None (no channel is aligned with the atlas) or a channel.
"""
import pytest

from ClearMap.Utils.event_bus import EventBus
from ClearMap.config.registration_channels import NO_REFERENCE, REFERENCE_UNDECIDED
from ClearMap.gui import params_interfaces
from ClearMap.gui.params import NO_PARTNER_TEXT, ReferenceChannelParams, reference_channel_choices


class FakeComboBox:
    def objectName(self):
        return 'referenceChannelComboBox'

    def parentWidget(self):
        return None


class FakeTab:
    def __init__(self):
        self.referenceChannelComboBox = FakeComboBox()


class FakeWidgetOps:
    """The value each widget shows, and the callback each was connected to"""

    def __init__(self):
        self.values, self.callbacks = {}, {}

    def get(self, widget):
        return self.values.get(id(widget))

    def set(self, widget, value, silent=False):
        self.values[id(widget)] = value

    def connect(self, widget, callback):
        self.callbacks[id(widget)] = callback
        return None


@pytest.fixture()
def widget_ops(monkeypatch):
    ops = FakeWidgetOps()
    monkeypatch.setattr(params_interfaces, 'WIDGET_OPS', ops)
    monkeypatch.setattr(params_interfaces, '_is_alive', lambda widget: widget is not None)
    return ops


def _params(reference_channel, patches):
    config = {'registration': {'reference_channel': reference_channel, 'channels': {}}}
    return ReferenceChannelParams(FakeTab(), event_bus=EventBus(), get_view=lambda: config,
                                  apply_patch=patches.append)


def test_the_choices_are_texts_for_each_state():
    assert reference_channel_choices(['dapi', 'fos']) == [REFERENCE_UNDECIDED, NO_PARTNER_TEXT, 'dapi', 'fos']


@pytest.mark.parametrize('reference_channel, shown', [('dapi', 'dapi'), (REFERENCE_UNDECIDED, REFERENCE_UNDECIDED),
                                                      (NO_REFERENCE, NO_PARTNER_TEXT)])
def test_the_combo_box_shows_the_reference_channel_of_the_config(widget_ops, reference_channel, shown):
    params = _params(reference_channel, [])

    params.cfg_to_ui()

    assert widget_ops.get(params.tab.referenceChannelComboBox) == shown


@pytest.mark.parametrize('picked, written', [('dapi', 'dapi'), (None, NO_REFERENCE)])  # The getter gives None for 'None'
def test_a_reference_picked_is_written_to_the_registration(widget_ops, picked, written):
    patches = []
    params = _params(REFERENCE_UNDECIDED, patches)
    combo_box = params.tab.referenceChannelComboBox
    widget_ops.set(combo_box, picked)

    widget_ops.callbacks[id(combo_box)]()

    assert patches == [{'registration': {'reference_channel': written}}]
