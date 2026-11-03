"""
Checking or unchecking "use existing layout" on the stitching page of a channel which computes its layout (its own
layout channel) writes it to the config. Its relevance test read the layout channel at a path the view of the page
does not have (stitching.layout_channel instead of layout_channel), so the edit was always dropped: once a layout
was placed (the stitching then sets use_existing_layout), it could not be computed again from the GUI.
"""
import pytest

from ClearMap.Utils.event_bus import EventBus
from ClearMap.config.channel import ChannelId
from ClearMap.gui import params_interfaces
from ClearMap.gui.params import GeneralChannelStitchingParams
from ClearMap.gui.widgets import ChannelPageId, set_channel_page_id

CHANNEL = 'auto'


class FakeWidget:
    """Any widget: answers neutral values to any Qt call"""

    def __init__(self, value=None):
        self.value = value

    def __getattr__(self, name):
        return FakeWidget()

    def __call__(self, *args, **kwargs):
        return None


class FakeStitchingPage(FakeWidget):
    def __init__(self):
        super().__init__()
        self._properties = {}
        self.useExistingLayoutCheckBox = FakeWidget(value=True)

    def setProperty(self, name, value):
        self._properties[name] = value

    def property(self, name):
        return self._properties.get(name)


class FakeChannelsTabWidget(FakeWidget):
    def __init__(self):
        super().__init__()
        self.page = FakeStitchingPage()
        set_channel_page_id(self.page, ChannelPageId((ChannelId.new(),)))

    def get_channel_widget(self, channel_name, return_idx=False):
        return (self.page, 0) if return_idx else self.page

    def get_channel_widget_by_page_id(self, page_id, return_idx=False):
        return (self.page, 0) if return_idx else self.page


class FakeTab(FakeWidget):
    def __init__(self):
        super().__init__()
        self.channelsParamsTabWidget = FakeChannelsTabWidget()


class FakeWidgetOps:
    """The value of a widget is its value attribute"""

    def get(self, widget):
        return getattr(widget, 'value', None)

    def set(self, widget, value, silent=False):
        widget.value = value

    def connect(self, widget, callback):
        return None


def _is_fake_widget_alive(widget):
    return widget is not None


@pytest.fixture
def patches(monkeypatch):
    monkeypatch.setattr(params_interfaces, 'WIDGET_OPS', FakeWidgetOps())
    monkeypatch.setattr(params_interfaces, '_is_alive', _is_fake_widget_alive)
    return []


def _stitching_page_params(layout_channel, patches):
    config = {'stitching': {'channels': {CHANNEL: {'use_npy': False, 'run': True, 'layout_channel': layout_channel,
                                                   'use_existing_layout': True}}}}

    def view():
        return config
    return GeneralChannelStitchingParams(FakeTab(), CHANNEL, event_bus=EventBus(), get_view=view,
                                         apply_patch=patches.append)


def test_unchecking_use_existing_layout_on_a_layout_channel_is_written(patches):
    page_params = _stitching_page_params(layout_channel=CHANNEL, patches=patches)
    page_params.tab.useExistingLayoutCheckBox.value = False

    page_params.handle_widget_changed(attr_name='use_existing_layout')

    assert {'stitching': {'channels': {CHANNEL: {'use_existing_layout': False}}}} in patches


def test_use_existing_layout_is_not_written_for_a_channel_using_the_layout_of_another(patches):
    page_params = _stitching_page_params(layout_channel='other', patches=patches)

    page_params.handle_widget_changed(attr_name='use_existing_layout')

    assert patches == []
