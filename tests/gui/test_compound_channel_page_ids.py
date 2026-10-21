"""
The params of a channel find its page by the ids the page holds: one for a channel, those of its parts, in order, for a
compound channel (e.g. a colocalization pair, which has no id of its own in the registry of the sample).
"""
import pytest

from ClearMap.Utils.event_bus import EventBus
from ClearMap.Utils.exceptions import ClearMapValueError
from ClearMap.config.channel import ChannelId
from ClearMap.gui import params_interfaces
from ClearMap.gui.params_interfaces import ChannelUiParameter
from ClearMap.gui.widgets import ChannelPageId, channel_page_id_of, set_channel_page_id

PAIR = 'fos-test'


class FakeWidget:
    def __init__(self):
        self._properties = {}

    def setProperty(self, name, value):
        self._properties[name] = value

    def property(self, name):
        return self._properties.get(name)

    def __getattr__(self, name):
        return lambda *args, **kwargs: None


class FakeChannelsTabWidget(FakeWidget):
    def __init__(self, names):
        super().__init__()
        self.pages = {name: FakeWidget() for name in names}

    def get_channel_widget(self, name, return_idx=False):
        page = self.pages.get(name)
        index = list(self.pages).index(name) if page is not None else -1
        return (page, index) if return_idx else page

    def get_channel_widget_by_page_id(self, page_id, return_idx=False):
        for index, page in enumerate(self.pages.values()):
            if channel_page_id_of(page) == page_id:
                return (page, index) if return_idx else page
        return (None, -1) if return_idx else None


class FakeTab(FakeWidget):
    def __init__(self, names):
        super().__init__()
        self.channelsParamsTabWidget = FakeChannelsTabWidget(names)


class FakeWidgetOps:
    def get(self, widget):
        return None

    def set(self, widget, value, silent=False):
        pass

    def connect(self, widget, callback):
        return None


class MinimalChannelParams(ChannelUiParameter):
    def __init__(self, tab, channel_name, *, event_bus):
        super().__init__(tab, channel_name, event_bus=event_bus)

    @property
    def cfg_subtree(self):
        return ['minimal', 'channels', self.name]

    def build_params_dict(self):
        return {}


@pytest.fixture(autouse=True)
def no_qt(monkeypatch):
    monkeypatch.setattr(params_interfaces, 'WIDGET_OPS', FakeWidgetOps())
    monkeypatch.setattr(params_interfaces, '_is_alive', lambda widget: widget is not None)
    monkeypatch.setattr(params_interfaces, 'QFrame', FakeWidget)


def test_the_params_of_a_pair_find_its_page_by_the_ids_of_its_parts():
    tab = FakeTab(['fos', PAIR])
    set_channel_page_id(tab.channelsParamsTabWidget.pages['fos'], ChannelPageId((ChannelId.new(),)))
    set_channel_page_id(tab.channelsParamsTabWidget.pages[PAIR], ChannelPageId((ChannelId.new(), ChannelId.new())))

    params = MinimalChannelParams(tab, PAIR, event_bus=EventBus())

    assert params.page is tab.channelsParamsTabWidget.pages[PAIR]
    assert params.page_index == 1


def test_the_params_of_a_page_without_ids_are_refused():
    tab = FakeTab([PAIR])

    with pytest.raises(ClearMapValueError, match='no channel id'):
        MinimalChannelParams(tab, PAIR, event_bus=EventBus())
