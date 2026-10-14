"""
A channel page holds the text of the ChannelId of its channel as a Qt property. The page is found by that id, and the
id is recreated from the property (it is checked again then, so a property which is not an id is refused).
"""
import pytest

from ClearMap.Utils.exceptions import InvalidChannelIdError
from ClearMap.config.channel import ChannelId
from ClearMap.gui.widgets import CHANNEL_ID_PROPERTY, ExtendableTabWidget, channel_id_of, set_channel_id


class FakePage:
    """A QWidget for the property API only"""

    def __init__(self):
        self._properties = {}

    def setProperty(self, name, value):
        self._properties[name] = value

    def property(self, name):
        return self._properties.get(name)


class FakeTabBar:
    """Only the tab widget's own lookups are under test: pages in order, with their tab text"""

    def __init__(self):
        self.pages: list[tuple[str, FakePage]] = []
        self.last_real_tab_idx = 0

    def widget(self, index):
        return self.pages[index][1]

    def tabText(self, index):
        return self.pages[index][0]

    def add(self, name, page):
        self.pages.append((name, page))
        self.last_real_tab_idx = len(self.pages)


def _tab_widget_with(*names_and_pages):
    tab_widget = FakeTabBar()
    for name, page in names_and_pages:
        tab_widget.add(name, page)
    return tab_widget


def test_a_page_given_an_id_gives_back_the_same_id():
    page = FakePage()
    channel_id = ChannelId.new()

    set_channel_id(page, channel_id)

    assert channel_id_of(page) == channel_id


def test_the_property_holds_the_text_of_the_id():
    page = FakePage()
    channel_id = ChannelId.new()

    set_channel_id(page, channel_id)

    assert page.property(CHANNEL_ID_PROPERTY) == channel_id.value


def test_a_page_without_an_id_has_none():
    assert channel_id_of(FakePage()) is None


def test_a_property_which_is_not_an_id_is_refused_when_read():
    page = FakePage()
    page.setProperty(CHANNEL_ID_PROPERTY, 'auto')  # e.g. a name written where an id is expected

    with pytest.raises(InvalidChannelIdError):
        channel_id_of(page)


def test_the_page_of_an_id_is_found_whatever_its_tab_text_or_position():
    fos, auto = FakePage(), FakePage()
    fos_id, auto_id = ChannelId.new(), ChannelId.new()
    set_channel_id(fos, fos_id)
    set_channel_id(auto, auto_id)
    tab_widget = _tab_widget_with(('fos', fos), ('auto', auto))
    tab_widget.pages[1] = ('nuclear', auto)  # the tab text was changed by someone else: the id still finds the page

    page, index = ExtendableTabWidget.get_channel_widget_by_id(tab_widget, auto_id, return_idx=True)

    assert page is auto and index == 1


def test_an_id_with_no_page_gives_none():
    page = FakePage()
    set_channel_id(page, ChannelId.new())
    tab_widget = _tab_widget_with(('auto', page))

    assert ExtendableTabWidget.get_channel_widget_by_id(tab_widget, ChannelId.new(), return_idx=True) == (None, -1)
    assert ExtendableTabWidget.get_channel_widget_by_id(tab_widget, ChannelId.new()) is None
