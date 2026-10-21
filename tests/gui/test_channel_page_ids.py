"""
A channel page holds the texts of the ChannelIds of its channel as a Qt property: one id for a channel, those of its
parts, in order, for a compound channel (e.g. a pair). The page is found by those ids, and they are recreated from the
property (they are checked again then, so a property which is not made of ids is refused).
"""
import pytest

from ClearMap.Utils.exceptions import ClearMapValueError, InvalidChannelIdError
from ClearMap.config.channel import ChannelId
from ClearMap.gui.widgets import (CHANNEL_PAGE_ID_PROPERTY, ChannelPageId, ExtendableTabWidget, channel_page_id_of,
                                  set_channel_page_id)


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

    set_channel_page_id(page, ChannelPageId((channel_id,)))

    assert channel_page_id_of(page) == ChannelPageId((channel_id,))


def test_a_compound_page_gives_back_the_ids_of_its_parts_in_order():
    page = FakePage()
    fos_id, auto_id = ChannelId.new(), ChannelId.new()

    set_channel_page_id(page, ChannelPageId((fos_id, auto_id)))

    assert channel_page_id_of(page) == ChannelPageId((fos_id, auto_id))


def test_the_property_holds_the_text_of_the_id():
    page = FakePage()
    channel_id = ChannelId.new()

    set_channel_page_id(page, ChannelPageId((channel_id,)))

    assert page.property(CHANNEL_PAGE_ID_PROPERTY) == channel_id.value


def test_a_page_without_an_id_has_none():
    assert channel_page_id_of(FakePage()) is None


def test_a_property_which_is_not_an_id_is_refused_when_read():
    page = FakePage()
    page.setProperty(CHANNEL_PAGE_ID_PROPERTY, 'auto')  # e.g. a name written where an id is expected

    with pytest.raises(InvalidChannelIdError):
        channel_page_id_of(page)


def test_the_page_of_an_id_is_found_whatever_its_tab_text_or_position():
    fos, auto = FakePage(), FakePage()
    fos_id, auto_id = ChannelId.new(), ChannelId.new()
    set_channel_page_id(fos, ChannelPageId((fos_id,)))
    set_channel_page_id(auto, ChannelPageId((auto_id,)))
    tab_widget = _tab_widget_with(('fos', fos), ('auto', auto))
    tab_widget.pages[1] = ('nuclear', auto)  # the tab text was changed by someone else: the id still finds the page

    page, index = ExtendableTabWidget.get_channel_widget_by_page_id(tab_widget, ChannelPageId((auto_id,)),
                                                                    return_idx=True)

    assert page is auto and index == 1


def test_an_id_with_no_page_gives_none():
    page = FakePage()
    set_channel_page_id(page, ChannelPageId((ChannelId.new(),)))
    tab_widget = _tab_widget_with(('auto', page))

    unknown = ChannelPageId((ChannelId.new(),))
    assert ExtendableTabWidget.get_channel_widget_by_page_id(tab_widget, unknown, return_idx=True) == (None, -1)
    assert ExtendableTabWidget.get_channel_widget_by_page_id(tab_widget, ChannelPageId((ChannelId.new(),))) is None


def test_the_page_of_a_pair_is_not_the_page_of_its_reversed_pair():
    fos_id, auto_id = ChannelId.new(), ChannelId.new()
    fos_auto, auto_fos = FakePage(), FakePage()
    set_channel_page_id(fos_auto, ChannelPageId((fos_id, auto_id)))
    set_channel_page_id(auto_fos, ChannelPageId((auto_id, fos_id)))
    tab_widget = _tab_widget_with(('fos-auto', fos_auto), ('auto-fos', auto_fos))

    assert ExtendableTabWidget.get_channel_widget_by_page_id(tab_widget, ChannelPageId((auto_id, fos_id))) is auto_fos


# ------------------------------------------------------------------------------------ a page made before its channel is known

from ClearMap.config.channel import ChannelRegistry  # noqa: E402
from ClearMap.gui.widgets import assign_channel_page_id_if_known  # noqa: E402


def test_a_page_made_before_its_channel_is_known_gets_the_id_once_it_is():
    registry = ChannelRegistry()
    page = FakePage()  # A new channel tab: named before the config is committed, so the registry does not know it yet

    assign_channel_page_id_if_known(page, 'Channel_0', registry)
    assert channel_page_id_of(page) is None

    channel = registry.add('Channel_0')  # The commit of the channel: the registry knows it now
    assign_channel_page_id_if_known(page, 'Channel_0', registry)

    assert channel_page_id_of(page) == ChannelPageId((channel.id,))


def test_a_page_keeps_its_id_when_it_is_given_again():
    registry = ChannelRegistry()
    channel = registry.add('auto')
    page = FakePage()
    assign_channel_page_id_if_known(page, 'auto', registry)

    assign_channel_page_id_if_known(page, 'auto', registry)

    assert channel_page_id_of(page) == ChannelPageId((channel.id,))


def test_a_compound_page_gets_the_ids_of_its_parts_once_both_are_known():
    registry = ChannelRegistry()
    fos = registry.add('fos')
    page = FakePage()

    assign_channel_page_id_if_known(page, 'fos-auto', registry)
    assert channel_page_id_of(page) is None

    auto = registry.add('auto')
    assign_channel_page_id_if_known(page, 'fos-auto', registry)

    assert channel_page_id_of(page) == ChannelPageId((fos.id, auto.id))


# ------------------------------------------------------------------------------------ the id of a page itself

def test_the_page_id_of_a_compound_channel_is_the_ids_of_its_parts_in_order():
    registry = ChannelRegistry()
    fos, auto = registry.add('fos'), registry.add('auto')

    assert ChannelPageId.of_channel('auto-fos', registry) == ChannelPageId((auto.id, fos.id))
    assert ChannelPageId.of_channel('fos', registry) == ChannelPageId((fos.id,))


def test_a_page_id_is_not_given_while_a_part_is_unknown():
    registry = ChannelRegistry()
    registry.add('fos')

    assert ChannelPageId.of_channel('fos-auto', registry) is None


def test_a_page_id_is_known_while_its_channel_and_every_part_are():
    registry = ChannelRegistry()
    fos, auto = registry.add('fos'), registry.add('auto')
    page_id = ChannelPageId.of_channel('fos-auto', registry)
    assert page_id.is_known_by(registry)

    registry.remove(auto)

    assert not page_id.is_known_by(registry)


@pytest.mark.parametrize('channel_ids', [(), ('fos',), None])
def test_a_page_id_is_made_of_channel_ids_only(channel_ids):
    with pytest.raises(ClearMapValueError):
        ChannelPageId(channel_ids)


def test_a_page_id_goes_through_its_text_unchanged():
    page_id = ChannelPageId((ChannelId.new(), ChannelId.new()))

    assert ChannelPageId.from_text(page_id.text) == page_id
