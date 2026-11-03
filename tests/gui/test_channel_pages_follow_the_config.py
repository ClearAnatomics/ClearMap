"""
The channel pages of a tab follow the channels of the config, from the registry and not from the payload of an event:
the pages of the channels which are gone are dropped, those of the new channels are added, the page of a renamed
channel is made again under its new name (in the same place), and none of it writes the config.
"""
import pytest

from ClearMap.Utils.exceptions import UnknownChannelError
from ClearMap.config.channel import ChannelRegistry
from ClearMap.config.compound_keys import PairKey
from ClearMap.gui import tabs_interfaces
from ClearMap.gui.params_interfaces import ChannelsUiParameterCollection
from ClearMap.gui.tabs import ColocalizationTab
from ClearMap.gui.tabs_interfaces import ExperimentTab

FIRST, SECOND, THIRD, NEW_NAME = 'fos', 'auto', 'nuclei', 'cfos'


class FakeChannelsTabWidget:
    """The tab widget with one page per channel: the pages are found by their tab text, like the real one"""

    def __init__(self):
        self.pages: list[tuple[str, object]] = []  # (tab text, page)
        self.current_page = None

    def get_channels_names(self):
        return [name for name, _ in self.pages]

    def get_channel_widget(self, name=None, return_idx=False):
        for index, (page_name, page) in enumerate(self.pages):
            if page_name == name:
                return (page, index) if return_idx else page
        return (None, -1) if return_idx else None

    def add_page(self, name):
        page = FakePage()  # No id: the reconcile gives it, from the registry
        self.pages.append((name, page))
        self.current_page = page
        return page

    def remove_channel_widget(self, name):
        page, index = self.get_channel_widget(name, return_idx=True)
        if page is not None:
            del self.pages[index]
            if self.current_page is page:
                self.current_page = self.pages[-1][1] if self.pages else None

    def move_channel_widget(self, name, index):
        page, current_index = self.get_channel_widget(name, return_idx=True)
        if page is not None:
            self.pages.insert(index, self.pages.pop(current_index))

    def current_channel(self):
        for name, page in self.pages:
            if page is self.current_page:
                return name
        return ''

    def setCurrentWidget(self, page):
        self.current_page = page


class FakePage:
    """A page: the Qt property API only (the page holds the id of its channel)"""

    def __init__(self):
        self._properties = {}

    def setProperty(self, name, value):
        self._properties[name] = value

    def property(self, name):
        return self._properties.get(name)


class FakeChannelParams:
    def teardown(self):
        pass


class FakeParams:
    """The real drop_channel, on the minimum it needs"""
    drop_channel = ChannelsUiParameterCollection.drop_channel

    def __init__(self, tab, config_writes):
        self.tab = tab
        self.channel_params = {}
        self._config_writes = config_writes

    def _emit_patch(self, *args):
        self._config_writes.append(args)

    def __bool__(self):
        return True


class FakeUi:
    def __init__(self):
        self.channelsParamsTabWidget = FakeChannelsTabWidget()


class FakeSampleManager:
    def __init__(self):
        self.channel_registry = ChannelRegistry()


class PageTab:
    """A tab with the real reconcile and the minimum around it"""
    name = 'fake tab'
    reconcile_channel_pages = ExperimentTab.reconcile_channel_pages
    _reconcile_channel_pages_once = ExperimentTab._reconcile_channel_pages_once
    _reconciling = ExperimentTab._reconciling
    _reconcile_again = ExperimentTab._reconcile_again
    _desired_channel_pages = ExperimentTab._desired_channel_pages
    _drop_channel_page = ExperimentTab._drop_channel_page
    _rename_channel_page = ExperimentTab._rename_channel_page
    _channel_registry = ExperimentTab._channel_registry
    _assign_channel_page_id = ExperimentTab._assign_channel_page_id

    def __init__(self, sample_manager, config_writes):
        self._sample_manager = sample_manager
        self.sample_manager = sample_manager
        self.ui = FakeUi()
        self.params = FakeParams(self.ui, config_writes)
        self._page_names = {}
        self.channels_in_config = []
        self.added, self.removed, self.reconciled = [], [], []
        self.advanced_applied_with_pages = []  # The pages there were each time the Advanced checkbox was applied

    def _get_channels(self):
        return list(self.channels_in_config)

    def add_channel_tab(self, name):
        self.ui.channelsParamsTabWidget.add_page(name)
        self.params.channel_params[name] = FakeChannelParams()
        self.added.append(name)

    def _on_channel_removed(self, name):
        self.removed.append(name)

    def _after_channels_reconciled(self, desired_channels):
        self.reconciled.append(desired_channels)

    def _setup_workers(self):
        pass

    def handle_advanced_checked(self):
        self.advanced_applied_with_pages.append(self.widget.get_channels_names())

    @property
    def widget(self):
        return self.ui.channelsParamsTabWidget

    def set_channels(self, *names):
        self.channels_in_config = list(names)


@pytest.fixture(autouse=True)
def fake_widget_class(monkeypatch):
    monkeypatch.setattr(tabs_interfaces, 'ExtendableTabWidget', FakeChannelsTabWidget)  # The tab is no Qt one


@pytest.fixture()
def config_writes():
    return []


@pytest.fixture()
def tab(config_writes):
    return PageTab(FakeSampleManager(), config_writes)


@pytest.fixture()
def registry(tab):
    return tab.sample_manager.channel_registry


@pytest.fixture()
def three_pages(tab, registry):
    channels = [registry.add(name) for name in (FIRST, SECOND, THIRD)]
    tab.set_channels(FIRST, SECOND, THIRD)
    tab.reconcile_channel_pages()
    return channels


def test_a_new_channel_gets_its_page(tab, registry):
    registry.add(FIRST)
    tab.set_channels(FIRST)

    tab.reconcile_channel_pages()

    assert tab.widget.get_channels_names() == [FIRST]
    assert FIRST in tab.params.channel_params


def test_a_removed_channel_loses_its_page_and_its_params_and_the_config_is_not_written(
        tab, registry, config_writes, three_pages):
    registry.remove(three_pages[1])
    tab.set_channels(FIRST, THIRD)

    tab.reconcile_channel_pages()

    assert tab.widget.get_channels_names() == [FIRST, THIRD]
    assert set(tab.params.channel_params) == {FIRST, THIRD}
    assert tab.removed == [SECOND]
    assert config_writes == []


def test_a_renamed_channel_has_a_page_under_its_new_name_in_the_same_place(tab, registry, three_pages):
    registry.rename_many({three_pages[1]: NEW_NAME})
    tab.set_channels(FIRST, NEW_NAME, THIRD)
    first_page = tab.widget.get_channel_widget(FIRST)

    tab.reconcile_channel_pages()

    assert tab.widget.get_channels_names() == [FIRST, NEW_NAME, THIRD]
    assert set(tab.params.channel_params) == {FIRST, NEW_NAME, THIRD}
    assert tab.widget.get_channel_widget(FIRST) is first_page  # The others are left alone


def test_the_selected_page_stays_selected_when_another_channel_is_renamed(tab, registry, three_pages):
    tab.widget.setCurrentWidget(tab.widget.get_channel_widget(THIRD))
    registry.rename_many({three_pages[0]: NEW_NAME})
    tab.set_channels(NEW_NAME, SECOND, THIRD)

    tab.reconcile_channel_pages()

    assert tab.widget.current_channel() == THIRD


def test_the_selected_page_stays_selected_when_it_is_the_one_renamed(tab, registry, three_pages):
    tab.widget.setCurrentWidget(tab.widget.get_channel_widget(SECOND))
    registry.rename_many({three_pages[1]: NEW_NAME})
    tab.set_channels(FIRST, NEW_NAME, THIRD)

    tab.reconcile_channel_pages()

    assert tab.widget.current_channel() == NEW_NAME


def test_a_page_already_renamed_by_the_user_in_that_page_is_not_made_again(tab, registry, three_pages):
    registry.rename_many({three_pages[1]: NEW_NAME})
    tab.set_channels(FIRST, NEW_NAME, THIRD)
    _, index = tab.widget.get_channel_widget(SECOND, return_idx=True)
    page = tab.widget.get_channel_widget(SECOND)
    tab.widget.pages[index] = (NEW_NAME, page)  # The page of the sample tab sets its own tab text first
    tab.params.channel_params[NEW_NAME] = tab.params.channel_params.pop(SECOND)
    tab.added.clear()

    tab.reconcile_channel_pages()

    assert tab.widget.get_channel_widget(NEW_NAME) is page
    assert tab.added == []


def test_channels_which_swap_their_names_have_the_pages_of_their_new_names(tab, registry, three_pages):
    registry.rename_many({three_pages[0]: SECOND, three_pages[1]: FIRST})
    tab.set_channels(SECOND, FIRST, THIRD)

    tab.reconcile_channel_pages()

    assert sorted(tab.widget.get_channels_names()) == sorted([FIRST, SECOND, THIRD])
    assert set(tab.params.channel_params) == {FIRST, SECOND, THIRD}


def test_reconciling_again_changes_nothing(tab, three_pages):
    pages_before = list(tab.widget.pages)
    tab.added.clear()
    tab.reconciled.clear()

    tab.reconcile_channel_pages()

    assert tab.widget.pages == pages_before
    assert tab.added == [] and tab.reconciled == []


def test_a_channel_which_is_not_in_the_registry_is_refused(tab, registry):
    tab.set_channels('not_a_channel')

    with pytest.raises(UnknownChannelError):
        tab.reconcile_channel_pages()


def test_drop_channel_of_a_channel_without_params_does_nothing(tab):
    assert tab.params.drop_channel('not_a_channel') is None


class PairTab(PageTab):
    _desired_channel_pages = ColocalizationTab._desired_channel_pages
    pairs_oriented = False

    def __init__(self, *args):
        super().__init__(*args)
        self.pairs = []

    def complete_colocalization_pairs(self):
        return list(self.pairs)

    def _get_channels(self):
        return self.pairs


def test_the_page_of_a_pair_follows_the_rename_of_one_of_its_channels(config_writes):
    sample_manager = FakeSampleManager()
    pair_tab = PairTab(sample_manager, config_writes)
    sample_manager.complete_colocalization_pairs = pair_tab.complete_colocalization_pairs
    registry = sample_manager.channel_registry
    first, second = registry.add(FIRST), registry.add(SECOND)
    pair_tab.pairs = [(FIRST, SECOND)]
    pair_tab.reconcile_channel_pages()
    assert pair_tab.widget.get_channels_names() == [str(PairKey(FIRST, SECOND, oriented=False))]

    registry.rename_many({first: NEW_NAME})
    pair_tab.pairs = [(NEW_NAME, SECOND)]
    pair_tab.reconcile_channel_pages()

    assert pair_tab.widget.get_channels_names() == [str(PairKey(NEW_NAME, SECOND, oriented=False))]
    assert set(pair_tab.params.channel_params) == {str(PairKey(NEW_NAME, SECOND, oriented=False))}


def test_the_advanced_checkbox_is_applied_once_the_new_pages_exist(tab, registry):
    registry.add(FIRST)
    registry.add(SECOND)
    tab.set_channels(FIRST, SECOND)

    tab.reconcile_channel_pages()

    assert tab.advanced_applied_with_pages == [[FIRST, SECOND]]  # Once, when all the pages are there


def test_the_advanced_checkbox_is_not_applied_again_when_no_page_changed(tab, three_pages):
    tab.advanced_applied_with_pages.clear()

    tab.reconcile_channel_pages()

    assert tab.advanced_applied_with_pages == []
