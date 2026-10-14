"""
The open risks of the channel pages (see the notes of 0008). Each risk is pinned by a test:

- a plain test is a behaviour we keep on purpose (it passes today);
- an ``xfail(strict=True)`` test is a known bug: it fails today, and it must be turned into a plain test when the bug
  is fixed (strict: an unexpected pass is a failure, so the marker cannot be left behind).

Run: ``pytest tests/gui/test_channel_pages_open_risks.py -v``. Expected: N passed, M xfailed, 0 failed.
"""
import pytest

from ClearMap.config.channel import ChannelId, ChannelRegistry
from ClearMap.gui import params_interfaces, tabs_interfaces
from ClearMap.gui.params import SampleParameters
from ClearMap.gui.widgets import channel_id_of, set_channel_id
from ClearMap.gui.params_interfaces import ChannelsUiParameterCollection, ChannelUiParameter
from ClearMap.gui.tabs_interfaces import ExperimentTab
from ClearMap.Utils.utilities import DELETE

FIRST, SECOND, THIRD = 'fos', 'auto', 'nuclei'


class FakePage:
    """A page: the Qt property API only (the page holds the id of its channel)"""

    def __init__(self):
        self._properties = {}

    def setProperty(self, name, value):
        self._properties[name] = value

    def property(self, name):
        return self._properties.get(name)


class FakeChannelsTabWidget:
    """The tab widget: one page per channel, found by tab text or by the id of the channel (as the real one)"""

    def __init__(self):
        self.pages: list[tuple[str, object]] = []
        self.current_page = None

    def get_channels_names(self):
        return [name for name, _ in self.pages]

    def tabText(self, index):
        return self.pages[index][0]

    def get_channel_widget(self, name=None, return_idx=False):
        for index, (page_name, page) in enumerate(self.pages):
            if page_name == name:
                return (page, index) if return_idx else page
        return (None, -1) if return_idx else None

    def get_channel_widget_by_id(self, channel_id, return_idx=False):
        for index, (_, page) in enumerate(self.pages):
            if channel_id_of(page) == channel_id:
                return (page, index) if return_idx else page
        return (None, -1) if return_idx else None

    def add_page(self, name):
        page = FakePage()
        set_channel_id(page, ChannelId.new())  # As the sample tab does when it makes a page
        self.pages.append((name, page))
        self.current_page = page
        return page

    def remove_channel_widget(self, name):
        page, index = self.get_channel_widget(name, return_idx=True)
        if page is not None:
            del self.pages[index]

    def move_channel_widget(self, name, index):
        page, current_index = self.get_channel_widget(name, return_idx=True)
        if page is not None:
            self.pages.insert(index, self.pages.pop(current_index))

    def current_channel(self):
        return next((name for name, page in self.pages if page is self.current_page), '')

    def setCurrentWidget(self, page):
        self.current_page = page

    def indexOf(self, page):
        return next((index for index, (_, candidate) in enumerate(self.pages) if candidate is page), -1)

    def setTabText(self, index, text):
        _, page = self.pages[index]
        self.pages[index] = (text, page)


class FakeTab:
    def __init__(self):
        self.channelsParamsTabWidget = FakeChannelsTabWidget()


class FakeChannelParams:
    def __init__(self, name):
        self.name = name
        self.torn_down = False

    def teardown(self):
        self.torn_down = True


class FakeCollection:
    """A channels collection with the real ``drop_channel``, ``pop`` and rename, and a patch sink we can make fail"""
    drop_channel = ChannelsUiParameterCollection.drop_channel
    pop = ChannelsUiParameterCollection.pop
    handle_channel_rename_request = SampleParameters.handle_channel_rename_request
    _revert_rename = SampleParameters._revert_rename

    def __init__(self, *, patch_fails: bool = False):
        self.tab = FakeTab()
        self.channel_params: dict[str, FakeChannelParams] = {}
        self.patches: list[dict] = []
        self._apply_patch = self.patches.append
        self._patch_fails = patch_fails

    @property
    def channels(self):
        return list(self.channel_params)

    @property
    def config_channels(self):  # The config knows the channels of the params: a rename is written to both
        return list(self.channel_params)

    def _publish_channels_changed(self, names):
        pass

    def add(self, name):
        self.tab.channelsParamsTabWidget.add_page(name)
        self.channel_params[name] = FakeChannelParams(name)

    def _emit_patch(self, relative_path, value):
        if self._patch_fails:
            raise RuntimeError('config refused the removal')
        self.patches.append({'channels': {relative_path[1]: value}})


@pytest.fixture(autouse=True)
def fake_tab_widget_class(monkeypatch):
    monkeypatch.setattr(params_interfaces, 'ExtendableTabWidget', FakeChannelsTabWidget)
    monkeypatch.setattr(tabs_interfaces, 'ExtendableTabWidget', FakeChannelsTabWidget)  # The reconcile guards on it


# ------------------------------------------------------------------------------------------------- user removes a channel

def test_removing_a_channel_removes_its_page_and_params_and_sends_the_removal_once():
    collection = FakeCollection()
    collection.add(FIRST)
    collection.add(SECOND)

    collection.pop(FIRST)

    assert collection.tab.channelsParamsTabWidget.get_channels_names() == [SECOND]
    assert collection.channels == [SECOND]
    assert collection.patches == [{'channels': {FIRST: DELETE}}]


@pytest.mark.xfail(strict=True, reason='OPEN RISK 1: if the config refuses the removal, the page and the params are '
                                       'already gone: the next reconcile re-adds the channel. The page should be '
                                       'put back (or the removal not done) when the config refuses it.')
def test_a_refused_removal_leaves_the_page_of_the_channel():
    collection = FakeCollection(patch_fails=True)
    collection.add(FIRST)

    collection.pop(FIRST)

    assert collection.tab.channelsParamsTabWidget.get_channels_names() == [FIRST]
    assert FIRST in collection.channels


def test_a_refused_removal_warns_so_the_failure_is_not_silent():
    collection = FakeCollection(patch_fails=True)
    collection.add(FIRST)

    with pytest.warns(UserWarning, match='Could not remove'):
        collection.pop(FIRST)


def test_drop_channel_without_params_does_not_fail():
    collection = FakeCollection()

    assert collection.drop_channel('not_a_channel') is None


# ------------------------------------------------------------------------------------- page index read back by a page

class FakeNameTab(FakeChannelsTabWidget):
    pass


class SomeChannelParams(ChannelUiParameter):
    @property
    def cfg_subtree(self):
        return ['sample', 'channels', self.name]


def _param_for_channel(tab_widget: FakeNameTab, name: str):
    """Binds a channel parameter to the page of ``name``, as ``ChannelUiParameter.__init__`` does"""
    param = object.__new__(SomeChannelParams)  # Skips the Qt construction: only the id and the name are under test
    param._tab_widget = tab_widget
    param._channel_id = channel_id_of(tab_widget.get_channel_widget(name))
    param._channel_name = name
    param.nameWidget = tab_widget  # Not a line edit: the name setter then only moves the tab text
    return param


def test_a_page_reads_its_own_name_after_a_page_to_its_left_is_removed():
    tab_widget = FakeNameTab()
    tab_widget.add_page(FIRST)
    tab_widget.add_page(SECOND)
    param_of_second = _param_for_channel(tab_widget, SECOND)
    tab_widget.remove_channel_widget(FIRST)

    assert param_of_second.name == SECOND
    assert param_of_second.page_index == 0


def test_a_page_reads_its_own_name_before_any_removal():
    tab_widget = FakeNameTab()
    tab_widget.add_page(FIRST)
    tab_widget.add_page(SECOND)

    assert _param_for_channel(tab_widget, SECOND).name == SECOND


def test_renaming_a_page_after_a_page_to_its_left_is_removed_renames_that_page_only():
    # The user's case: remove a channel, then rename the one after it. The page was made at index 1 and sits at 0 now.
    collection = FakeCollection()
    collection.add(FIRST)
    collection.add(SECOND)
    collection.channel_params[SECOND] = _param_for_channel(collection.tab.channelsParamsTabWidget, SECOND)
    collection.channel_params[SECOND].handle_name_changed = (
        lambda old, new: collection.patches.append({'$rename': {'channels': {old: new}}}))
    collection.pop(FIRST)

    collection.handle_channel_rename_request(SECOND, THIRD, requester=collection.channel_params[SECOND])

    assert collection.tab.channelsParamsTabWidget.get_channels_names() == [THIRD]
    assert collection.channel_params[THIRD].name == THIRD
    assert collection.patches[-1] == {'$rename': {'channels': {SECOND: THIRD}}}


# ------------------------------------------------------------------------------- reconcile: what runs after a change

class PageTab(ExperimentTab):
    """The real reconcile, around a fake view: the parts it calls are recorded instead of done"""

    def __init__(self, sample_manager):
        self._sample_manager = sample_manager
        self._page_names = {}
        self.ui = type('Ui', (), {})()
        self.ui.channelsParamsTabWidget = FakeChannelsTabWidget()
        self.params = FakeCollection()
        self.params.tab = self.ui
        self.channels_in_config: list[str] = []
        self.after_calls: list[list[str]] = []
        self.workers_calls = 0
        self.name = 'fake tab'

    def _desired_channel_pages(self):
        registry = self._sample_manager.channel_registry
        return [(registry.resolve(name), name) for name in self.channels_in_config]

    def add_channel_tab(self, name):
        self.ui.channelsParamsTabWidget.add_page(name)
        self.params.channel_params[name] = FakeChannelParams(name)

    def _on_channel_removed(self, name):
        pass

    def _after_channels_reconciled(self, desired_channels):
        self.after_calls.append(list(desired_channels))

    def _setup_workers(self):
        self.workers_calls += 1


@pytest.fixture()
def registry():
    return ChannelRegistry()


@pytest.fixture()
def tab(registry):
    class SampleManagerStub:
        channel_registry = None
    manager = SampleManagerStub()
    manager.channel_registry = registry
    return PageTab(manager)


@pytest.mark.xfail(strict=True, reason='OPEN RISK 3: the follow-up work (combo boxes, run lists, workers) is skipped '
                                       'when nothing changed, but tabs refresh state there that is not about the '
                                       'page list (e.g. stitching _refresh_ui). It should run on every reconcile.')
def test_the_follow_up_work_runs_on_every_reconcile_even_without_a_change(registry, tab):
    registry.add(FIRST)
    tab.channels_in_config = [FIRST]
    tab.reconcile_channel_pages()
    tab.after_calls.clear()

    tab.reconcile_channel_pages()

    assert len(tab.after_calls) == 1


def test_a_reconcile_without_change_does_nothing_to_the_pages(registry, tab):
    registry.add(FIRST)
    tab.channels_in_config = [FIRST]
    tab.reconcile_channel_pages()
    page_before = tab.ui.channelsParamsTabWidget.get_channel_widget(FIRST)

    tab.reconcile_channel_pages()

    assert tab.ui.channelsParamsTabWidget.get_channel_widget(FIRST) is page_before


@pytest.mark.xfail(strict=True, reason='OPEN RISK 4: a channel removed and another added with the same name in one '
                                       'commit keep the old page: the page is not made again although its channel '
                                       'changed. It works today only because params are keyed by name; it must be '
                                       'made again (as on a rename) once the pages are keyed by channel.')
def test_a_channel_removed_and_re_added_under_the_same_name_gets_a_new_page(registry, tab):
    old = registry.add(FIRST)
    tab.channels_in_config = [FIRST]
    tab.reconcile_channel_pages()
    page_of_old_channel = tab.ui.channelsParamsTabWidget.get_channel_widget(FIRST)
    registry.remove(old)
    registry.add(FIRST)

    tab.reconcile_channel_pages()

    assert tab.ui.channelsParamsTabWidget.get_channel_widget(FIRST) is not page_of_old_channel


def test_a_channel_leaving_a_pipeline_loses_its_page(registry, tab):
    """Intended: a data type change that moves a channel out of a tab removes its page there (a behaviour change)"""
    registry.add(FIRST)
    registry.add(SECOND)
    tab.channels_in_config = [FIRST, SECOND]
    tab.reconcile_channel_pages()

    tab.channels_in_config = [FIRST]
    tab.reconcile_channel_pages()

    assert tab.ui.channelsParamsTabWidget.get_channels_names() == [FIRST]
