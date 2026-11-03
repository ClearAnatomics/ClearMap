"""
The open risks of the channel pages (see the notes of 0008). Each risk is pinned by a test:

- a plain test is a behaviour we keep on purpose (it passes today);
- an ``xfail(strict=True)`` test is a known bug: it fails today, and it must be turned into a plain test when the bug
  is fixed (strict: an unexpected pass is a failure, so the marker cannot be left behind).

Run: ``pytest tests/gui/test_channel_pages_open_risks.py -v``. Expected: N passed, M xfailed, 0 failed.
"""
import pytest

from ClearMap.IO.FileUtils import Rename
from ClearMap.IO.channel_file_renaming import ChannelRenamePlan, NotRenamed
from ClearMap.config.channel import ChannelId, ChannelRegistry
from ClearMap.config.compound_keys import compound_channel_parts
from ClearMap.gui import params, params_interfaces, tabs_interfaces
from ClearMap.gui.params import SampleParameters
from ClearMap.gui.widgets import ChannelPageId, channel_page_id_of, set_channel_page_id
from ClearMap.gui.params_interfaces import ChannelsUiParameterCollection, ChannelUiParameter
from ClearMap.gui.tabs import SampleInfoTab
from ClearMap.gui.tabs_interfaces import ExperimentTab
from ClearMap.Utils.exceptions import ChannelNameTakenError, ClearMapConfigError, ClearMapValueError
from ClearMap.config.config_adjusters.patch_ops import DELETE

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

    def get_channel_widget_by_page_id(self, page_id, return_idx=False):
        for index, (_, page) in enumerate(self.pages):
            if channel_page_id_of(page) == page_id:
                return (page, index) if return_idx else page
        return (None, -1) if return_idx else None

    def add_page(self, name, *, with_id=True):
        page = FakePage()
        if with_id:
            set_channel_page_id(page, ChannelPageId((ChannelId.new(),)))  # As the sample tab does when it makes a page
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

    def add_channel_widget(self, widget, name=''):
        self.pages.append((name, widget))
        self.current_page = widget
        return name

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


def plan_without_files(channel_renames):
    return ChannelRenamePlan()


class FakeCollection:
    """A channels collection with the real ``drop_channel``, ``pop`` and rename, and a patch sink we can make fail"""
    drop_channel = ChannelsUiParameterCollection.drop_channel
    pop = ChannelsUiParameterCollection.pop
    handle_channel_rename_request = SampleParameters.handle_channel_rename_request
    _revert_rename = SampleParameters._revert_rename
    _refuse_rename = SampleParameters._refuse_rename
    _vet_file_renames = SampleParameters._vet_file_renames
    _rename_files = SampleParameters._rename_files

    def __init__(self, *, patch_fails: bool = False):
        self.tab = FakeTab()
        self._plan_file_renames = plan_without_files
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


def test_a_refused_removal_tells_the_user_why_and_removes_nothing_from_the_config(monkeypatch):
    collection = FakeCollection(patch_fails=True)
    collection.add(FIRST)
    dialogs = []
    monkeypatch.setattr(params_interfaces, 'handle_exception',
                        lambda reason, parent=None, context='': dialogs.append((str(reason), parent, context)))

    assert collection.pop(FIRST) is None

    assert dialogs == [('config refused the removal', collection.tab, f"Remove '{FIRST}'")]
    assert collection.patches == []


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
    param._page_id = channel_page_id_of(tab_widget.get_channel_widget(name))
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


# --------------------------------------------------------------------------------------------- a refused rename

class RenameRequest:
    """A page whose name was typed over in its name field: the rename of its channel is asked to the collection"""

    def __init__(self, collection, old_name, new_name, *, write_fails_with=None, config_drops_it=False, others=()):
        self.collection = collection
        self.old_name, self.new_name = old_name, new_name
        self.dialogs = []
        collection.add(old_name)
        for other in others:
            collection.add(other)
        self.shown = []
        self.requester = _param_for_channel(collection.tab.channelsParamsTabWidget, old_name)
        collection.channel_params[old_name] = self.requester

        def handle_name_changed(old, new):
            if write_fails_with is not None:
                raise write_fails_with
            if config_drops_it:
                return  # The adjusters dropped the rename: the config has no channel of the new name
            collection.patches.append({'$rename': {'channels': {old: new}}})

        self.requester.handle_name_changed = handle_name_changed
        self.requester.show_name = lambda: self.shown.append(self.requester.name)
        if config_drops_it:
            type(collection).config_channels = property(lambda _: [old_name])

    def ask(self, monkeypatch):
        monkeypatch.setattr(params, 'handle_exception',
                            lambda reason, parent=None, context='': self.dialogs.append((reason, parent, context)))
        self.collection.handle_channel_rename_request(self.old_name, self.new_name, requester=self.requester)


@pytest.fixture()
def rename_collection():
    class Collection(FakeCollection):  # Own class: a test may change its config_channels
        pass
    return Collection()


def test_a_rename_the_config_refuses_puts_the_old_name_back_and_tells_the_user_why(rename_collection, monkeypatch):
    taken = ChannelNameTakenError("Cannot rename channel 'a' to 'b': a channel named 'b' exists")
    request = RenameRequest(rename_collection, FIRST, SECOND, write_fails_with=taken)

    request.ask(monkeypatch)

    assert rename_collection.channels == [FIRST]
    assert request.requester.name == FIRST
    assert rename_collection.tab.channelsParamsTabWidget.get_channels_names() == [FIRST]
    assert [(reason, parent) for reason, parent, _ in request.dialogs] == [(taken, rename_collection.tab)]


def test_a_rename_the_config_drops_puts_the_old_name_back_and_tells_the_user(rename_collection, monkeypatch):
    request = RenameRequest(rename_collection, FIRST, SECOND, config_drops_it=True)

    request.ask(monkeypatch)

    assert rename_collection.channels == [FIRST]
    assert request.requester.name == FIRST
    assert rename_collection.tab.channelsParamsTabWidget.get_channels_names() == [FIRST]
    (reason, _, context), = request.dialogs
    assert isinstance(reason, ClearMapValueError)
    assert FIRST in str(reason) and SECOND in str(reason)
    assert FIRST in context and SECOND in context


@pytest.mark.parametrize('new_name', [SECOND, SECOND.upper()], ids=['same case', 'other case'])
def test_a_rename_onto_the_name_of_another_channel_is_refused_before_anything_changes(rename_collection, monkeypatch,
                                                                                      new_name):
    # The params of the other channel were overwritten by the ones of the renamed channel, then lost by the revert
    request = RenameRequest(rename_collection, FIRST, new_name, others=[SECOND])
    params_before = dict(rename_collection.channel_params)

    request.ask(monkeypatch)

    assert rename_collection.channel_params == params_before  # The same channels, with the same params objects
    assert list(rename_collection.channel_params) == [FIRST, SECOND]
    assert rename_collection.patches == []  # The config was not even asked
    assert request.shown == [FIRST]  # The name field is put back, it holds the text typed over it
    (reason, parent, _), = request.dialogs
    assert isinstance(reason, ChannelNameTakenError) and parent is rename_collection.tab


def test_the_other_channel_can_still_be_renamed_after_a_refused_rename_onto_its_name(rename_collection, monkeypatch):
    # The user's sequence: refused (onto the name of the other channel), then the other channel gets a free name
    request = RenameRequest(rename_collection, FIRST, SECOND, others=[SECOND])
    request.ask(monkeypatch)
    other = _param_for_channel(rename_collection.tab.channelsParamsTabWidget, SECOND)
    other.handle_name_changed = lambda old, new: rename_collection.patches.append({'$rename': {'channels': {old: new}}})
    rename_collection.channel_params[SECOND] = other

    rename_collection.handle_channel_rename_request(SECOND, THIRD, requester=other)

    assert rename_collection.channels == [FIRST, THIRD]
    assert other.name == THIRD
    assert rename_collection.patches == [{'$rename': {'channels': {SECOND: THIRD}}}]
    assert len(request.dialogs) == 1  # The refusal only


def test_a_channel_can_change_the_case_of_its_own_name(rename_collection, monkeypatch):
    request = RenameRequest(rename_collection, FIRST, FIRST.upper())

    request.ask(monkeypatch)

    assert rename_collection.channels == [FIRST.upper()]
    assert request.dialogs == []


def test_an_accepted_rename_tells_the_user_nothing(rename_collection, monkeypatch):
    request = RenameRequest(rename_collection, FIRST, SECOND)

    request.ask(monkeypatch)

    assert rename_collection.channels == [SECOND]
    assert request.requester.name == SECOND
    assert request.dialogs == []


# ------------------------------------------------------------------------------- the files of a renamed channel

class FakeRenamingDialog:
    """The dialog where the user vets the renames: it answers as the test says, with the renames as proposed"""
    accepted = True
    shown: list = []

    def __init__(self, renames, *, not_renamed=(), title='', message='', accept_text='', parent=None):
        self.selected_renames = list(renames)
        self.not_renamed = list(not_renamed)
        type(self).shown.append(self)

    def exec(self):
        return params.QDialog.Accepted if self.accepted else params.QDialog.Rejected


@pytest.fixture()
def renaming_dialog(monkeypatch):
    class Dialog(FakeRenamingDialog):  # Own class: a test may make the user cancel
        shown = []
    monkeypatch.setattr(params, 'VerifyRenamingDialog', Dialog)
    return Dialog


@pytest.fixture()
def cells_of_first(tmp_path):
    """A file of the channel FIRST, and the rename the plan proposes for it"""
    cells = tmp_path / f'{FIRST}_cells.npy'
    cells.touch()
    return Rename(cells, tmp_path / f'{SECOND}_cells.npy')


def _plan_renaming(collection, rename, planned_with):
    def plan(channel_renames):
        planned_with.append((dict(channel_renames), list(collection.patches)))
        return ChannelRenamePlan(renames=[rename])
    collection._plan_file_renames = plan


def test_the_files_of_a_renamed_channel_are_renamed_once_the_user_accepts(rename_collection, monkeypatch,
                                                                          renaming_dialog, cells_of_first):
    planned_with = []
    _plan_renaming(rename_collection, cells_of_first, planned_with)
    request = RenameRequest(rename_collection, FIRST, SECOND)

    request.ask(monkeypatch)

    assert planned_with == [({FIRST: SECOND}, [])]  # Planned before the rename: the files have the old names
    assert rename_collection.channels == [SECOND]
    assert not cells_of_first.source.exists() and cells_of_first.target.exists()
    assert request.dialogs == []


def test_cancelling_the_renaming_of_the_files_cancels_the_rename_of_the_channel(rename_collection, monkeypatch,
                                                                                renaming_dialog, cells_of_first):
    renaming_dialog.accepted = False
    _plan_renaming(rename_collection, cells_of_first, [])
    request = RenameRequest(rename_collection, FIRST, SECOND)

    request.ask(monkeypatch)

    assert rename_collection.channels == [FIRST]
    assert rename_collection.patches == []  # The config was not asked
    assert request.shown == [FIRST]  # The name field is put back
    assert cells_of_first.source.exists() and not cells_of_first.target.exists()
    assert request.dialogs == []


def test_a_channel_without_files_is_renamed_without_asking(rename_collection, monkeypatch, renaming_dialog):
    request = RenameRequest(rename_collection, FIRST, SECOND)

    request.ask(monkeypatch)

    assert rename_collection.channels == [SECOND]
    assert renaming_dialog.shown == []


def test_the_files_left_alone_are_shown_even_if_nothing_is_renamed(rename_collection, monkeypatch, renaming_dialog,
                                                                   tmp_path):
    raw = NotRenamed(tmp_path / 'raw', 'raw data is never renamed')
    rename_collection._plan_file_renames = lambda channel_renames: ChannelRenamePlan(not_renamed=[raw])
    request = RenameRequest(rename_collection, FIRST, SECOND)

    request.ask(monkeypatch)

    (dialog,) = renaming_dialog.shown
    assert dialog.not_renamed == [raw]
    assert rename_collection.channels == [SECOND]


def test_a_rename_whose_files_cannot_be_planned_is_cancelled_and_the_user_told_why(rename_collection, monkeypatch,
                                                                                  renaming_dialog):
    inconsistent = ClearMapConfigError("moving_channel 'auto' is neither the channel nor its align_with")

    def plan_fails(channel_renames):
        raise inconsistent
    rename_collection._plan_file_renames = plan_fails
    request = RenameRequest(rename_collection, FIRST, SECOND)

    request.ask(monkeypatch)

    assert rename_collection.channels == [FIRST]
    assert rename_collection.patches == []
    assert request.shown == [FIRST]
    (reason, parent, context), = request.dialogs
    assert reason is inconsistent and parent is rename_collection.tab
    assert 'keeps its name' in context


def test_the_files_are_not_renamed_if_the_config_refuses_the_rename(rename_collection, monkeypatch, renaming_dialog,
                                                                    cells_of_first):
    _plan_renaming(rename_collection, cells_of_first, [])
    request = RenameRequest(rename_collection, FIRST, SECOND, config_drops_it=True)

    request.ask(monkeypatch)

    assert rename_collection.channels == [FIRST]
    assert cells_of_first.source.exists() and not cells_of_first.target.exists()


def test_files_which_cannot_be_renamed_keep_their_names_and_the_user_is_told(rename_collection, monkeypatch,
                                                                            renaming_dialog, cells_of_first):
    _plan_renaming(rename_collection, cells_of_first, [])
    request = RenameRequest(rename_collection, FIRST, SECOND)
    cells_of_first.target.touch()  # Appeared after the user vetted the renames

    request.ask(monkeypatch)

    assert rename_collection.channels == [SECOND]
    assert cells_of_first.source.exists()
    (reason, parent, context), = request.dialogs
    assert 'onto what exists' in str(reason) and parent is rename_collection.tab
    assert 'keep their old names' in context


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
        """A page of a channel is a page of its Channel, a page of a compound channel of the Channels of its parts"""
        registry = self._sample_manager.channel_registry
        pages = []
        for name in self.channels_in_config:
            parts = compound_channel_parts(name)
            page_of = tuple(registry.resolve(part) for part in parts) if parts else registry.resolve(name)
            pages.append((page_of, name))
        return pages

    def add_channel_tab(self, name):
        page = self.ui.channelsParamsTabWidget.add_page(name, with_id=False)
        self._assign_channel_page_id(page, name)  # As _get_or_init_channel_ui does: the id of its channel
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


def test_a_refused_removal_leaves_the_page_of_the_channel(registry, tab, monkeypatch):
    """Was OPEN RISK 1: the page and the params were dropped before the config was asked, and stayed dropped"""
    monkeypatch.setattr(params_interfaces, 'handle_exception', lambda *args, **kwargs: 'dismiss')
    registry.add(FIRST)
    tab.channels_in_config = [FIRST]
    tab.params = FakeCollection(patch_fails=True)
    tab.params.tab = tab.ui
    tab.reconcile_channel_pages()

    SampleInfoTab.remove_channel(tab, FIRST)

    assert tab.ui.channelsParamsTabWidget.get_channels_names() == [FIRST]
    assert FIRST in tab.params.channels


def test_an_accepted_removal_does_not_make_the_page_again(registry, tab):
    channel = registry.add(FIRST)
    tab.channels_in_config = [FIRST]
    tab.params = FakeCollection()
    tab.params.tab = tab.ui
    tab.reconcile_channel_pages()
    registry.remove(channel)
    tab.channels_in_config = []

    SampleInfoTab.remove_channel(tab, FIRST)

    assert tab.ui.channelsParamsTabWidget.get_channels_names() == []
    assert tab.params.channels == []


def test_a_reconcile_without_change_does_nothing_to_the_pages(registry, tab):
    registry.add(FIRST)
    tab.channels_in_config = [FIRST]
    tab.reconcile_channel_pages()
    page_before = tab.ui.channelsParamsTabWidget.get_channel_widget(FIRST)

    tab.reconcile_channel_pages()

    assert tab.ui.channelsParamsTabWidget.get_channel_widget(FIRST) is page_before


def test_a_channel_removed_and_re_added_under_the_same_name_gets_a_new_page(registry, tab):
    """The old page holds the id of the removed channel, which the registry no longer knows"""
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


# ------------------------------------------------------------------------------------- the id of a page, from the registry

def test_a_page_made_before_its_channel_was_committed_gets_its_id_at_the_reconcile(registry, tab):
    # The (+) of the sample tab: the page is made (Channel_0), then the commit adds the channel to the registry, then the
    # refresh reconciles. The page exists, so nothing is added: the reconcile must still give it its id.
    tab_widget = tab.ui.channelsParamsTabWidget
    page = tab_widget.add_page('Channel_0', with_id=False)
    channel = registry.add('Channel_0')
    tab.channels_in_config = ['Channel_0']

    tab.reconcile_channel_pages()

    assert channel_page_id_of(page) == ChannelPageId((channel.id,))


def test_a_tab_which_is_not_a_pipeline_tab_gives_its_new_pages_an_id(registry, tab, monkeypatch):
    # The sample tab is an ExperimentTab, not a PipelineTab: the id must not depend on the kind of tab
    monkeypatch.setattr(tabs_interfaces, 'create_clearmap_widget', lambda *_, **__: FakePage())
    tab.channels_ui_name = 'some_channel_page'
    channel = registry.add(FIRST)

    name, page, creating = tab._get_or_init_channel_ui(FIRST)

    assert creating and name == FIRST
    assert channel_page_id_of(page) == ChannelPageId((channel.id,))


def test_a_page_of_a_channel_the_registry_does_not_know_has_no_id(registry, tab, monkeypatch):
    monkeypatch.setattr(tabs_interfaces, 'create_clearmap_widget', lambda *_, **__: FakePage())
    tab.channels_ui_name = 'some_channel_page'

    _, page, _ = tab._get_or_init_channel_ui(f'{FIRST}-{SECOND}')  # A compound channel whose parts are not known

    assert channel_page_id_of(page) is None


def test_the_page_of_a_compound_channel_holds_the_ids_of_its_parts(registry, tab, monkeypatch):
    monkeypatch.setattr(tabs_interfaces, 'create_clearmap_widget', lambda *_, **__: FakePage())
    tab.channels_ui_name = 'some_channel_page'
    first, second = registry.add(FIRST), registry.add(SECOND)

    _, page, _ = tab._get_or_init_channel_ui(f'{SECOND}-{FIRST}')  # Oriented: the order of the parts is kept

    assert channel_page_id_of(page) == ChannelPageId((second.id, first.id))


def test_the_page_of_a_pair_is_made_again_when_one_of_its_parts_is_replaced(registry, tab):
    # The second channel is removed then added again under the same name: a new channel, with a new id
    first, second = registry.add(FIRST), registry.add(SECOND)
    pair = f'{FIRST}-{SECOND}'
    tab.channels_in_config = [FIRST, SECOND, pair]
    tab.reconcile_channel_pages()
    tab_widget = tab.ui.channelsParamsTabWidget
    assert channel_page_id_of(tab_widget.get_channel_widget(pair)) == ChannelPageId((first.id, second.id))
    registry.remove(second)
    new_second = registry.add(SECOND)

    tab.reconcile_channel_pages()

    assert channel_page_id_of(tab_widget.get_channel_widget(pair)) == ChannelPageId((first.id, new_second.id))


def test_the_pages_of_a_replaced_registry_are_made_again_with_the_ids_of_the_new_one(tab):
    # A directory load replaces the registry (SampleManager.setup): the pages hold ids it does not know
    tab._sample_manager.channel_registry.add(FIRST)
    tab._sample_manager.channel_registry.add(SECOND)
    tab.channels_in_config = [FIRST, SECOND]
    tab.reconcile_channel_pages()
    tab_widget = tab.ui.channelsParamsTabWidget
    old_pages = [tab_widget.get_channel_widget(name) for name in (FIRST, SECOND)]
    old_params = [tab.params.channel_params[name] for name in (FIRST, SECOND)]
    new_registry = ChannelRegistry()
    new_ids = [new_registry.add(name).id for name in (FIRST, SECOND)]
    tab._sample_manager.channel_registry = new_registry

    tab.reconcile_channel_pages()

    assert tab_widget.get_channels_names() == [FIRST, SECOND]
    new_pages = [tab_widget.get_channel_widget(name) for name in (FIRST, SECOND)]
    assert [channel_page_id_of(page) for page in new_pages] == [ChannelPageId((new_id,)) for new_id in new_ids]
    assert all(new is not old for new, old in zip(new_pages, old_pages))
    assert all(tab.params.channel_params[name] is not old for name, old in zip((FIRST, SECOND), old_params))
    assert all(params.torn_down for params in old_params)


def test_pages_of_the_current_registry_are_kept(registry, tab):
    registry.add(FIRST)
    tab.channels_in_config = [FIRST]
    tab.reconcile_channel_pages()
    page = tab.ui.channelsParamsTabWidget.get_channel_widget(FIRST)

    tab.reconcile_channel_pages()

    assert tab.ui.channelsParamsTabWidget.get_channel_widget(FIRST) is page


# ----------------------------------------------------------------------- a reconcile asked from inside a reconcile

def test_a_reconcile_asked_while_one_runs_runs_once_more_after_it_not_inside_it(registry, tab):
    # A page setup which commits (and so announces the channels again) must not start a second reconcile in the first
    registry.add(FIRST)
    registry.add(SECOND)
    tab.channels_in_config = [FIRST, SECOND]
    depth, deepest, passes = [0], [0], [0]
    once = tab._reconcile_channel_pages_once
    add_page = tab.add_channel_tab

    def counted_once():
        depth[0] += 1
        deepest[0] = max(deepest[0], depth[0])
        passes[0] += 1
        try:
            once()
        finally:
            depth[0] -= 1

    def add_page_which_announces(name):
        add_page(name)
        tab.reconcile_channel_pages()  # As the event of the commit of the page would

    tab._reconcile_channel_pages_once = counted_once
    tab.add_channel_tab = add_page_which_announces

    tab.reconcile_channel_pages()

    assert tab.ui.channelsParamsTabWidget.get_channels_names() == [FIRST, SECOND]
    assert deepest[0] == 1  # Never one inside another
    assert passes[0] == 2  # The announcements of the first pass were answered by one more pass
    assert not tab._reconciling


def test_a_reconcile_asked_while_one_runs_sees_what_changed_meanwhile(registry, tab):
    registry.add(FIRST)
    registry.add(SECOND)
    tab.channels_in_config = [FIRST]
    add_page = tab.add_channel_tab

    def add_page_while_the_config_changes(name):
        add_page(name)
        tab.channels_in_config = [FIRST, SECOND]  # A commit lands during the page setup...
        tab.reconcile_channel_pages()  # ...and announces itself

    tab.add_channel_tab = add_page_while_the_config_changes

    tab.reconcile_channel_pages()

    assert tab.ui.channelsParamsTabWidget.get_channels_names() == [FIRST, SECOND]


def test_a_reconcile_which_fails_does_not_block_the_next_one(registry, tab):
    registry.add(FIRST)
    tab.channels_in_config = [FIRST]

    def fail(name):
        raise RuntimeError('page setup failed')

    original, tab.add_channel_tab = tab.add_channel_tab, fail
    with pytest.raises(RuntimeError):
        tab.reconcile_channel_pages()
    tab.add_channel_tab = original

    tab.reconcile_channel_pages()

    assert tab.ui.channelsParamsTabWidget.get_channels_names() == [FIRST]


# ------------------------------------------------------------------------------------ showing the name of a channel

class FakeLineEdit:
    """A name field which holds text typed over the name of the channel"""

    def __init__(self, text):
        self.text, self.blocked_while_set = text, None
        self._blocked = False

    def blockSignals(self, blocked):
        self._blocked = blocked

    def setText(self, text):
        self.text, self.blocked_while_set = text, self._blocked


def test_showing_the_name_puts_it_back_in_the_name_field_without_committing_it(monkeypatch):
    monkeypatch.setattr(params_interfaces, 'QLineEdit', FakeLineEdit)
    tab_widget = FakeNameTab()
    tab_widget.add_page(FIRST)
    param = _param_for_channel(tab_widget, FIRST)
    param.nameWidget = FakeLineEdit(SECOND)  # Typed over the name of the channel, not committed (refused)

    param.show_name()

    assert (param.nameWidget.text, param.nameWidget.blocked_while_set) == (FIRST, True)
    assert tab_widget.get_channels_names() == [FIRST]
