"""
Creating the page of a channel never writes the config: the config adjusters create (and fill from the defaults)
the entry of every channel in each section, in the commit that adds the channel to the sample. A page is only
a view of that entry.
"""
import contextlib

import pytest

from ClearMap.Utils.event_bus import EventBus
from ClearMap.config.channel import ChannelId
from ClearMap.config.compound_keys import compound_channel_parts
from ClearMap.gui.widgets import ChannelPageId, channel_page_id_of, set_channel_page_id
from ClearMap.gui import params, params_interfaces
from ClearMap.gui.params_interfaces import ChannelUiParameter
from ClearMap.gui.tabs_interfaces import GenericTab
from ClearMap.gui.params import (CellMapParams, ColocalizationParams, RegistrationParams, StitchingParams,
                                 TractMapParams, VesselParams)

CHANNEL_NAME = 'nuclei'
PAIR_NAME = 'nuclei-other_nuclei'
MINIMAL_PARAMS_SECTION_NAME = 'minimal'


class FakeWidget:
    """Any widget of a page: records nothing, answers neutral values"""

    def __init__(self, name=''):
        self._name = name

    def __getattr__(self, name):
        return FakeWidget(name)

    def __call__(self, *args, **kwargs):
        return None

    def objectName(self):
        return self._name

    def parentWidget(self):
        return None

    def findChildren(self, *_):
        return []


class FakeChannelPage(FakeWidget):
    """A page holds the id of its channel, as a page made by the sample tab does (Qt property API)"""

    def __init__(self, name=''):
        super().__init__(name)
        self._properties = {}

    def setProperty(self, name, value):
        self._properties[name] = value

    def property(self, name):
        return self._properties.get(name)


class FakeChannelsTabWidget(FakeWidget):
    """The tab widget holding one page per channel"""

    def __init__(self):
        super().__init__('channelsParamsTabWidget')
        self.pages = {}
        self.ids = {}  # The registry: one id per channel name

    def get_channel_widget(self, channel_name, return_idx=False):
        if channel_name not in self.pages:
            page = FakeChannelPage(channel_name)
            parts = compound_channel_parts(channel_name) or (channel_name,)  # The parts of a pair have the ids
            ids = tuple(self.ids.setdefault(part, ChannelId.new()) for part in parts)
            set_channel_page_id(page, ChannelPageId(ids))
            self.pages[channel_name] = page
        page = self.pages[channel_name]
        return (page, list(self.pages).index(channel_name)) if return_idx else page

    def get_channel_widget_by_page_id(self, page_id, return_idx=False):
        for index, (_, page) in enumerate(self.pages.items()):
            if channel_page_id_of(page) == page_id:
                return (page, index) if return_idx else page
        return (None, -1) if return_idx else None

    def tabText(self, index):
        return list(self.pages)[index]


class FakeTab(FakeWidget):
    def __init__(self):
        super().__init__('tab')
        self.channelsParamsTabWidget = FakeChannelsTabWidget()


class FakeWidgetOps:
    def get(self, widget):
        return None

    def set(self, widget, value, silent=False):
        pass

    def connect(self, widget, callback):
        return None


class MinimalChannelParams(ChannelUiParameter):
    """Stands for a channel params class whose real widgets cannot be faked here (pipeline widget, layouts...)"""

    def __init__(self, tab, channel_name, *, event_bus, get_view=None, apply_patch=None, **_):
        super().__init__(tab, channel_name, event_bus=event_bus, get_view=get_view, apply_patch=apply_patch)

    @property
    def cfg_subtree(self):
        return [MINIMAL_PARAMS_SECTION_NAME, 'channels', self.name]

    def build_params_dict(self):
        return {}


def _config():
    """The config as the adjusters leave it once the channel is in the sample: each section has its entry"""
    channel = {'data_type': 'nuclei', 'path': '', 'extension': '.tif'}
    return {'sample': {'channels': {CHANNEL_NAME: channel, 'other_nuclei': dict(channel)}},
            'registration': {'channels': {CHANNEL_NAME: {}}},
            'cell_map': {'channels': {CHANNEL_NAME: {}}, 'performance': {'channels': {CHANNEL_NAME: {}}}},
            'stitching': {'channels': {CHANNEL_NAME: {}}},
            'tract_map': {'channels': {CHANNEL_NAME: {}}},
            'colocalization': {'channels': {PAIR_NAME: {}}},
            'vasculature': {'binarization': {'single_channels': {CHANNEL_NAME: {}}}}}


def _is_fake_widget_alive(widget):
    return widget is not None


@pytest.fixture
def patches(monkeypatch):
    monkeypatch.setattr(params_interfaces, 'WIDGET_OPS', FakeWidgetOps())
    monkeypatch.setattr(params_interfaces, '_is_alive', _is_fake_widget_alive)
    monkeypatch.setattr(params_interfaces, 'QFrame', FakeWidget.__mro__[1])  # No Qt needed: the fakes are no frames
    monkeypatch.setattr(params_interfaces, 'ExtendableTabWidget', FakeChannelsTabWidget)  # Channel names = tab texts
    monkeypatch.setattr(params, 'LandmarksWeightsPanel', FakeWidget)  # A real QWidget, needs a real parent
    return []


def _collection(collection_class, patches, config=None, **extra_kwargs):
    config = _config() if config is None else config

    def get_view():
        return config

    return collection_class(FakeTab(), event_bus=EventBus(), get_view=get_view,
                            apply_patch=patches.append, **extra_kwargs)


@pytest.mark.parametrize('collection_class, extra_kwargs', [
    (RegistrationParams, {}),
    (CellMapParams, {'sample_params': None}),
    (TractMapParams, {'sample_params': None}),
], ids=['registration', 'cell_map', 'tract_map'])
def test_adding_a_channel_page_writes_no_config(patches, collection_class, extra_kwargs):
    collection = _collection(collection_class, patches, **extra_kwargs)

    collection.add_channel(CHANNEL_NAME)

    assert CHANNEL_NAME in collection.channels
    assert patches == []


def test_adding_a_colocalization_pair_page_writes_no_config(patches):
    collection = _collection(ColocalizationParams, patches, sample_params=None)

    collection.add_channel(PAIR_NAME)

    assert PAIR_NAME in collection.channels
    assert patches == []


def test_adding_a_vasculature_channel_page_writes_no_config(patches, monkeypatch):
    monkeypatch.setattr(params, 'VesselBinarizationParams', MinimalChannelParams)
    monkeypatch.setattr(params, 'QSignalBlocker', contextlib.nullcontext)  # The fake combo boxes emit nothing
    collection = _collection(VesselParams, patches, sample_params=None)

    collection.add_channel(CHANNEL_NAME)

    assert CHANNEL_NAME in collection.channels
    assert patches == []


def test_a_channel_the_adjusters_do_not_stitch_gets_no_stitching_entry_from_its_page(patches, monkeypatch):
    monkeypatch.setattr(params, 'ChannelStitchingParams', MinimalChannelParams)
    config = _config()
    del config['stitching']['channels'][CHANNEL_NAME]  # Not stitchable (e.g. no tile pattern)
    collection = _collection(StitchingParams, patches, config=config)

    collection.add_channel(CHANNEL_NAME)

    assert patches == []


# ------------------------------------------------------------------------- setting a page up is not an edit

class SetUpParams:
    """The params of a page: handlers are suppressed (a widget painted from the config is not an edit) in the context"""

    def __init__(self):
        self.suppressed = False
        self.torn_down = False

    @contextlib.contextmanager
    def _suppress_handlers(self):
        self.suppressed = True
        try:
            yield
        finally:
            self.suppressed = False

    def teardown(self):
        self.torn_down = True


class SetUpTab:
    """The real add_channel_tab around a fake page: what it calls is recorded"""
    channels_ui_name = 'channel_page'
    name = 'fake tab'
    add_channel_tab = GenericTab.add_channel_tab
    _set_up_channel_page = GenericTab._set_up_channel_page

    def __init__(self, creating, known=False):
        self.params = {CHANNEL_NAME: SetUpParams()} if known else {}
        self.creating = creating
        self.calls = []

    def _get_or_init_channel_ui(self, channel):
        return channel, FakeWidget('page'), self.creating

    def _create_channel_params(self, channel):
        self.params[channel] = SetUpParams()

    def _setup_channel(self, page_widget, channel):
        self.calls.append(('setup', self.params[channel].suppressed))  # Fills combo boxes, which emit signals

    def _bind_channel(self, page_widget, channel):
        self.calls.append(('bind', self.params[channel].suppressed))

    def _connect_children_whats_this(self, page_widget):
        self.calls.append(('whats_this', None))

    def _on_channel_added(self, channel):
        self.calls.append(('added', None))


def test_a_new_page_is_set_up_with_the_handlers_of_its_params_suppressed():
    tab = SetUpTab(creating=True)

    tab.add_channel_tab(CHANNEL_NAME)

    assert tab.calls == [('setup', True), ('bind', False), ('whats_this', None), ('added', None)]


def test_a_page_made_again_for_known_params_is_set_up_with_the_handlers_suppressed():
    tab = SetUpTab(creating=True, known=True)

    tab.add_channel_tab(CHANNEL_NAME)

    assert tab.params[CHANNEL_NAME].torn_down
    assert tab.calls == [('setup', True), ('bind', False), ('whats_this', None), ('added', None)]


def test_a_page_which_already_exists_for_known_params_is_not_set_up_again():
    tab = SetUpTab(creating=False, known=True)

    tab.add_channel_tab(CHANNEL_NAME)

    assert tab.calls == []
