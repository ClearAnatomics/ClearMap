"""
Creating the page of a channel never writes the config: the config adjusters create (and fill from the defaults)
the entry of every channel in each section, in the commit that adds the channel to the sample. A page is only
a view of that entry.
"""
import contextlib

import pytest

from ClearMap.Utils.event_bus import EventBus
from ClearMap.config.channel import ChannelId
from ClearMap.gui.widgets import channel_id_of, set_channel_id
from ClearMap.gui import params, params_interfaces
from ClearMap.gui.params_interfaces import ChannelUiParameter
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

    def get_channel_widget(self, channel_name, return_idx=False):
        if channel_name not in self.pages:
            page = FakeChannelPage(channel_name)
            set_channel_id(page, ChannelId.new())  # The page of a channel of the registry holds its id
            self.pages[channel_name] = page
        page = self.pages[channel_name]
        return (page, list(self.pages).index(channel_name)) if return_idx else page

    def get_channel_widget_by_id(self, channel_id, return_idx=False):
        for index, (_, page) in enumerate(self.pages.items()):
            if channel_id_of(page) == channel_id:
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
