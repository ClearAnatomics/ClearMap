"""
The stitching params of the tab follow the stitching channels of the config by themselves (reconcile_children_from_view)
besides the reconcile of the pages: the params of each page are added and dropped as those of any page are, checked
on the way in and torn down on the way out.
"""
import pytest

from ClearMap.Utils.event_bus import EventBus
from ClearMap.Utils.exceptions import ClearMapValueError
from ClearMap.gui import params, params_interfaces
from ClearMap.gui.params import StitchingParams


class FakeWidget:
    def __getattr__(self, name):
        return lambda *args, **kwargs: None


class FakeChannelsTabWidget(FakeWidget):
    """Every channel has a page (the stitching params are made only for a channel which has one)"""

    def get_channel_widget(self, channel_name, return_idx=False):
        return (FakeWidget(), 0) if return_idx else FakeWidget()


class FakeTab(FakeWidget):
    def __init__(self):
        self.channelsParamsTabWidget = FakeChannelsTabWidget()


class FakePageParams:
    """Stands for ChannelStitchingParams, whose real widgets cannot be faked here"""

    def __init__(self, tab, channel, **_):
        self.name = channel
        self.torn_down = False

    def teardown(self):
        self.torn_down = True


@pytest.fixture()
def config():
    return {'stitching': {'channels': {'fos': {}, 'auto': {}}}}


@pytest.fixture()
def stitching_params(monkeypatch, config):
    monkeypatch.setattr(params, 'ChannelStitchingParams', FakePageParams)
    monkeypatch.setattr(params_interfaces, 'ChannelUiParameter', FakePageParams)  # It passes as a page's params
    return StitchingParams(FakeTab(), event_bus=EventBus(), get_view=lambda: config, apply_patch=lambda patch: None)


def test_the_params_of_a_channel_gone_from_the_config_are_torn_down(stitching_params, config):
    stitching_params.reconcile_children_from_view()
    auto = stitching_params['auto']

    del config['stitching']['channels']['auto']
    stitching_params.reconcile_children_from_view()

    assert stitching_params.channels == ['fos']
    assert auto.torn_down


def test_the_params_made_for_a_page_are_checked_as_those_of_any_page(monkeypatch, stitching_params):
    monkeypatch.setattr(params, 'ChannelStitchingParams', lambda *args, **kwargs: object())

    with pytest.raises(ClearMapValueError):
        stitching_params.reconcile_children_from_view()
