"""
At the start of a sample, the sample tab offers to prepare the raw data of the channels that need it, then the
stitching worker prepares them (prepare_all_channels_raw_data). Both ask the worker which channels need it
(needs_raw_data_preparation): a tiled channel not set to use npy is stitched from its tif tiles, so it is not offered.
A copy of the rule in the tab offered it, and saying yes then did nothing for it.
"""
from types import SimpleNamespace

from ClearMap.gui import tabs
from ClearMap.gui.tabs import SampleInfoTab

CHANNELS_NEEDING_PREPARATION = {'tiled_npy': True, 'tiled_tif': False, 'prestitched': True}  # As the worker says


class FakeStitchingWorker:
    def needs_raw_data_preparation(self, channel, force=False):
        return CHANNELS_NEEDING_PREPARATION[channel]

    def prepare_all_channels_raw_data(self, force=False):
        pass


class FakeSampleManager:
    pipeline_ready_channels = list(CHANNELS_NEEDING_PREPARATION)

    def is_tiled(self, channel):
        return channel.startswith('tiled')

    def has_npy(self, channel):
        return False

    def get(self, asset_type, channel):
        return SimpleNamespace(exists=False)


class FakeExperimentController:
    def get_worker(self, pipeline):
        assert pipeline == 'stitching'
        return FakeStitchingWorker()


class FakeSampleTab:
    prompt_prepare_all_channels_raw_data = SampleInfoTab.prompt_prepare_all_channels_raw_data

    def __init__(self):
        self._preparation_offered = False
        self.sample_manager = FakeSampleManager()
        self.exp_controller = FakeExperimentController()


class RecordingPromptDialog:
    def __init__(self):
        self.messages = []

    def __call__(self, title, message):
        self.messages.append(message)
        return False  # The user declines: nothing is prepared


def test_the_dialog_offers_the_channels_the_stitching_worker_prepares(monkeypatch):
    prompt = RecordingPromptDialog()
    monkeypatch.setattr(tabs, 'prompt_dialog', prompt)

    FakeSampleTab().prompt_prepare_all_channels_raw_data()

    [message] = prompt.messages
    assert 'tiled_npy (convert tiles)' in message
    assert 'prestitched (stack/import)' in message
    assert 'tiled_tif' not in message
