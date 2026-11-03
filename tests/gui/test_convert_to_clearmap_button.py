"""
The "Convert to ClearMap" button of a channel page asks the stitching worker to prepare the raw data of the channel
(prepare_channel_raw_data), a method it did not have: every click raised AttributeError.
"""
from unittest import mock

from ClearMap.Utils.events import UiPrepareRawDataForClearMap
from ClearMap.gui.tabs import SampleInfoTab
from ClearMap.pipeline_orchestrators.stitching_orchestrator import StitchingProcessor


class FakeSampleInfoTab:
    prepare_channel_raw_data = SampleInfoTab.prepare_channel_raw_data

    def __init__(self, stitching_worker):
        self.exp_controller = mock.Mock()
        self.exp_controller.get_worker.return_value = stitching_worker


def test_the_button_prepares_the_raw_data_of_its_channel():
    stitching_worker = mock.create_autospec(StitchingProcessor, instance=True)
    tab = FakeSampleInfoTab(stitching_worker)

    tab.prepare_channel_raw_data(UiPrepareRawDataForClearMap(channel_name='cfos'))

    tab.exp_controller.get_worker.assert_called_once_with('stitching')
    stitching_worker.prepare_channel_raw_data.assert_called_once_with('cfos')
