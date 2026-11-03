"""
Filtering the cells of a channel (preview or run) voxelizes them with the weights column chosen on the page of the
channel. It read it from the parameters of the tab, which have no such attribute: AttributeError at every filtering.
"""
import contextlib

from ClearMap.gui.tabs import CellCounterTab

CHANNEL = 'cfos'
WEIGHTS_COLUMN = 'size'


class FakeChannelParams:
    voxelization_weights = WEIGHTS_COLUMN


class FakeCellMapParams:
    """The parameters of the tab: per channel only"""

    def __getitem__(self, channel):
        return {CHANNEL: FakeChannelParams()}[channel]


class FakeAsset:
    exists = True


class FakeSampleManager:
    def get(self, *args, **kwargs):
        return FakeAsset()


class FakeDetector:
    def filter_cells(self):
        pass

    def voxelize(self, *args, **kwargs):
        pass

    def stop_process(self):
        pass


class FakeCellCounterTab:
    preview_cell_filter = CellCounterTab.preview_cell_filter
    _CellCounterTab__filter_cells = CellCounterTab._CellCounterTab__filter_cells

    def __init__(self):
        self.params = FakeCellMapParams()
        self.sample_manager = FakeSampleManager()
        self.detector = FakeDetector()
        self.steps = {}

    @contextlib.contextmanager
    def debug_mode(self, channel, debug_status):
        yield

    def get_worker(self, channel):
        return self.detector

    def wrap_step(self, task_name, func, step_args=None, step_kw_args=None, **kwargs):
        self.steps[task_name] = step_kw_args

    def update_cell_number(self, channel):
        pass

    def plot_cell_filter_results(self, channel):
        pass


def test_the_filtered_cells_are_voxelized_with_the_weights_of_the_page_of_the_channel():
    tab = FakeCellCounterTab()

    tab.preview_cell_filter(CHANNEL)

    assert tab.steps['Voxelizing'] == {'weights_column': WEIGHTS_COLUMN}
