"""
Selecting the section of the cell filter shows the histograms of the detected cells. It read their table with np.load,
but cell detection writes it in the feather format since the cell tables moved to pandas: an error popup at every
selection of the section. It reads it as the worker does (get_coords), which reads feather and the npy of before.
"""
import pandas as pd

from ClearMap.gui.tabs import CellCounterTab

CHANNEL = 'cfos'
FEATHER_MAGIC = b'ARROW1'


class FakeSampleManager:
    def __init__(self, table_path):
        self.table_path = table_path

    def get_path(self, *args, **kwargs):
        return self.table_path


class FakeDetector:
    def __init__(self, cells_table):
        self.cells_table = cells_table

    def get_coords(self, coord_type='filtered', aligned=False):
        return self.cells_table, self.cells_table[['x', 'y', 'z']].to_numpy()


class FakeCellCounterTab:
    _CellCounterTab__plot_histograms = CellCounterTab._CellCounterTab__plot_histograms

    def __init__(self, table_path, cells_table):
        self.sample_manager = FakeSampleManager(table_path)
        self.detector = FakeDetector(cells_table)
        self.cell_size_histogram = None
        self.cell_intensity_histogram = None
        self.plotted_columns = []

    def get_worker(self, channel):
        return self.detector

    def _CellCounterTab__plot_histogram(self, channel, cells_df, key, histogram):
        self.plotted_columns.append(list(cells_df[key]))
        return histogram


def test_the_histograms_show_the_cells_of_the_feather_table(tmp_path):
    table_path = tmp_path / 'cells_raw.feather'
    table_path.write_bytes(FEATHER_MAGIC + b'\0' * 32)  # Not an npy file: np.load cannot read it
    cells_table = pd.DataFrame({'x': [1, 2], 'y': [3, 4], 'z': [5, 6], 'size': [10, 20], 'source': [100., 200.]})
    tab = FakeCellCounterTab(table_path, cells_table)

    tab._CellCounterTab__plot_histograms(CHANNEL)

    assert tab.plotted_columns == [[10, 20], [100., 200.]]
