"""
The data frame widget (the colocalization tables) runs with pandas 2 and later (ClearMap3.yml asks for pandas>=2.2.2,
the first pandas that runs with NumPy 2): a Series indexed by the column names is not indexed by position any more
(KeyError: 0 with pandas 3).
"""
import pandas as pd

import ClearMap.gui.widgets as widgets


def test_the_floats_of_a_data_frame_widget_have_n_digits(monkeypatch):
    texts = []
    monkeypatch.setattr(widgets, 'QTableWidgetItem', texts.append)

    widgets.DataFrameWidget(pd.DataFrame({'count': [3], 'density': [0.123456]}), n_digits=3)

    assert texts == ['3', '0.123']
