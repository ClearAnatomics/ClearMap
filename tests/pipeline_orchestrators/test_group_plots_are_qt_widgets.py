"""
The group plots (volcanoes, histograms) are returned as Qt widgets (matplotlib canvas and toolbar) by ``run_plots``,
in any Qt application.

``run_plots`` only made its views when the application was named 'ClearMap' (a guard copied from the Sphinx
protection of the GUI tabs), so with any other application, e.g. one made by ``ensure_qapp()`` in a script, and
with none, it raised an UnboundLocalError. Without an application it now says so before any widget is made (Qt aborts
the process on a widget without a QApplication). The widgets are made by ClearMap.Visualization.Qt.utils, so that the
orchestrator imports no Qt, and without QtWebEngine, which PyQt6 has not on conda-forge. Qt is faked here: the sandbox and the CI have no display, and the rule tested is the one of
ClearMap, not of Qt.
"""
import sys
from types import ModuleType, SimpleNamespace

import pandas as pd
import pytest

from ClearMap.Utils.exceptions import ClearMapRuntimeError
from ClearMap.Visualization.Qt import utils as qt_utils
from ClearMap.pipeline_orchestrators.group_orchestrators import DensityGroupAnalysisOrchestrator

GROUP_NAMES = ('control', 'treated')


class FakeApplication:
    """A QApplication with the name a script gets by default (``QApplication([])``)"""
    current = None

    @classmethod
    def instance(cls):
        return cls.current

    def applicationName(self):
        return 'python'


class FakeWidget:
    pass


class FakeLayout:
    """Like a QVBoxLayout, it belongs to its widget and keeps the widgets added, in order"""
    def __init__(self, parent):
        parent.layout = self
        self.children = []

    def addWidget(self, widget):
        self.children.append(widget)


class FakeCanvas:
    def __init__(self, figure):
        self.figure = figure


class FakeToolbar:
    def __init__(self, canvas, parent):
        self.canvas = canvas


def _fake_module(name, **attributes):
    module = ModuleType(name)
    module.__dict__.update(attributes)
    return module


@pytest.fixture()
def fake_qt(monkeypatch):
    """The Qt application, widgets and matplotlib Qt backend of the group plots, without a display"""
    monkeypatch.setattr(qt_utils, 'QApplication', FakeApplication)
    monkeypatch.setattr(qt_utils, 'QWidget', FakeWidget)
    monkeypatch.setattr(qt_utils, 'QVBoxLayout', FakeLayout)
    monkeypatch.setitem(sys.modules, 'matplotlib.backends.backend_qtagg',
                        _fake_module('matplotlib.backends.backend_qtagg', FigureCanvasQTAgg=FakeCanvas,
                                     NavigationToolbar2QT=FakeToolbar))
    monkeypatch.setattr(FakeApplication, 'current', None)
    return FakeApplication


@pytest.fixture()
def analysis_with_stats_table(tmp_path):
    """A group analysis whose statistics table of the comparison control/treated is written"""
    controller = SimpleNamespace(groups={name: [str(tmp_path / name)] for name in GROUP_NAMES},
                                 group_base_dir=tmp_path)
    analysis = DensityGroupAnalysisOrchestrator(group_controller=controller, pipeline='CellMap')
    table_path = analysis.assets.stats_table_path('cells', *GROUP_NAMES)
    table_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({'id': [1, 2], 'p_value': [0.01, 0.5]}).to_csv(table_path)
    return analysis


def plot_the_number_of_rows(df, **_):
    return f'figure of {len(df)} rows'


def test_the_group_plots_are_canvases_with_a_toolbar_in_an_application_not_named_clearmap(
        fake_qt, analysis_with_stats_table):
    fake_qt.current = FakeApplication()
    widgets = analysis_with_stats_table.run_plots(plot_the_number_of_rows, [GROUP_NAMES], channel='cells',
                                                  plot_kw_args={'group_names': None})
    assert len(widgets) == 1
    toolbar, canvas = widgets[0].layout.children
    assert canvas.figure == 'figure of 2 rows' and toolbar.canvas is canvas


def test_the_group_plots_ask_for_an_application_when_there_is_none(fake_qt, analysis_with_stats_table):
    with pytest.raises(ClearMapRuntimeError, match='QApplication'):
        analysis_with_stats_table.run_plots(plot_the_number_of_rows, [GROUP_NAMES], channel='cells',
                                            plot_kw_args={'group_names': None})
