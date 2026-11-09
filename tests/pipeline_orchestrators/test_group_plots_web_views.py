"""
The group plots (volcanoes, histograms) are returned as Qt web views by ``run_plots``, in any Qt application.

``run_plots`` only imported ``QWebEngineView`` when the application was named 'ClearMap' (a guard copied from the
Sphinx protection of the GUI tabs), so with any other application, e.g. one made by ``ensure_qapp()`` in a script, and
with none, it raised an UnboundLocalError. Without an application it now says so before any widget is made (Qt aborts
the process on a widget without a QApplication). Qt is faked here: the sandbox and the CI have no display, and the rule
tested is the one of ClearMap, not of Qt.
"""
import sys
from types import ModuleType, SimpleNamespace

import pandas as pd
import pytest

from ClearMap.Utils.exceptions import ClearMapRuntimeError
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


class FakeWebView:
    def setHtml(self, html):
        self.html = html


def _html_of(figure):
    return f'<html>{figure}</html>'


def _fake_module(name, **attributes):
    module = ModuleType(name)
    module.__dict__.update(attributes)
    return module


@pytest.fixture()
def fake_qt(monkeypatch):
    """The Qt and mpld3 modules of run_plots, without a display"""
    monkeypatch.setitem(sys.modules, 'qtpy', _fake_module('qtpy', __path__=[]))
    monkeypatch.setitem(sys.modules, 'qtpy.QtWidgets', _fake_module('qtpy.QtWidgets', QApplication=FakeApplication))
    monkeypatch.setitem(sys.modules, 'qtpy.QtWebEngineWidgets',
                        _fake_module('qtpy.QtWebEngineWidgets', QWebEngineView=FakeWebView))
    monkeypatch.setitem(sys.modules, 'mpld3', _fake_module('mpld3', fig_to_html=_html_of))
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


def test_the_group_plots_are_web_views_in_an_application_not_named_clearmap(fake_qt, analysis_with_stats_table):
    fake_qt.current = FakeApplication()
    views = analysis_with_stats_table.run_plots(plot_the_number_of_rows, [GROUP_NAMES], channel='cells',
                                                plot_kw_args={'group_names': None})
    assert [view.html for view in views] == ['<html>figure of 2 rows</html>']


def test_the_group_plots_ask_for_an_application_when_there_is_none(fake_qt, analysis_with_stats_table):
    with pytest.raises(ClearMapRuntimeError, match='QApplication'):
        analysis_with_stats_table.run_plots(plot_the_number_of_rows, [GROUP_NAMES], channel='cells',
                                            plot_kw_args={'group_names': None})
