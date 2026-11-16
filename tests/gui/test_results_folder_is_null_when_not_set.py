"""The results folder of the batch and group analysis tabs is null when it is not set, never an empty string"""
from types import SimpleNamespace

import pytest

from ClearMap.gui import params, params_interfaces
from ClearMap.gui.params import BatchProcessingParams
from ClearMap.gui.widgets import GroupsWidgetAdapter
from tests.gui.test_line_edits_are_read_as_text import FakeCheckBox, FakeLineEdit


class FakeGroupsAdapter(GroupsWidgetAdapter):
    """The groups of the tab, without their Qt pages (the registry knows it as a GroupsWidgetAdapter)"""
    def __init__(self, **_):
        self.groups = {}

    def get_value(self):
        return self.groups

    def set_value(self, groups):
        self.groups = groups

    def connect(self, on_changed):
        pass


class FakeBus:
    def __init__(self):
        self.events = []

    def publish(self, event):
        self.events.append(event)


@pytest.fixture
def batch_params(monkeypatch):
    monkeypatch.setattr(params_interfaces, '_is_alive', lambda widget: widget is not None)
    monkeypatch.setattr(params, 'GroupsWidgetAdapter', FakeGroupsAdapter)
    tab = SimpleNamespace(sampleFoldersToolBox=object(), sampleFoldersPageLayout=None, addGroupPushButton=None,
                          removeGroupPushButton=None, resultsFolderLineEdit=FakeLineEdit(),
                          batchAlignCheckBox=FakeCheckBox(), batchCountCellsCheckBox=FakeCheckBox(),
                          batchVasculatureCheckBox=FakeCheckBox())
    patches = []
    batch = BatchProcessingParams(tab, event_bus=FakeBus(), get_view=lambda: {'batch_processing': {}},
                                  apply_patch=patches.append)
    return tab, batch, patches


@pytest.mark.parametrize('text', ['', 'None'])
def test_an_empty_results_folder_field_is_written_null(batch_params, text):
    tab, batch, patches = batch_params
    tab.resultsFolderLineEdit.setText(text)

    batch.handle_widget_changed(attr_name='results_folder')

    assert patches == [{'batch_processing': {'paths': {'results_folder': None}}}]


def test_a_results_folder_is_written_as_typed(batch_params):
    tab, batch, patches = batch_params
    tab.resultsFolderLineEdit.setText('/data/results')

    batch.handle_widget_changed(attr_name='results_folder')

    assert patches == [{'batch_processing': {'paths': {'results_folder': '/data/results'}}}]


@pytest.mark.parametrize('results_folder', [None, ''])  # '': not set, as ClearMap 3.1.x wrote it
def test_a_results_folder_which_is_not_set_is_shown_empty(batch_params, results_folder):
    tab, batch, _ = batch_params
    batch.results_folder = results_folder

    assert tab.resultsFolderLineEdit.text() == ''
