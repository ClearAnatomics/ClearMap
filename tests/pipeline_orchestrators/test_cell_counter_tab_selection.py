"""
Selecting the cell map tab counts the cells of the channels which can have some (are in the workspace), not of
a channel which has a page (it is a channel to detect) but no path yet: its worker is not set up, with no workspace.
"""
from types import SimpleNamespace

from ClearMap.gui.tabs import CellCounterTab

COMPLETE_CHANNEL = 'complete_nuclei'
CHANNEL_WITHOUT_PATH = 'nuclei_without_path'


def _selected_tab(experiment, counted_channels: list):
    """The code of CellCounterTab.on_selected, on the real sample manager (no Qt tab to build here)"""
    sample_manager = experiment.sample_manager

    def get_channels():  # The channels with a page: all those to detect, complete or not
        return sample_manager.channels_to_detect

    return SimpleNamespace(sample_manager=sample_manager, main_window=SimpleNamespace(print_warning_msg=print),
                           _get_channels=get_channels, update_cell_number=counted_channels.append)


def test_selecting_the_tab_counts_cells_only_of_channels_which_are_in_the_workspace(experiment, channel_editor):
    channel_editor.add(COMPLETE_CHANNEL, 'nuclei', with_path=True)
    channel_editor.add(CHANNEL_WITHOUT_PATH, 'nuclei')
    counted_channels = []
    tab = _selected_tab(experiment, counted_channels)
    assert set(tab._get_channels()) == {COMPLETE_CHANNEL, CHANNEL_WITHOUT_PATH}  # Both have a page

    CellCounterTab.on_selected(tab)

    assert counted_channels == [COMPLETE_CHANNEL]
