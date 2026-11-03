"""
Renaming a channel to the name of another channel is refused: it would take over the settings and the results
of that channel. The other channel must be renamed first.
"""
import pytest

from ClearMap.Utils.events import CfgChanged, ChannelRenamed, ChannelsChanged
from ClearMap.Utils.exceptions import ChannelNameTakenError

CHANNEL_A, CHANNEL_B, FREE_NAME = 'nuclei_a', 'nuclei_b', 'nuclei_free'
COMMIT_FACTS = (CfgChanged, ChannelsChanged, ChannelRenamed)


class FactRecorder:
    def __init__(self, bus):
        self.facts = []
        for fact_type in COMMIT_FACTS:
            bus.subscribe(fact_type, self.on_fact)

    def on_fact(self, event):
        self.facts.append(event)


@pytest.fixture()
def two_channels(channel_editor):
    for name in (CHANNEL_A, CHANNEL_B):
        channel_editor.add(name, 'nuclei', with_path=True)


def _rename(experiment, old: str, new: str) -> None:
    experiment.apply_ui_patch({'sample': {'$rename': {'channels': {old: new}}}})


def test_renaming_onto_an_existing_channel_is_refused_and_changes_nothing(experiment, two_channels):
    config_before = experiment.cfg_coordinator.get_config_view()
    workspace_channels_before = set(experiment.sample_manager.workspace.asset_collections)
    recorder = FactRecorder(experiment._bus)

    with pytest.raises(ChannelNameTakenError):
        _rename(experiment, CHANNEL_A, CHANNEL_B)

    assert experiment.cfg_coordinator.get_config_view() == config_before
    assert set(experiment.sample_manager.workspace.asset_collections) == workspace_channels_before
    assert recorder.facts == []


def test_the_name_is_free_once_the_other_channel_is_renamed_first(experiment, two_channels):
    _rename(experiment, CHANNEL_B, FREE_NAME)
    _rename(experiment, CHANNEL_A, CHANNEL_B)

    assert set(experiment.cfg_coordinator.get_config_view('sample')['channels']) == {CHANNEL_B, FREE_NAME}
