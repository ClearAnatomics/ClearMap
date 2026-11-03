"""
Channel names become file names, and some file systems fold the case: two channels whose names differ only by
case are refused, before anything is written. A channel can still change the case of its own name.
"""
import pytest

from ClearMap.Utils.events import CfgChanged, ChannelRenamed, ChannelsChanged
from ClearMap.Utils.exceptions import AggregatedValidationError, ChannelNameTakenError

CHANNEL_A, CHANNEL_B = 'nuclei_a', 'nuclei_b'
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


def _sample_channel_names(experiment) -> set[str]:
    return set(experiment.cfg_coordinator.get_config_view('sample')['channels'])


def test_adding_a_channel_which_differs_only_by_case_is_refused_and_changes_nothing(
        experiment, channel_editor, two_channels):
    config_before = experiment.cfg_coordinator.get_config_view()
    recorder = FactRecorder(experiment._bus)

    with pytest.raises(AggregatedValidationError, match='differing only by case'):
        channel_editor.add(CHANNEL_A.upper())

    assert experiment.cfg_coordinator.get_config_view() == config_before
    assert recorder.facts == []


def test_renaming_onto_a_name_which_differs_only_by_case_is_refused_and_changes_nothing(experiment, two_channels):
    config_before = experiment.cfg_coordinator.get_config_view()
    recorder = FactRecorder(experiment._bus)

    with pytest.raises(ChannelNameTakenError):
        experiment.apply_ui_patch({'sample': {'$rename': {'channels': {CHANNEL_A: CHANNEL_B.upper()}}}})

    assert experiment.cfg_coordinator.get_config_view() == config_before
    assert recorder.facts == []


def test_a_channel_can_change_the_case_of_its_own_name(experiment, two_channels):
    own_name_in_capitals = CHANNEL_A.upper()

    experiment.apply_ui_patch({'sample': {'$rename': {'channels': {CHANNEL_A: own_name_in_capitals}}}})

    assert _sample_channel_names(experiment) == {own_name_in_capitals, CHANNEL_B}
