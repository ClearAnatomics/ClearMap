"""
A patch refused by the adjusters or the validators leaves no trace: neither in the working config, nor in
the files, nor in the sample manager. Otherwise, the next (valid) edits are refused in its place.
"""
import pytest

import ClearMap.config.config_coordinator as config_coordinator_module
from ClearMap.Utils.exceptions import AggregatedValidationError

CHANNEL_A, CHANNEL_B = 'nuclei_a', 'nuclei_b'
INVALID_RESOLUTION = 'not_a_resolution'
INVALID_CHANNEL_NAME = 'bad-name'  # '-' is the separator of the compound channels
ADJUSTED_COMMENT = 'written by an adjuster'


@pytest.fixture()
def two_channels(channel_editor):
    for name in (CHANNEL_A, CHANNEL_B):
        channel_editor.add(name, 'nuclei', with_path=True)


def _sample_channels(experiment) -> dict:
    return experiment.cfg_coordinator.get_config_view('sample')['channels']


def test_a_refused_value_does_not_stay_in_the_working_config(experiment, two_channels):
    before = _sample_channels(experiment)

    with pytest.raises(AggregatedValidationError):
        experiment.apply_ui_patch({'sample': {'channels': {CHANNEL_A: {'resolution': INVALID_RESOLUTION}}}})

    assert _sample_channels(experiment) == before


def test_a_refused_value_does_not_refuse_the_next_edits(experiment, two_channels):
    with pytest.raises(AggregatedValidationError):
        experiment.apply_ui_patch({'sample': {'channels': {CHANNEL_A: {'resolution': INVALID_RESOLUTION}}}})

    experiment.apply_ui_patch({'cell_map': {'channels': {CHANNEL_B: {'run': {'plot_when_finished': False}}}}})

    cell_map = experiment.cfg_coordinator.get_config_view('cell_map')
    assert cell_map['channels'][CHANNEL_B]['run']['plot_when_finished'] is False


def test_a_refused_rename_does_not_come_back_with_the_next_edit(experiment, two_channels):
    with pytest.raises(ValueError):
        experiment.apply_ui_patch({'sample': {'$rename': {'channels': {CHANNEL_A: INVALID_CHANNEL_NAME}}}})

    assert experiment.sample_manager.renamed_channels == {}
    experiment.apply_ui_patch({'sample': {'channels': {CHANNEL_B: {'comments': 'an unrelated edit'}}}})
    assert set(_sample_channels(experiment)) == {CHANNEL_A, CHANNEL_B}


def _adjusters_writing_a_comment(**_):
    return {'sample': {'comments': ADJUSTED_COMMENT}}


def _validators_refusing_everything(*_, **__):
    raise AggregatedValidationError([])


def test_submit_keeps_no_adjusted_value_when_the_validators_refuse(experiment, two_channels, monkeypatch):
    coordinator = experiment.cfg_coordinator
    before = coordinator.get_config_view()
    monkeypatch.setattr(config_coordinator_module, 'run_adjusters', _adjusters_writing_a_comment)
    monkeypatch.setattr(config_coordinator_module, 'validate_all', _validators_refusing_everything)

    with pytest.raises(AggregatedValidationError):
        coordinator.submit(sample_manager=experiment.sample_manager)

    assert coordinator.get_config_view() == before
