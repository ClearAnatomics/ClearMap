"""
A committed config always went through the adjusters and the validators: they cannot be skipped
(there is no flag for it), neither when a patch is submitted nor when the working config is.
"""
import pytest

import ClearMap.config.config_coordinator as config_coordinator_module

CHANNEL = 'nuclei_a'
SKIPPING_FLAGS = ('do_run_adjusters', 'validate', 'commit')


class CheckRecorder:
    def __init__(self, monkeypatch):
        self.adjusters_runs = 0
        self.validators_runs = 0
        run_adjusters, validate_all = config_coordinator_module.run_adjusters, config_coordinator_module.validate_all

        def counting_adjusters(*args, **kwargs):
            self.adjusters_runs += 1
            return run_adjusters(*args, **kwargs)

        def counting_validators(*args, **kwargs):
            self.validators_runs += 1
            return validate_all(*args, **kwargs)

        monkeypatch.setattr(config_coordinator_module, 'run_adjusters', counting_adjusters)
        monkeypatch.setattr(config_coordinator_module, 'validate_all', counting_validators)


@pytest.fixture()
def recorder(experiment, channel_editor, monkeypatch):
    channel_editor.add(CHANNEL, 'nuclei', with_path=True)
    return CheckRecorder(monkeypatch)


def test_a_submitted_patch_runs_the_adjusters_and_the_validators(experiment, recorder):
    experiment.apply_ui_patch({'sample': {'channels': {CHANNEL: {'comments': 'an edit'}}}})

    assert recorder.adjusters_runs > 0
    assert recorder.validators_runs > 0


def test_a_submitted_config_runs_the_adjusters_and_the_validators(experiment, recorder):
    experiment.cfg_coordinator.submit(sample_manager=experiment.sample_manager)

    assert recorder.adjusters_runs > 0
    assert recorder.validators_runs > 0


@pytest.mark.parametrize('flag', SKIPPING_FLAGS)
def test_there_is_no_flag_to_skip_the_checks(experiment, recorder, flag):
    coordinator = experiment.cfg_coordinator
    with pytest.raises(TypeError):
        coordinator.submit_patch({'sample': {'comments': 'x'}}, sample_manager=experiment.sample_manager, **{flag: False})
    with pytest.raises(TypeError):
        coordinator.submit(sample_manager=experiment.sample_manager, **{flag: False})
