"""
The reference channel is a registration choice (registration.reference_channel): the channel aligned with the atlas,
which the other channels are aligned with by default. It is the only autofluorescence channel by default; without
one (e.g. dapi aligned with the atlas and fos with dapi) the user chooses it. The atlas side (atlas resolution, ...)
reads it.
"""
import pytest

from ClearMap.Utils.exceptions import MissingRequirementException
from ClearMap.config.config_adjusters.patch_ops import DELETE
from ClearMap.config.registration_channels import REFERENCE_UNDECIDED


def reference_channel(experiment):
    return experiment.get_config_view()['registration']['reference_channel']


def registration(experiment) -> dict:
    """(align_with, moving_channel) of each channel"""
    channels = experiment.get_config_view()['registration']['channels']
    return {channel: (config['align_with'], config['moving_channel']) for channel, config in channels.items()}


def choose_reference(experiment, channel):
    experiment.apply_ui_patch({'registration': {'reference_channel': channel}})


def test_the_autofluorescence_channel_is_the_reference_by_default(experiment, channel_editor):
    channel_editor.add('auto', 'autofluorescence')
    channel_editor.add('fos', 'nuclei')

    assert reference_channel(experiment) == 'auto'
    assert registration(experiment) == {'auto': ('atlas', 'other'), 'fos': ('auto', 'other')}


def test_without_autofluorescence_the_user_chooses_the_reference_and_the_others_follow(experiment, channel_editor):
    channel_editor.add('dapi', 'nuclei')
    channel_editor.add('fos', 'nuclei')
    assert reference_channel(experiment) == REFERENCE_UNDECIDED

    choose_reference(experiment, 'dapi')

    assert registration(experiment) == {'dapi': ('atlas', 'other'), 'fos': ('dapi', 'other')}


def test_a_reference_chosen_by_hand_stays_when_an_autofluorescence_channel_comes(experiment, channel_editor):
    channel_editor.add('dapi', 'nuclei')
    choose_reference(experiment, 'dapi')

    channel_editor.add('auto', 'autofluorescence')

    assert reference_channel(experiment) == 'dapi'


def test_a_renamed_reference_channel_stays_the_reference(experiment, channel_editor):
    channel_editor.add('auto', 'autofluorescence')
    channel_editor.add('fos', 'nuclei')

    experiment.apply_ui_patch({'sample': {'$rename': {'channels': {'auto': 'af'}}}})

    assert reference_channel(experiment) == 'af'
    assert registration(experiment)['fos'] == ('af', 'other')


def test_a_removed_reference_channel_is_replaced_by_the_default(experiment, channel_editor):
    channel_editor.add('dapi', 'nuclei')
    channel_editor.add('fos', 'nuclei')
    choose_reference(experiment, 'dapi')

    experiment.apply_ui_patch({'sample': {'channels': {'dapi': DELETE}}})

    assert reference_channel(experiment) == REFERENCE_UNDECIDED


def test_the_atlas_side_reads_the_reference_channel(experiment, channel_editor):
    channel_editor.add('dapi', 'nuclei', with_path=True)
    channel_editor.add('fos', 'nuclei', with_path=True)
    worker = experiment.get_worker('registration', None)
    with pytest.raises(MissingRequirementException, match='reference channel is not chosen'):
        worker.ref_channel_cfg

    choose_reference(experiment, 'dapi')

    assert worker.ref_channel_cfg == experiment.get_config_view()['registration']['channels']['dapi']
    assert experiment.sample_manager.alignment_reference_channel == 'dapi'


def test_a_config_written_before_the_setting_keeps_its_choice_when_loaded(experiment, channel_editor):
    channel_editor.add('auto', 'autofluorescence')
    channel_editor.add('dapi', 'nuclei')
    coordinator = experiment.cfg_coordinator
    registration_config = coordinator.working['registration']
    del registration_config['reference_channel']  # As a config written before the setting existed, where the user
    registration_config['channels']['dapi']['align_with'] = 'atlas'  # aligned dapi with the atlas
    registration_config['channels']['auto']['align_with'] = 'dapi'

    coordinator.submit(sample_manager=experiment.sample_manager)  # What loading a config runs

    assert reference_channel(experiment) == 'dapi'
