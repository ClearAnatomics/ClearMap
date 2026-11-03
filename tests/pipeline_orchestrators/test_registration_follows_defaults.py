"""
The registration of a channel is created when the channel is, before the user told its data type and long before the
other channels exist: it follows the defaults (what the channels are aligned with) as the user completes the sample.
"""
import pytest

from ClearMap.Utils.exceptions import MissingRequirementException
from ClearMap.config.registration_channels import ALIGN_WITH_UNDECIDED


def registration(experiment) -> dict:
    """(align_with, moving_channel) of each channel"""
    channels = experiment.get_config_view()['registration']['channels']
    return {channel: (config['align_with'], config['moving_channel']) for channel, config in channels.items()}


def test_a_channel_which_becomes_the_reference_is_aligned_with_the_atlas(experiment, channel_editor):
    channel_editor.add('auto')  # The channel exists before the user tells what it is
    channel_editor.set_data_type('auto', 'autofluorescence')

    assert registration(experiment) == {'auto': ('atlas', 'other')}


def test_the_channels_follow_the_reference_when_it_appears(experiment, channel_editor):
    channel_editor.add('fos', 'nuclei')
    assert registration(experiment) == {'fos': (ALIGN_WITH_UNDECIDED, 'other')}  # To choose, never itself

    channel_editor.add('auto', 'autofluorescence')

    assert registration(experiment) == {'fos': ('auto', 'other'), 'auto': ('atlas', 'other')}


def test_the_arteries_follow_the_vessels_when_they_appear(experiment, channel_editor):
    channel_editor.add('auto', 'autofluorescence')
    channel_editor.add('arteries', 'arteries')
    assert registration(experiment)['arteries'] == ('auto', 'other')

    channel_editor.add('vessels', 'vessels')

    assert registration(experiment) == {'auto': ('atlas', 'other'), 'arteries': ('vessels', 'intrinsically_aligned'),
                                        'vessels': ('auto', 'other')}


def test_a_channel_set_by_hand_keeps_its_registration(experiment, channel_editor):
    channel_editor.add('fos', 'nuclei')
    experiment.apply_ui_patch({'registration': {'channels': {'fos': {'align_with': 'atlas'}}}})

    channel_editor.add('auto', 'autofluorescence')

    assert registration(experiment)['fos'] == ('atlas', 'other')


def test_a_channel_which_changes_its_data_type_follows_the_defaults_of_the_new_one(experiment, channel_editor):
    channel_editor.add('auto', 'autofluorescence')
    channel_editor.add('vessels', 'vessels')
    channel_editor.add('fos', 'nuclei')

    channel_editor.set_data_type('fos', 'arteries')

    assert registration(experiment)['fos'] == ('vessels', 'intrinsically_aligned')


def test_a_renamed_channel_keeps_following(experiment, channel_editor):
    channel_editor.add('fos', 'nuclei')
    experiment.apply_ui_patch({'sample': {'$rename': {'channels': {'fos': 'cfos'}}}})

    channel_editor.add('auto', 'autofluorescence')

    assert registration(experiment)['cfos'] == ('auto', 'other')


def test_the_registration_sequence_of_a_lone_channel_ends(experiment, channel_editor):
    channel_editor.add('fos', 'nuclei')

    config = experiment.get_config_view()['registration']
    from ClearMap.config.registration_channels import registration_sequence
    assert registration_sequence(config, 'fos') == ['fos']


def test_the_alignment_refuses_to_start_while_a_partner_is_not_chosen(experiment, channel_editor, monkeypatch):
    channel_editor.add('auto', 'autofluorescence', with_path=True)
    channel_editor.add('fos', 'nuclei', with_path=True)
    experiment.apply_ui_patch({'registration': {'channels': {'fos': {'align_with': ALIGN_WITH_UNDECIDED}}}})
    registration = experiment.get_worker('registration', None)
    aligned = []
    monkeypatch.setattr(registration, 'align_channel', aligned.append)

    with pytest.raises(MissingRequirementException, match="'fos' is not chosen"):
        registration.align()
    assert aligned == []  # Refused before the first channel, not halfway


def test_the_points_of_a_channel_whose_partner_is_not_chosen_cannot_reach_the_atlas(experiment, channel_editor):
    channel_editor.add('fos', 'nuclei', with_path=True)
    registration = experiment.get_worker('registration', None)

    with pytest.raises(MissingRequirementException, match="'fos' is not chosen"):
        registration.get_transform_directories('fos')


def test_a_channel_whose_partner_is_not_chosen_is_listed_as_not_registered(experiment, channel_editor):
    channel_editor.add('fos', 'nuclei', with_path=True)  # Undecided: the GUI lists the registered channels on setup
    registration = experiment.get_worker('registration', None)
    registration.setup_if_needed()

    assert registration.channel_was_registered('fos') is False
    registration.get('aligned', channel='fos')  # Nothing to parametrize: no registration is planned yet
