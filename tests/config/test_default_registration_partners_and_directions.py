"""
What a channel is aligned with by default (a pure function of the data types of the channels of the sample and of the
reference channel), which channel is the reference by default, and how the unedited ones follow when they change.
"""
import pytest

from ClearMap.Utils.exceptions import MissingRequirementException
from ClearMap.config.registration_channels import (NO_REFERENCE, REFERENCE_UNDECIDED, RegistrationState,
                                                   checked_reference_channel,
                                                   default_reference_channel,
                                                   default_registration_partners_and_directions,
                                                   new_defaults_of_unedited_registrations,
                                                   reference_channel_after_change, reference_channel_setting,
                                                   registration_changes_to_follow_defaults)


# ----------------------------------------------------------------------------------- the reference channel

def test_the_reference_channel_by_default_is_the_only_autofluorescence_channel():
    assert default_reference_channel({'auto': 'autofluorescence', 'fos': 'nuclei'}) == 'auto'


@pytest.mark.parametrize('data_types', [{'dapi': 'nuclei', 'fos': 'nuclei'},
                                        {'a': 'autofluorescence', 'b': 'autofluorescence'}])
def test_without_a_single_autofluorescence_channel_the_reference_is_to_choose(data_types):
    assert default_reference_channel(data_types) == REFERENCE_UNDECIDED


def test_the_setting_of_the_config_is_the_reference():
    data_types = {'auto': 'autofluorescence', 'fos': 'nuclei'}

    assert reference_channel_setting({'reference_channel': 'fos', 'channels': {}}, data_types) == 'fos'
    assert reference_channel_setting({'reference_channel': None, 'channels': {}}, data_types) is NO_REFERENCE


def test_a_config_written_before_the_setting_keeps_the_channel_it_aligns_with_the_atlas():
    data_types = {'auto': 'autofluorescence', 'dapi': 'nuclei', 'fos': 'nuclei'}
    channels = {'auto': {'align_with': 'dapi'}, 'dapi': {'align_with': 'atlas'}, 'fos': {'align_with': 'dapi'}}

    assert reference_channel_setting({'channels': channels}, data_types) == 'dapi'  # Not the autofluorescence one


@pytest.mark.parametrize('channels', [{}, {'auto': {'align_with': 'atlas'}, 'fos': {'align_with': 'atlas'}}])
def test_a_config_written_before_the_setting_without_one_channel_on_the_atlas_has_the_default(channels):
    data_types = {'auto': 'autofluorescence', 'fos': 'nuclei'}

    assert reference_channel_setting({'channels': channels}, data_types) == 'auto'


def test_an_unedited_reference_follows_the_data_types():
    assert reference_channel_after_change(reference_before=REFERENCE_UNDECIDED, reference_now=REFERENCE_UNDECIDED,
                                          data_types_before={'fos': 'nuclei'},
                                          data_types_after={'fos': 'nuclei', 'auto': 'autofluorescence'}) == 'auto'


@pytest.mark.parametrize('reference_now', ['dapi', NO_REFERENCE])
def test_a_reference_chosen_by_hand_stays(reference_now):
    data_types = {'dapi': 'nuclei', 'fos': 'nuclei', 'auto': 'autofluorescence'}

    assert reference_channel_after_change(reference_before=REFERENCE_UNDECIDED, reference_now=reference_now,
                                          data_types_before={'dapi': 'nuclei', 'fos': 'nuclei'},
                                          data_types_after=data_types) == reference_now


def test_a_reference_channel_which_is_gone_is_replaced_by_the_default():
    assert reference_channel_after_change(reference_before='dapi', reference_now='dapi',
                                          data_types_before={'dapi': 'nuclei', 'fos': 'nuclei'},
                                          data_types_after={'fos': 'nuclei'}) == REFERENCE_UNDECIDED


# ----------------------------------------------------------------------------------- the partners

def test_the_reference_channel_is_aligned_with_the_atlas_and_the_others_with_it():
    data_types = {'auto': 'autofluorescence', 'fos': 'nuclei', 'raw': 'undefined'}

    defaults = default_registration_partners_and_directions(data_types, 'auto')

    assert defaults == {'auto': ('atlas', 'other'), 'fos': ('auto', 'other'), 'raw': ('auto', 'other')}


def test_any_channel_can_be_the_reference():
    defaults = default_registration_partners_and_directions({'dapi': 'nuclei', 'fos': 'nuclei'}, 'dapi')

    assert defaults == {'dapi': ('atlas', 'other'), 'fos': ('dapi', 'other')}


@pytest.mark.parametrize('reference_channel', [REFERENCE_UNDECIDED, NO_REFERENCE])
def test_without_a_reference_channel_the_others_have_its_state_never_themselves(reference_channel):
    data_types = {'auto': 'autofluorescence', 'fos': 'nuclei'}

    defaults = default_registration_partners_and_directions(data_types, reference_channel)

    assert defaults == {'auto': (reference_channel, 'other'), 'fos': (reference_channel, 'other')}


def test_a_reference_which_is_not_a_channel_of_the_sample_is_to_choose():
    defaults = default_registration_partners_and_directions({'fos': 'nuclei'}, 'gone')

    assert defaults == {'fos': (REFERENCE_UNDECIDED, 'other')}


def test_the_vessel_parts_other_than_the_all_vessels_channel_are_acquired_with_it():
    data_types = {'auto': 'autofluorescence', 'vessels': 'vessels', 'arteries': 'arteries', 'veins': 'veins',
                  'fos': 'nuclei'}

    defaults = default_registration_partners_and_directions(data_types, 'auto')

    assert defaults == {'auto': ('atlas', 'other'), 'vessels': ('auto', 'other'), 'fos': ('auto', 'other'),
                        'arteries': ('vessels', 'intrinsically_aligned'), 'veins': ('vessels', 'intrinsically_aligned')}


@pytest.mark.parametrize('data_types', [{'arteries': 'arteries', 'auto': 'autofluorescence'},  # No all-vessels channel
                                        {'v1': 'vessels', 'v2': 'vessels', 'arteries': 'arteries',  # Several
                                         'auto': 'autofluorescence'}])
def test_without_exactly_one_all_vessels_channel_the_vessel_parts_are_registered_like_the_others(data_types):
    assert default_registration_partners_and_directions(data_types, 'auto')['arteries'] == ('auto', 'other')


# ----------------------------------------------------------------------------------- following the defaults

def test_a_channel_aligned_as_the_defaults_said_follows_the_new_reference():
    types = {'fos': 'nuclei', 'dapi': 'nuclei'}
    channels = {'fos': {'align_with': REFERENCE_UNDECIDED, 'moving_channel': 'other'}}

    changes = new_defaults_of_unedited_registrations(channels, data_types_before=types,
                                                     reference_before=REFERENCE_UNDECIDED,
                                                     data_types_after=types, reference_after='dapi')

    assert changes == {'fos': {'align_with': 'dapi', 'moving_channel': 'other'}}


def test_a_channel_set_by_hand_is_left_alone():
    before = {'fos': 'nuclei'}
    after = {'fos': 'nuclei', 'auto': 'autofluorescence'}

    for by_hand in ({'align_with': 'atlas', 'moving_channel': 'other'}, {'align_with': None, 'moving_channel': None},
                    {'align_with': None, 'moving_channel': 'self'}):
        assert new_defaults_of_unedited_registrations({'fos': by_hand}, data_types_before=before,
                                                      reference_before=REFERENCE_UNDECIDED,
                                                      data_types_after=after, reference_after='auto') == {}


def test_a_channel_which_is_new_or_gone_is_not_followed():
    channels = {'new': {'align_with': None, 'moving_channel': 'other'}}
    after = {'new': 'nuclei', 'auto': 'autofluorescence'}

    assert new_defaults_of_unedited_registrations(channels, data_types_before={}, reference_before=REFERENCE_UNDECIDED,
                                                  data_types_after=after, reference_after='auto') == {}
    assert new_defaults_of_unedited_registrations(channels, data_types_before={'new': 'nuclei'},
                                                  reference_before=REFERENCE_UNDECIDED,
                                                  data_types_after={'auto': 'autofluorescence'},
                                                  reference_after='auto') == {}


def test_nothing_changes_when_the_defaults_do_not():
    channels = {'fos': {'align_with': 'auto', 'moving_channel': 'other'}}
    types = {'fos': 'nuclei', 'auto': 'autofluorescence'}

    assert new_defaults_of_unedited_registrations(channels, data_types_before=types, reference_before='auto',
                                                  data_types_after={**types, 'other': 'nuclei'},
                                                  reference_after='auto') == {}


# ----------------------------------------------------------------------------------- what the atlas side needs

def test_the_atlas_side_takes_a_reference_aligned_with_the_atlas():
    registration = {'channels': {'dapi': {'align_with': 'atlas'}, 'fos': {'align_with': 'dapi'}}}

    assert checked_reference_channel(registration, 'dapi') == 'dapi'


@pytest.mark.parametrize('reference_channel, message', [(REFERENCE_UNDECIDED, 'not chosen'),
                                                        (NO_REFERENCE, 'no channel is aligned with the atlas'),
                                                        ('fos', "'fos' must be aligned with the atlas")])
def test_the_atlas_side_refuses_a_reference_it_cannot_use(reference_channel, message):
    registration = {'channels': {'dapi': {'align_with': 'atlas'}, 'fos': {'align_with': 'dapi'}}}

    with pytest.raises(MissingRequirementException, match=message):
        checked_reference_channel(registration, reference_channel)


# ----------------------------------------------------------------------------------- the changes, all together

def test_a_loaded_config_only_gets_its_missing_reference_channel():
    data_types = {'auto': 'autofluorescence', 'fos': 'nuclei'}
    registration = {'channels': {'auto': {'align_with': 'atlas', 'moving_channel': 'other'},
                                 'fos': {'align_with': 'auto', 'moving_channel': 'other'}}}

    assert registration_changes_to_follow_defaults(RegistrationState(registration, data_types), None) == {
        'reference_channel': 'auto'}
    loaded_with_it = RegistrationState({**registration, 'reference_channel': 'auto'}, data_types)
    assert registration_changes_to_follow_defaults(loaded_with_it, None) == {}


def test_choosing_the_reference_makes_the_unedited_registrations_follow():
    data_types = {'dapi': 'nuclei', 'fos': 'nuclei'}
    undecided = {'align_with': REFERENCE_UNDECIDED, 'moving_channel': 'other'}
    before = {'reference_channel': REFERENCE_UNDECIDED, 'channels': {'dapi': undecided, 'fos': undecided}}
    now = {**before, 'reference_channel': 'dapi'}

    changes = registration_changes_to_follow_defaults(RegistrationState(now, data_types),
                                                      RegistrationState(before, data_types))

    assert changes == {'channels': {'dapi': {'align_with': 'atlas', 'moving_channel': 'other'},
                                    'fos': {'align_with': 'dapi', 'moving_channel': 'other'}}}
