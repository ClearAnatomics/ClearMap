"""
What a channel is aligned with by default (a pure function of the data types of the channels of the sample), and how
the channels which still have it follow the defaults when the data types change.
"""
import pytest

from ClearMap.config.registration_channels import (default_registration_partners_and_directions,
                                                   new_defaults_of_unedited_registrations)


def test_the_reference_channel_is_aligned_with_the_atlas_and_the_others_with_it():
    data_types = {'auto': 'autofluorescence', 'fos': 'nuclei', 'raw': 'undefined'}

    defaults = default_registration_partners_and_directions(data_types)

    assert defaults == {'auto': ('atlas', 'other'), 'fos': ('auto', 'other'), 'raw': ('auto', 'other')}


@pytest.mark.parametrize('data_types', [{'fos': 'nuclei'}, {'fos': 'nuclei', 'raw': 'undefined'},
                                        {'a': 'autofluorescence', 'b': 'autofluorescence', 'fos': 'nuclei'}])
def test_without_a_single_reference_channel_the_others_are_aligned_with_nothing_never_with_themselves(data_types):
    defaults = default_registration_partners_and_directions(data_types)

    assert all(align_with is None for channel, (align_with, _) in defaults.items() if data_types[channel] == 'nuclei')
    assert all(align_with != channel for channel, (align_with, _) in defaults.items())


def test_the_vessel_parts_other_than_the_all_vessels_channel_are_acquired_with_it():
    data_types = {'auto': 'autofluorescence', 'vessels': 'vessels', 'arteries': 'arteries', 'veins': 'veins',
                  'fos': 'nuclei'}

    defaults = default_registration_partners_and_directions(data_types)

    assert defaults == {'auto': ('atlas', 'other'), 'vessels': ('auto', 'other'), 'fos': ('auto', 'other'),
                        'arteries': ('vessels', 'intrinsically_aligned'), 'veins': ('vessels', 'intrinsically_aligned')}


@pytest.mark.parametrize('data_types', [{'arteries': 'arteries', 'auto': 'autofluorescence'},  # No all-vessels channel
                                        {'v1': 'vessels', 'v2': 'vessels', 'arteries': 'arteries',  # Several
                                         'auto': 'autofluorescence'}])
def test_without_exactly_one_all_vessels_channel_the_vessel_parts_are_registered_like_the_others(data_types):
    assert default_registration_partners_and_directions(data_types)['arteries'] == ('auto', 'other')


def test_a_channel_aligned_as_the_defaults_said_follows_the_new_defaults():
    before = {'fos': 'nuclei'}
    after = {'fos': 'nuclei', 'auto': 'autofluorescence'}
    channels = {'fos': {'align_with': None, 'moving_channel': 'other'}}

    assert new_defaults_of_unedited_registrations(channels, before, after) == {
        'fos': {'align_with': 'auto', 'moving_channel': 'other'}}


def test_a_channel_set_by_hand_is_left_alone():
    before = {'fos': 'nuclei'}
    after = {'fos': 'nuclei', 'auto': 'autofluorescence'}

    for by_hand in ({'align_with': 'atlas', 'moving_channel': 'other'}, {'align_with': None, 'moving_channel': None},
                    {'align_with': None, 'moving_channel': 'self'}):
        assert new_defaults_of_unedited_registrations({'fos': by_hand}, before, after) == {}


def test_a_channel_which_is_new_or_gone_is_not_followed():
    channels = {'new': {'align_with': None, 'moving_channel': 'other'}}

    assert new_defaults_of_unedited_registrations(channels, {}, {'new': 'nuclei', 'auto': 'autofluorescence'}) == {}
    assert new_defaults_of_unedited_registrations(channels, {'new': 'nuclei'}, {'auto': 'autofluorescence'}) == {}


def test_nothing_changes_when_the_defaults_do_not():
    channels = {'fos': {'align_with': 'auto', 'moving_channel': 'other'}}
    types = {'fos': 'nuclei', 'auto': 'autofluorescence'}

    assert new_defaults_of_unedited_registrations(channels, types, {**types, 'other': 'nuclei'}) == {}
