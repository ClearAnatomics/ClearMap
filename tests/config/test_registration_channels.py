"""The registrations of the channels, read from the registration config"""
import pytest

from ClearMap.Utils.exceptions import MissingRequirementException
from ClearMap.config.registration_channels import (fixed_moving_channels, is_explicitly_not_registered,
                                                   is_intrinsically_aligned, is_selected_for_registration,
                                                   may_be_registered, moving_channel_of, needs_registration,
                                                   registration_pairs, registration_sequence)


def config(**channels):
    return {'channels': {name: {'align_with': align_with, 'moving_channel': moving}
                         for name, (align_with, moving) in channels.items()}}


def test_the_atlas_moves_onto_the_reference_channel():
    assert fixed_moving_channels(config(auto=('atlas', 'atlas')), 'auto') == ('auto', 'atlas')


def test_a_channel_moves_onto_what_it_is_aligned_with():
    assert fixed_moving_channels(config(fos=('auto', 'fos')), 'fos') == ('auto', 'fos')


def test_a_channel_aligned_with_nothing_has_no_fixed_channel():
    assert fixed_moving_channels(config(fos=(None, 'fos')), 'fos') == (None, 'fos')


def test_the_pair_of_a_channel_whose_partner_is_undecided_is_refused():
    with pytest.raises(MissingRequirementException):
        fixed_moving_channels(config(fos=('', 'fos')), 'fos')


@pytest.mark.parametrize('align_with, selected', [('auto', True), ('atlas', True), (None, False), ('', False)])
def test_a_channel_is_selected_when_aligned_with_something(align_with, selected):
    assert is_selected_for_registration({'align_with': align_with, 'moving_channel': 'fos'}) is selected


@pytest.mark.parametrize('align_with, explicitly_not, may_be', [('auto', False, True), (None, True, False),
                                                              ('', False, True)])
def test_only_a_channel_aligned_with_nothing_by_choice_may_not_be_registered(align_with, explicitly_not, may_be):
    channel_config = {'align_with': align_with, 'moving_channel': 'fos'}

    assert is_explicitly_not_registered(channel_config) is explicitly_not
    assert may_be_registered(channel_config) is may_be  # E.g. a partner offered to the other channels


@pytest.mark.parametrize('moving_channel, intrinsically', [('intrinsically_aligned', True), ('fos', False),
                                                           (None, False)])
def test_a_channel_is_intrinsically_aligned_when_it_says_so(moving_channel, intrinsically):
    assert is_intrinsically_aligned({'align_with': 'auto', 'moving_channel': moving_channel}) is intrinsically


@pytest.mark.parametrize('moving_channel, needed', [('fos', True), ('atlas', True), (None, False),
                                                    ('intrinsically_aligned', False)])
def test_a_registration_is_needed_when_something_moves_and_is_not_already_aligned(moving_channel, needed):
    assert needs_registration({'align_with': 'auto', 'moving_channel': moving_channel}) is needed


def test_the_pairs_are_moving_then_fixed_in_the_order_of_the_channels():
    registration_config = config(auto=('atlas', 'atlas'), fos=('auto', 'fos'), nuc=('auto', 'auto'))

    assert registration_pairs(registration_config) == [('atlas', 'auto'), ('fos', 'auto'), ('auto', 'nuc')]


def test_channels_without_a_computed_registration_have_no_pair():
    registration_config = config(auto=('atlas', 'atlas'), fos=(None, 'fos'), nuc=('auto', 'intrinsically_aligned'),
                                 tubes=('', 'tubes'), gfp=('auto', None))

    assert registration_pairs(registration_config) == [('atlas', 'auto')]


def test_the_sequence_follows_align_with_until_the_atlas():
    registration_config = config(auto=('atlas', 'atlas'), fos=('auto', 'fos'), gfp=('fos', 'gfp'))

    assert registration_sequence(registration_config, 'gfp') == ['gfp', 'fos', 'auto']


def test_the_sequence_stops_at_a_channel_aligned_with_nothing_or_at_the_stop_channel():
    registration_config = config(auto=(None, None), fos=('auto', 'fos'), gfp=('fos', 'gfp'))

    assert registration_sequence(registration_config, 'gfp') == ['gfp', 'fos', 'auto']
    assert registration_sequence(registration_config, 'gfp', stop_channel='fos') == ['gfp']


@pytest.mark.parametrize('moving_channel, align_with, moving', [('self', 'auto', 'fos'), ('other', 'auto', 'auto'),
                                                                ('other', None, None), ('other', '', None),
                                                                ('atlas', 'atlas', 'atlas'), ('fos', 'auto', 'fos')])
def test_self_and_other_are_resolved_to_the_name_of_the_moving_channel(moving_channel, align_with, moving):
    assert moving_channel_of(config(fos=(align_with, moving_channel)), 'fos') == moving


def test_self_moves_the_channel_onto_its_partner():
    assert fixed_moving_channels(config(fos=('auto', 'self')), 'fos') == ('auto', 'fos')


def test_other_moves_the_partner_onto_the_channel():
    assert fixed_moving_channels(config(fos=('auto', 'other')), 'fos') == ('fos', 'auto')


def test_the_pairs_of_self_and_other_are_named_after_the_channels():
    registration_config = config(auto=('atlas', 'other'), fos=('auto', 'self'))

    assert registration_pairs(registration_config) == [('atlas', 'auto'), ('fos', 'auto')]
