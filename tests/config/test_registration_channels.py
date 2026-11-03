"""The registrations of the channels, read from the registration config"""
import pytest

import itertools

from ClearMap.Utils.exceptions import ClearMapConfigError, MissingRequirementException
from ClearMap.config.registration_channels import (ALIGN_WITH_UNDECIDED, ATLAS, REGISTRATION_DIRECTIONS,
                                                   check_partners_chosen, fixed_moving_channels,
                                                   is_explicitly_not_registered, is_intrinsically_aligned,
                                                   is_selected_for_registration, may_be_registered, needs_registration,
                                                   partner_change, registration_direction, registration_pairs,
                                                   registration_sequence)
from ClearMap.config.validators._semantic import _validate_registration_refs


def config(**channels):
    return {'channels': {name: {'align_with': align_with, 'moving_channel': moving}
                         for name, (align_with, moving) in channels.items()}}


def test_the_atlas_moves_onto_the_reference_channel():
    assert fixed_moving_channels(config(auto=('atlas', 'atlas')), 'auto') == ('auto', 'atlas')


def test_a_channel_moves_onto_what_it_is_aligned_with():
    assert fixed_moving_channels(config(fos=('auto', 'fos')), 'fos') == ('auto', 'fos')


@pytest.mark.parametrize('align_with, moving_channel', [(None, 'fos'), (None, 'other'), ('auto', 'intrinsically_aligned'),
                                                       ('auto', None)])
def test_a_channel_without_a_computed_registration_has_neither_fixed_nor_moving_channel(align_with, moving_channel):
    assert fixed_moving_channels(config(fos=(align_with, moving_channel)), 'fos') == (None, None)


def test_the_pair_of_a_channel_whose_partner_is_undecided_is_refused():
    with pytest.raises(MissingRequirementException):
        fixed_moving_channels(config(fos=(ALIGN_WITH_UNDECIDED, 'fos')), 'fos')


def test_the_channels_whose_partner_is_not_chosen_are_named_together():
    registration_config = config(fos=(ALIGN_WITH_UNDECIDED, 'other'), raw=('auto', 'other'),
                                 dapi=(ALIGN_WITH_UNDECIDED, 'other'), auto=(ATLAS, 'other'))

    with pytest.raises(MissingRequirementException, match="'fos', 'dapi' is not chosen"):
        check_partners_chosen(registration_config, ['fos', 'raw', 'dapi', 'auto'])


@pytest.mark.parametrize('align_with', ['auto', ATLAS, None])  # None is a choice: not registered
def test_a_chosen_partner_passes(align_with):
    check_partners_chosen(config(fos=(align_with, 'other')), ['fos'])


def test_an_empty_partner_is_not_undecided():
    assert ALIGN_WITH_UNDECIDED != ''  # An empty string written by accident is refused by the schema, not taken


@pytest.mark.parametrize('align_with, selected', [('auto', True), ('atlas', True), (None, False),
                                                  (ALIGN_WITH_UNDECIDED, False)])
def test_a_channel_is_selected_when_aligned_with_something(align_with, selected):
    assert is_selected_for_registration({'align_with': align_with, 'moving_channel': 'fos'}) is selected


@pytest.mark.parametrize('align_with, explicitly_not, may_be', [('auto', False, True), (None, True, False),
                                                              (ALIGN_WITH_UNDECIDED, False, True)])
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


@pytest.mark.parametrize('moving_channel', ['self', 'other'])  # The user may choose either
@pytest.mark.parametrize('align_with', [None, ALIGN_WITH_UNDECIDED])
def test_no_registration_is_needed_without_a_partner_whatever_the_direction_kept(align_with, moving_channel):
    assert needs_registration({'align_with': align_with, 'moving_channel': moving_channel}) is False


def test_the_pairs_are_moving_then_fixed_in_the_order_of_the_channels():
    registration_config = config(auto=('atlas', 'atlas'), fos=('auto', 'fos'), nuc=('auto', 'auto'))

    assert registration_pairs(registration_config) == [('atlas', 'auto'), ('fos', 'auto'), ('auto', 'nuc')]


def test_channels_without_a_computed_registration_have_no_pair():
    registration_config = config(auto=('atlas', 'atlas'), fos=(None, 'fos'), nuc=('auto', 'intrinsically_aligned'),
                                 tubes=(ALIGN_WITH_UNDECIDED, 'tubes'), gfp=('auto', None))

    assert registration_pairs(registration_config) == [('atlas', 'auto')]


def test_the_sequence_follows_align_with_until_the_atlas():
    registration_config = config(auto=('atlas', 'atlas'), fos=('auto', 'fos'), gfp=('fos', 'gfp'))

    assert registration_sequence(registration_config, 'gfp') == ['gfp', 'fos', 'auto']


def test_the_sequence_stops_at_a_channel_aligned_with_nothing_or_at_the_stop_channel():
    registration_config = config(auto=(None, None), fos=('auto', 'fos'), gfp=('fos', 'gfp'))

    assert registration_sequence(registration_config, 'gfp') == ['gfp', 'fos', 'auto']
    assert registration_sequence(registration_config, 'gfp', stop_channel='fos') == ['gfp']


@pytest.mark.parametrize('align_with, moving_channel, direction', [
    ('auto', 'self', 'self'), ('auto', 'other', 'other'), ('auto', 'intrinsically_aligned', 'intrinsically_aligned'),
    ('auto', 'fos', 'self'), ('auto', 'auto', 'other'), ('atlas', 'atlas', 'other'),  # The names of older configs
    ('auto', None, None), (None, 'other', None), (ALIGN_WITH_UNDECIDED, 'other', None), (None, 'auto', None)])
def test_the_direction_is_relative_to_the_channel_whatever_way_it_is_written(align_with, moving_channel, direction):
    assert registration_direction('fos', {'align_with': align_with, 'moving_channel': moving_channel}) == direction


def test_a_moving_channel_which_is_neither_the_channel_nor_its_partner_is_refused():
    # E.g. the align_with was changed from auto to the atlas, and the moving_channel still says auto
    with pytest.raises(ClearMapConfigError, match="'auto'"):
        fixed_moving_channels(config(fos=('atlas', 'auto')), 'fos')


@pytest.mark.parametrize('align_with, moving_channel', itertools.product(
    ['auto', ATLAS, None], [*REGISTRATION_DIRECTIONS, 'fos', 'auto', ATLAS, None]))
def test_only_the_names_of_channels_or_the_atlas_name_the_results(align_with, moving_channel):
    # The paths are made of the pair: self, other and intrinsically_aligned must never reach them
    try:
        pair = fixed_moving_channels(config(fos=(align_with, moving_channel), auto=(ATLAS, 'other')), 'fos')
    except ClearMapConfigError:
        return  # Inconsistent: refused, no path
    assert pair == (None, None) or set(pair) <= {'fos', 'auto', ATLAS}


def test_self_moves_the_channel_onto_its_partner():
    assert fixed_moving_channels(config(fos=('auto', 'self')), 'fos') == ('auto', 'fos')


def test_other_moves_the_partner_onto_the_channel():
    assert fixed_moving_channels(config(fos=('auto', 'other')), 'fos') == ('fos', 'auto')


def test_the_pairs_of_self_and_other_are_named_after_the_channels():
    registration_config = config(auto=('atlas', 'other'), fos=('auto', 'self'))

    assert registration_pairs(registration_config) == [('atlas', 'auto'), ('fos', 'auto')]


# ------------------------------------------------------------------------------------- what the validator accepts

def _registration_errors(fos_align_with, fos_moving_channel):
    return _validate_registration_refs({'sample': {'channels': {'fos': {}, 'auto': {}}},
                                        'registration': config(fos=(fos_align_with, fos_moving_channel))})


@pytest.mark.parametrize('moving_channel', [*REGISTRATION_DIRECTIONS, ATLAS, 'auto', 'fos', None])
def test_the_directions_and_the_names_are_valid_moving_channels(moving_channel):
    assert _registration_errors('auto', moving_channel) == []


@pytest.mark.parametrize('align_with', ['self', 'other', 'intrinsically_aligned'])
def test_a_direction_is_not_a_partner(align_with):
    (error,) = _registration_errors(align_with, 'other')

    assert 'align_with' in error and align_with in error


# ------------------------------------------------------------------------------------- the partner is changed

@pytest.mark.parametrize('align_with, moving_channel, new_align_with, changes', [
    ('auto', 'other', 'atlas', {'align_with': 'atlas'}),  # Relative: nothing else to change
    ('auto', 'self', None, {'align_with': None}),  # Kept while it has no partner
    (None, 'self', 'atlas', {'align_with': 'atlas'}),  # And used again with the new one
    ('auto', 'auto', 'atlas', {'align_with': 'atlas', 'moving_channel': 'other'}),  # The name of the old partner
    ('auto', 'fos', 'atlas', {'align_with': 'atlas', 'moving_channel': 'self'}),  # The name of the channel
    ('auto', 'auto', None, {'align_with': None, 'moving_channel': 'other'}),
    ('atlas', 'auto', 'auto', {'align_with': 'auto', 'moving_channel': 'other'}),  # A left-over name: the default
    (None, 'auto', 'atlas', {'align_with': 'atlas', 'moving_channel': 'other'}),
    # A first partner: it moves by default
    (ALIGN_WITH_UNDECIDED, None, 'auto', {'align_with': 'auto', 'moving_channel': 'other'}),
    ('auto', None, ALIGN_WITH_UNDECIDED, {'align_with': ALIGN_WITH_UNDECIDED}),
    ('auto', 'intrinsically_aligned', 'dapi', {'align_with': 'dapi'}),
])
def test_changing_the_partner_keeps_the_direction_relative_to_the_channel(align_with, moving_channel, new_align_with,
                                                                         changes):
    assert partner_change('fos', {'align_with': align_with, 'moving_channel': moving_channel}, new_align_with) == changes


@pytest.mark.parametrize('partner', [None, ALIGN_WITH_UNDECIDED])  # Aligned with nothing, or the partner not chosen yet
def test_the_sequence_of_a_channel_stops_where_a_channel_has_no_partner(partner):
    registration_config = config(fos=('raw', 'other'), raw=(partner, 'other'))

    assert registration_sequence(registration_config, 'fos') == ['fos', 'raw']


@pytest.mark.parametrize('channels, loop', [(dict(fos=('fos', 'other')), 'fos -> fos'),  # Aligned with itself
                                            (dict(a=('b', 'other'), b=('a', 'other')), 'a -> b -> a')])
def test_channels_aligned_in_a_loop_never_reach_the_atlas_and_are_refused(channels, loop):
    with pytest.raises(ClearMapConfigError, match=loop):
        registration_sequence(config(**channels), next(iter(channels)))


@pytest.mark.parametrize('reference_channel, valid', [('auto', True), (ALIGN_WITH_UNDECIDED, True), (None, True),
                                                      ('gone', False)])
def test_the_reference_channel_must_be_a_channel_of_the_sample(reference_channel, valid):
    registration_config = {**config(fos=('auto', 'other')), 'reference_channel': reference_channel}

    errors = _validate_registration_refs({'sample': {'channels': {'fos': {}, 'auto': {}}},
                                          'registration': registration_config})

    assert (errors == []) is valid
