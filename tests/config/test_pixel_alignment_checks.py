"""
What the config must say for channels to match pixel for pixel (the parts of a TubeMap compound, a colocalization
pair): acquired together (registration) and stitched together (stitching).
"""
import pytest

from ClearMap.Utils.exceptions import ChannelsNotPixelAlignedError, ClearMapConfigError
from ClearMap.config.registration_channels import acquisition_root, check_intrinsically_aligned
from ClearMap.config.stitching_channels import check_stitched_together, layout_root


def registration(**channels):
    return {'channels': {name: {'align_with': align_with, 'moving_channel': moving}
                         for name, (align_with, moving) in channels.items()}}


def stitching(**layout_channels):
    return {'channels': {name: {'layout_channel': layout_channel} for name, layout_channel in layout_channels.items()}}


# ------------------------------------------------------------------------------------------------- acquired together

@pytest.mark.parametrize('registration_config', [
    registration(vessels=('auto', 'other'), arteries=('vessels', 'intrinsically_aligned')),
    registration(vessels=('auto', 'other'), arteries=('veins', 'intrinsically_aligned'),  # A chain to the one
                 veins=('vessels', 'intrinsically_aligned')),
    registration(vessels=(None, None), arteries=('vessels', 'intrinsically_aligned')),  # Not registered at all
])
def test_channels_acquired_together_with_one_of_them_are_intrinsically_aligned(registration_config):
    check_intrinsically_aligned(registration_config, list(registration_config['channels']))


@pytest.mark.parametrize('channels_acquired_with_auto', [
    ['Podo_CD31', 'SM22'],
    ['SM22', 'Podo_CD31'],
])
def test_channels_both_acquired_with_a_channel_which_is_not_among_them_are_aligned_with_each_other(
        channels_acquired_with_auto):
    registration_config = registration(auto=('atlas', 'other'), Podo_CD31=('auto', 'intrinsically_aligned'),
                                       SM22=('auto', 'intrinsically_aligned'))

    check_intrinsically_aligned(registration_config, channels_acquired_with_auto)


def test_channels_acquired_with_each_other_through_a_chain_ending_outside_of_them_are_aligned_with_each_other():
    registration_config = registration(auto=('atlas', 'other'), Podo_CD31=('auto', 'intrinsically_aligned'),
                                       SM22=('Podo_CD31', 'intrinsically_aligned'))

    check_intrinsically_aligned(registration_config, ['SM22', 'auto'])  # The root is among them
    check_intrinsically_aligned(registration_config, ['SM22', 'Podo_CD31'])  # The root is not


def test_the_root_of_a_channel_is_the_channel_registered_on_its_own_at_the_end_of_its_chain():
    registration_config = registration(auto=('atlas', 'other'), Podo_CD31=('auto', 'intrinsically_aligned'),
                                       SM22=('Podo_CD31', 'intrinsically_aligned'))

    assert [acquisition_root(registration_config, channel) for channel in ('auto', 'Podo_CD31', 'SM22')] == ['auto'] * 3


def test_channels_registered_each_on_its_own_are_not():
    # Their results end at the same place, but not pixel for pixel
    with pytest.raises(ChannelsNotPixelAlignedError, match='with the same channel, got vessels: vessels, '
                                                           'arteries: arteries'):
        check_intrinsically_aligned(registration(vessels=('auto', 'other'), arteries=('auto', 'other')),
                                    ['vessels', 'arteries'])


def test_channels_acquired_with_different_channels_are_not():
    registration_config = registration(auto=('atlas', 'other'), dapi=('atlas', 'other'),
                                       vessels=('auto', 'intrinsically_aligned'),
                                       arteries=('dapi', 'intrinsically_aligned'))
    with pytest.raises(ChannelsNotPixelAlignedError, match='with the same channel, got vessels: auto, arteries: dapi'):
        check_intrinsically_aligned(registration_config, ['vessels', 'arteries'])


@pytest.mark.parametrize('partner', ['atlas', None, '__UNDECIDED__', 'not_a_channel'])
def test_a_channel_acquired_with_what_is_not_a_channel_of_the_sample_is_not_aligned(partner):
    registration_config = registration(vessels=('auto', 'other'), arteries=(partner, 'intrinsically_aligned'))
    with pytest.raises(ChannelsNotPixelAlignedError, match='which is not a channel of the sample'):
        check_intrinsically_aligned(registration_config, ['vessels', 'arteries'])


def test_channels_acquired_together_with_each_other_but_not_with_the_one_registered_are_not():
    registration_config = registration(vessels=('auto', 'other'), arteries=('veins', 'intrinsically_aligned'),
                                       veins=('arteries', 'intrinsically_aligned'))
    with pytest.raises(ChannelsNotPixelAlignedError, match='loop: arteries -> veins -> arteries'):
        check_intrinsically_aligned(registration_config, ['vessels', 'arteries', 'veins'])


def test_all_the_requirements_which_are_not_met_are_named():
    registration_config = registration(vessels=('atlas', 'intrinsically_aligned'),
                                       arteries=(None, 'intrinsically_aligned'))
    with pytest.raises(ChannelsNotPixelAlignedError) as error:
        check_intrinsically_aligned(registration_config, ['vessels', 'arteries'])

    assert len(error.value.unmet_requirements) == 2 and error.value.channels == ('vessels', 'arteries')


# ------------------------------------------------------------------------------------------------- stitched together

def test_the_layout_root_follows_the_layout_channels_to_the_one_which_computes_its_own():
    stitching_config = stitching(vessels='vessels', arteries='vessels', veins='arteries')

    assert [layout_root(stitching_config, channel) for channel in ('vessels', 'arteries', 'veins')] == ['vessels'] * 3


@pytest.mark.parametrize('stitching_config, message', [
    (stitching(vessels='undefined'), 'not chosen'),
    (stitching(vessels='auto'), 'not stitched'),
    (stitching(vessels='arteries', arteries='vessels'), 'loop: vessels -> arteries -> vessels'),
])
def test_a_layout_root_which_cannot_be_told_is_refused(stitching_config, message):
    with pytest.raises(ClearMapConfigError, match=message):
        layout_root(stitching_config, 'vessels')


def test_channels_stitched_with_the_layout_of_the_same_channel_are_stitched_together():
    check_stitched_together(stitching(vessels='vessels', arteries='vessels', veins='arteries'),
                            ['vessels', 'arteries', 'veins'])


def test_channels_stitched_with_their_own_layouts_are_not():
    with pytest.raises(ChannelsNotPixelAlignedError, match='the same layout, got vessels: the layout of vessels, '
                                                           'arteries: the layout of arteries'):
        check_stitched_together(stitching(vessels='vessels', arteries='arteries'), ['vessels', 'arteries'])


def test_the_channels_which_are_not_stitched_are_not_concerned():
    check_stitched_together(stitching(vessels='vessels'), ['vessels', 'arteries'])  # arteries is not tiled
    check_stitched_together({}, ['vessels', 'arteries'])  # No channel is tiled
