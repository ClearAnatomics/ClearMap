"""
What the config must say for channels to match pixel for pixel (the parts of a TubeMap compound, a colocalization
pair): acquired together (registration) and stitched together (stitching).
"""
import pytest

from ClearMap.Utils.exceptions import ChannelsNotPixelAlignedError, ClearMapConfigError
from ClearMap.config.registration_channels import check_intrinsically_aligned
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


def test_channels_registered_each_on_its_own_are_not():
    # Their results end at the same place, but not pixel for pixel
    with pytest.raises(ChannelsNotPixelAlignedError, match=r"got root_channels=\['vessels', 'arteries'\]"):
        check_intrinsically_aligned(registration(vessels=('auto', 'other'), arteries=('auto', 'other')),
                                    ['vessels', 'arteries'])


def test_a_channel_acquired_together_with_another_channel_than_them_is_not():
    with pytest.raises(ChannelsNotPixelAlignedError, match="arteries is acquired together with 'auto'"):
        check_intrinsically_aligned(registration(vessels=('auto', 'other'), arteries=('auto', 'intrinsically_aligned')),
                                    ['vessels', 'arteries'])


def test_channels_acquired_together_with_each_other_but_not_with_the_one_registered_are_not():
    registration_config = registration(vessels=('auto', 'other'), arteries=('veins', 'intrinsically_aligned'),
                                       veins=('arteries', 'intrinsically_aligned'))
    with pytest.raises(ChannelsNotPixelAlignedError, match='loop: arteries -> veins -> arteries'):
        check_intrinsically_aligned(registration_config, ['vessels', 'arteries', 'veins'])


def test_all_the_requirements_which_are_not_met_are_named():
    registration_config = registration(vessels=('auto', 'intrinsically_aligned'),
                                       arteries=('auto', 'intrinsically_aligned'))
    with pytest.raises(ChannelsNotPixelAlignedError) as error:
        check_intrinsically_aligned(registration_config, ['vessels', 'arteries'])

    assert len(error.value.unmet_requirements) == 3 and error.value.channels == ('vessels', 'arteries')


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
