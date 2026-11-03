"""
Channels processed pixel by pixel together (the parts of a TubeMap compound, a colocalization pair) must be set up
to match pixel for pixel: same atlas settings, resolution and stitched shape, acquired together, stitched together.
"""
import numpy as np
import pytest

from ClearMap.Utils.exceptions import ChannelsNotPixelAlignedError, MissingRequirementException

RESOLUTION = (1, 1, 2)
SHAPE = (4, 5, 6)


def _patch_channel(experiment, section, channel, channel_patch):
    experiment.apply_ui_patch({section: {'channels': {channel: channel_patch}}})


def _write_stitched(experiment, channel, shape=SHAPE):
    sample_manager = experiment.sample_manager
    np.save(sample_manager.get('stitched', channel=channel, sample_id=sample_manager.prefix).path,
            np.zeros(shape, dtype=np.uint16))


@pytest.fixture()
def imaged_together(experiment, channel_editor):
    """The vessels and the arteries, acquired together"""
    channel_editor.add('vessels', 'vessels', with_path=True, resolution=RESOLUTION)
    channel_editor.add('arteries', 'arteries', with_path=True, resolution=RESOLUTION)
    _patch_channel(experiment, 'registration', 'arteries',
                   {'align_with': 'vessels', 'moving_channel': 'intrinsically_aligned'})
    return experiment.sample_manager


def test_channels_imaged_together_are_pixel_aligned(experiment, imaged_together):
    for channel in ('vessels', 'arteries'):
        _write_stitched(experiment, channel)

    imaged_together.check_pixel_aligned('arteries', 'vessels', require_stitched=True)


def test_before_the_stitching_the_shapes_are_not_compared_but_the_user_is_told(imaged_together):
    with pytest.warns(UserWarning, match=r"stitched images of \['arteries', 'vessels'\]"):
        imaged_together.check_pixel_aligned('arteries', 'vessels')


def test_a_computation_needs_the_stitched_images(imaged_together):
    with pytest.raises(MissingRequirementException, match=r"run the stitching .* of these channels first"):
        imaged_together.check_pixel_aligned('arteries', 'vessels', require_stitched=True)


@pytest.mark.parametrize('setting, value', [('resolution', [1, 1, 3]), ('orientation', [-1, 2, 3]),
                                            ('slicing', {'x': [0, 100], 'y': None, 'z': None})])
def test_channels_with_other_atlas_settings_or_resolution_are_not(experiment, imaged_together, setting, value):
    _patch_channel(experiment, 'sample', 'arteries', {setting: value})

    with pytest.raises(ChannelsNotPixelAlignedError, match=f'the same {setting}, got arteries: '):
        imaged_together.check_pixel_aligned('arteries', 'vessels')


def test_channels_of_other_stitched_shapes_are_not(experiment, imaged_together):
    _write_stitched(experiment, 'vessels')
    _write_stitched(experiment, 'arteries', shape=(4, 5, 7))

    with pytest.raises(ChannelsNotPixelAlignedError, match=r'the same stitched shape, got arteries: \(4, 5, 7\)'):
        imaged_together.check_pixel_aligned('arteries', 'vessels')


def test_channels_registered_each_on_its_own_are_not(experiment, imaged_together):
    _patch_channel(experiment, 'registration', 'arteries', {'moving_channel': 'other'})

    with pytest.raises(ChannelsNotPixelAlignedError, match='exactly one channel must be the root'):
        imaged_together.check_pixel_aligned('arteries', 'vessels')


def test_all_the_settings_which_differ_are_named_at_once(experiment, imaged_together):
    _patch_channel(experiment, 'sample', 'arteries', {'resolution': [1, 1, 3]})
    _patch_channel(experiment, 'registration', 'arteries', {'moving_channel': 'other'})

    with pytest.raises(ChannelsNotPixelAlignedError) as error:
        imaged_together.check_pixel_aligned('arteries', 'vessels')

    assert len(error.value.unmet_requirements) == 2
