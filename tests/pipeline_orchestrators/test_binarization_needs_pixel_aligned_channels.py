"""
The vessel channels are binarized one by one then combined voxel for voxel: they must match pixel for pixel. The
setup tells the user what is not set up yet; the binarization refuses to start until it is.
"""
import numpy as np
import pytest

from ClearMap.Utils.exceptions import ChannelsNotPixelAlignedError, MissingRequirementException

RESOLUTION = (1, 1, 2)
SHAPE = (4, 5, 6)


def _write_stitched(experiment, channel, shape=SHAPE):
    sample_manager = experiment.sample_manager
    np.save(sample_manager.get('stitched', channel=channel, sample_id=sample_manager.prefix).path,
            np.zeros(shape, dtype=np.uint16))


def _register_arteries_on_their_own(experiment):
    experiment.apply_ui_patch({'registration': {'channels': {'arteries': {
        'align_with': 'atlas', 'moving_channel': 'other'}}}})


@pytest.fixture()
def vessel_channels(experiment, channel_editor):
    channel_editor.add('vessels', 'vessels', with_path=True, resolution=RESOLUTION)
    channel_editor.add('arteries', 'arteries', with_path=True, resolution=RESOLUTION)


@pytest.fixture()
def binarized(monkeypatch):
    """The channels the binarization processed"""
    from ClearMap.pipeline_orchestrators.tube_map import BinaryVesselProcessor
    channels = []
    monkeypatch.setattr(BinaryVesselProcessor, 'binarize_channel', lambda self, channel: channels.append(channel))
    return channels


def test_the_setup_tells_what_is_not_set_up_yet_without_refusing(experiment, vessel_channels):
    with pytest.warns(UserWarning, match=r"stitched images of \['arteries', 'vessels'\]"):
        worker = experiment.get_worker('vasculature', substep='binary')

    assert worker.channels_to_binarize == ('arteries', 'vessels')


def test_channels_registered_each_on_its_own_are_not_binarized(experiment, vessel_channels, binarized):
    _register_arteries_on_their_own(experiment)
    for channel in ('vessels', 'arteries'):
        _write_stitched(experiment, channel)
    with pytest.warns(UserWarning):
        worker = experiment.get_worker('vasculature', substep='binary')

    with pytest.raises(ChannelsNotPixelAlignedError, match='exactly one channel must be the root'):
        worker.binarize()

    assert binarized == []


def test_channels_of_other_shapes_are_not_binarized(experiment, vessel_channels, binarized):
    _write_stitched(experiment, 'vessels')
    _write_stitched(experiment, 'arteries', shape=(4, 5, 7))
    with pytest.warns(UserWarning, match='stitched shape'):
        worker = experiment.get_worker('vasculature', substep='binary')

    with pytest.raises(ChannelsNotPixelAlignedError, match='the same stitched shape'):
        worker.binarize()

    assert binarized == []


def test_the_binarization_needs_the_stitched_images(experiment, vessel_channels, binarized):
    with pytest.warns(UserWarning, match='stitched images'):
        worker = experiment.get_worker('vasculature', substep='binary')

    with pytest.raises(MissingRequirementException, match='run the stitching'):
        worker.binarize()

    assert binarized == []


def test_channels_imaged_together_are_binarized(experiment, vessel_channels, binarized):
    for channel in ('vessels', 'arteries'):
        _write_stitched(experiment, channel)
    worker = experiment.get_worker('vasculature', substep='binary')

    worker.binarize()

    assert binarized == ['arteries', 'vessels']


def test_a_single_vessel_channel_has_nothing_to_match(experiment, channel_editor, binarized):
    channel_editor.add('vessels', 'vessels', with_path=True, resolution=RESOLUTION)
    worker = experiment.get_worker('vasculature', substep='binary')

    worker.binarize()

    assert binarized == ['vessels']
