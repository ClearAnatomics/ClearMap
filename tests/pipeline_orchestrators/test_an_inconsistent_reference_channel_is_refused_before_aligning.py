"""
The reference channel and the align_with of the channels are two settings, and they can disagree: the reference
channel 'fos' aligned with 'auto', which is aligned with the atlas (#P0017, the config of #P0018). The registration
ran, then the cell filtering refused the reference channel. The alignment now refuses it before aligning any channel,
naming both settings.
"""
import pytest

from ClearMap.Utils.exceptions import MissingRequirementException
from ClearMap.config.registration_channels import check_reference_selected_as_aligned_with_the_atlas
from ClearMap.pipeline_orchestrators.registration_orchestrator import RegistrationProcessor

# The registration config of #P0018
CHANNELS_OF_P0018 = {'auto': {'align_with': 'atlas', 'moving_channel': 'atlas'},
                     'fos': {'align_with': 'auto', 'moving_channel': 'auto'}}


class FakeSampleManager:
    def __init__(self, reference_channel):
        self.reference_channel_setting = reference_channel


class FakeRegistrationProcessor:
    """What align uses of a RegistrationProcessor; records the channels it aligns"""
    align = RegistrationProcessor.align

    def __init__(self, reference_channel):
        self.stopped = False
        self.config = {'reference_channel': reference_channel, 'channels': CHANNELS_OF_P0018}
        self.channels_to_register = list(CHANNELS_OF_P0018)
        self.sample_manager = FakeSampleManager(reference_channel)
        self.aligned = []

    def align_channel(self, channel):
        self.aligned.append(channel)

    def update_watcher_main_progress(self):
        pass

    def publish(self, event):
        pass


def test_an_alignment_with_a_reference_channel_aligned_with_another_channel_aligns_none():
    processor = FakeRegistrationProcessor(reference_channel='fos')

    with pytest.raises(MissingRequirementException, match="'fos' must be aligned with the atlas"):
        processor.align()

    assert processor.aligned == []


def test_an_alignment_with_the_reference_channel_aligned_with_the_atlas_aligns_all():
    processor = FakeRegistrationProcessor(reference_channel='auto')

    processor.align()

    assert processor.aligned == ['auto', 'fos']


def test_the_message_names_the_reference_channel_its_align_with_and_the_channel_aligned_with_the_atlas():
    with pytest.raises(MissingRequirementException) as error:
        check_reference_selected_as_aligned_with_the_atlas({'channels': CHANNELS_OF_P0018}, 'fos')

    message = str(error.value)
    assert "(reference_channel: 'fos', fos.align_with: 'auto')" in message
    assert "The channel aligned with the atlas is 'auto': choose it as the reference channel" in message


def test_without_a_channel_aligned_with_the_atlas_the_message_says_to_align_the_reference_with_it():
    channels = {'auto': {'align_with': 'fos'}, 'fos': {'align_with': 'auto'}}

    with pytest.raises(MissingRequirementException, match="No channel is aligned with the atlas: align 'fos' with it"):
        check_reference_selected_as_aligned_with_the_atlas({'channels': channels}, 'fos')
