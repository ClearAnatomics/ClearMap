"""
The reference channel is the channel aligned with the atlas, usually the autofluorescence channel (the default of the
data types). In #P0017 the reference channel was 'fos' (data type nuclei) while 'auto' was the autofluorescence channel.
A reference channel which does not agree with the align_with of the channels is refused before aligning; one which
agrees but is not the autofluorescence channel of the sample is allowed (e.g. dapi aligned with the atlas on purpose),
so the alignment warns.
"""
import warnings

import pytest

from ClearMap.Utils.exceptions import ReferenceNotAutofluorescenceWarning
from ClearMap.config.registration_channels import (NO_REFERENCE, REFERENCE_UNDECIDED,
                                                   warn_if_the_reference_is_not_the_autofluorescence_channel)
from ClearMap.pipeline_orchestrators.registration_orchestrator import RegistrationProcessor

DATA_TYPES_OF_P0017 = {'auto': 'autofluorescence', 'fos': 'nuclei'}


@pytest.mark.parametrize('reference_channel', ['auto', REFERENCE_UNDECIDED, NO_REFERENCE])
def test_the_autofluorescence_channel_or_no_reference_does_not_warn(reference_channel):
    with warnings.catch_warnings():
        warnings.simplefilter('error', ReferenceNotAutofluorescenceWarning)
        warn_if_the_reference_is_not_the_autofluorescence_channel(reference_channel, DATA_TYPES_OF_P0017)


def test_another_reference_without_an_autofluorescence_channel_does_not_warn():
    with warnings.catch_warnings():
        warnings.simplefilter('error', ReferenceNotAutofluorescenceWarning)
        warn_if_the_reference_is_not_the_autofluorescence_channel('dapi', {'dapi': 'nuclei', 'fos': 'nuclei'})


def test_a_reference_which_is_not_the_autofluorescence_channel_warns_naming_it():
    with pytest.warns(ReferenceNotAutofluorescenceWarning,
                      match="'fos' \\(data type 'nuclei'\\) .* choose 'auto' as the reference channel"):
        warn_if_the_reference_is_not_the_autofluorescence_channel('fos', DATA_TYPES_OF_P0017)


class FakeSampleManager:
    reference_channel_setting = 'fos'
    data_types_by_channel = DATA_TYPES_OF_P0017


class FakeRegistrationProcessor:
    """What align uses of a RegistrationProcessor: 'fos' aligned with the atlas, 'auto' with 'fos' (both swapped)"""
    align = RegistrationProcessor.align

    def __init__(self):
        self.stopped = False
        self.config = {'reference_channel': 'fos', 'channels': {'fos': {'align_with': 'atlas'},
                                                                'auto': {'align_with': 'fos'}}}
        self.channels_to_register = ['fos', 'auto']
        self.sample_manager = FakeSampleManager()
        self.aligned = []

    def align_channel(self, channel):
        self.aligned.append(channel)

    def update_watcher_main_progress(self):
        pass

    def publish(self, event):
        pass


def test_the_alignment_warns_about_such_a_reference_and_aligns_the_channels():
    processor = FakeRegistrationProcessor()

    with pytest.warns(ReferenceNotAutofluorescenceWarning):
        processor.align()

    assert processor.aligned == ['fos', 'auto']
