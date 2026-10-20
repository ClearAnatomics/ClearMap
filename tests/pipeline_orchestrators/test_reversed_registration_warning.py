"""
The points of a channel are transformed through the registrations of its align_with chain, as computed: a
registration computed the other way round (the channel moving onto its partner) is used, with a warning
(scripts may want it), as the points would need its inverse.
"""
import functools
import warnings
from types import SimpleNamespace

import pytest

from ClearMap.Utils.exceptions import ReversedRegistrationWarning
from ClearMap.config.registration_channels import (ATLAS, fixed_moving_channels, registration_sequence,
                                                   reversed_registrations)
from ClearMap.pipeline_orchestrators.registration_orchestrator import RegistrationProcessor


def config(**channels):
    return {'channels': {name: {'align_with': align_with, 'moving_channel': moving}
                         for name, (align_with, moving) in channels.items()}}


def _registration_processor(registration_config, results_folder):
    """The parts of the processor get_transform_directories reads: the config, and the results of each channel"""
    def get(asset_type, channel):
        fixed, moving = fixed_moving_channels(registration_config, channel)
        result_dir = results_folder / f'{moving}_to_{fixed}'
        result_dir.mkdir(exist_ok=True)
        return SimpleNamespace(path=result_dir / 'result.0.mhd')
    return SimpleNamespace(registration_config=registration_config, get=get,
                           get_registration_sequence_channels=functools.partial(registration_sequence,
                                                                                registration_config))


def test_the_channels_registered_the_other_way_round_are_the_ones_which_move():
    registration_config = config(auto=(ATLAS, 'other'), fos=('auto', 'self'), gfp=('fos', 'gfp'), dapi=('fos', 'fos'),
                                 nuc=('auto', 'intrinsically_aligned'))

    assert reversed_registrations(registration_config, ['auto', 'fos', 'gfp', 'dapi', 'nuc']) == ['fos', 'gfp']


def test_transforming_through_a_registration_computed_the_other_way_round_warns_and_names_it(tmp_path):
    processor = _registration_processor(config(auto=(ATLAS, 'other'), fos=('auto', 'self')), tmp_path)

    with pytest.warns(ReversedRegistrationWarning, match='"fos" moving onto "auto"'):
        directories = RegistrationProcessor.get_transform_directories(processor, 'fos')

    assert directories == [tmp_path / 'fos_to_auto', tmp_path / 'atlas_to_auto']  # Still given: not refused


def test_transforming_through_registrations_of_the_partners_moving_onto_the_channels_does_not_warn(tmp_path):
    processor = _registration_processor(config(auto=(ATLAS, ATLAS), fos=('auto', 'other')), tmp_path)

    with warnings.catch_warnings():
        warnings.simplefilter('error', ReversedRegistrationWarning)
        directories = RegistrationProcessor.get_transform_directories(processor, 'fos')

    assert directories == [tmp_path / 'auto_to_fos', tmp_path / 'atlas_to_auto']
