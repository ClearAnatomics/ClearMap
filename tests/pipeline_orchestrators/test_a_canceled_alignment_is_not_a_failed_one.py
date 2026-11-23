"""
The Abort button terminates elastix (stop_process), which then exits with an error: the alignment of the channel is
canceled, not failed. A failed elastix registration raises ElastixFailedError (#P0018); once stopped, the alignment
raises CanceledProcessing instead, which align() expects, and the channels after it are not aligned.
"""
import re

import pytest

from ClearMap.Utils.exceptions import ElastixFailedError
from ClearMap.pipeline_orchestrators import registration_orchestrator
from ClearMap.pipeline_orchestrators.generic_orchestrators import CanceledProcessing
from ClearMap.pipeline_orchestrators.registration_orchestrator import RegistrationProcessor

CHANNEL = 'auto'
PARAMS_FILES = ['align_affine.txt', 'align_bspline.txt']


class FakeAsset:
    def __init__(self, path):
        self.path = path
        self.existing_path = path


class FakeRegistrationProcessor:
    """What align_channel uses of a RegistrationProcessor, without landmarks"""
    align_channel = RegistrationProcessor.align_channel
    _RegistrationProcessor__bspline_registration_re = re.compile('bspline')
    _RegistrationProcessor__affine_registration_re = re.compile('affine')

    def __init__(self, folder, *, stopped):
        self.folder = folder
        self.stopped = stopped
        self.config = {'channels': {CHANNEL: {'params_files': PARAMS_FILES, 'landmarks_weights': [0, 0]}}}
        self.registration_params_files = {CHANNEL: [folder / name for name in PARAMS_FILES]}
        self.workspace = 'workspace'

    def get_fixed_moving_channels(self, channel):
        return 'fos', channel

    def prepare_watcher_for_substep(self, *args, **kwargs):
        pass

    def get_moving_image(self, channel):
        return FakeAsset(self.folder / 'auto_resampled.tif')

    def get_fixed_image(self, channel):
        return FakeAsset(self.folder / 'fos_resampled.tif')

    def get_elx_asset(self, asset_type, channel):
        return FakeAsset(self.folder / 'auto_to_fos' / 'result.1.mhd')


def is_registered(channel_cfg):
    return True


def terminated_elastix(align_parameters, landmarks_files, landmarks_weights=None):
    raise ElastixFailedError('Elastix exited with code -15, check its log: "auto_to_fos/elastix.log"')


@pytest.fixture()
def elastix_exits_with_an_error(monkeypatch):
    monkeypatch.setattr(registration_orchestrator, 'needs_registration', is_registered)
    monkeypatch.setattr(registration_orchestrator.elastix, 'align_from_dict', terminated_elastix)


def test_an_elastix_error_after_the_abort_button_cancels_the_alignment(tmp_path, elastix_exits_with_an_error):
    processor = FakeRegistrationProcessor(tmp_path, stopped=True)

    with pytest.raises(CanceledProcessing):
        processor.align_channel(CHANNEL)


def test_an_elastix_error_without_the_abort_button_is_a_failed_alignment(tmp_path, elastix_exits_with_an_error):
    processor = FakeRegistrationProcessor(tmp_path, stopped=False)

    with pytest.raises(ElastixFailedError, match='exited with code -15'):
        processor.align_channel(CHANNEL)
