"""
The config holds the landmarks weights in percent, as the sliders of the GUI show them: the registration gives
elastix the weights of these percents (elastix_landmarks_weight), growing geometrically from 1 to 100 percent.

A registration with landmarks weights needs the landmarks (to align without them, set the weights to 0). Without
their files, elastix aligned without them, silently; with weights for a subset of the steps, the message was lost
(raise NotImplemented: TypeError).
"""
import re

import pytest

from ClearMap.Utils.exceptions import AssetNotFoundError, ClearMapNotImplementedError
from ClearMap.pipeline_orchestrators import registration_orchestrator
from ClearMap.pipeline_orchestrators.registration_orchestrator import (LANDMARKS_WEIGHT_MAX, LANDMARKS_WEIGHT_MIN,
                                                                     RegistrationProcessor, elastix_landmarks_weight)

CHANNEL = 'autofluorescence'
PARAMS_FILES = ['align_affine.txt', 'align_bspline.txt']


class FakeAsset:
    def __init__(self, path):
        self.path = path
        self.existing_path = path


class FakeRegistrationProcessor:
    align_channel = RegistrationProcessor.align_channel
    _RegistrationProcessor__bspline_registration_re = re.compile('bspline')
    _RegistrationProcessor__affine_registration_re = re.compile('affine')

    def __init__(self, folder, landmarks_weights):
        self.folder = folder
        self.config = {'channels': {CHANNEL: {'params_files': PARAMS_FILES, 'landmarks_weights': landmarks_weights}}}
        self.registration_params_files = {CHANNEL: [folder / name for name in PARAMS_FILES]}
        self.workspace = 'workspace'

    def get_fixed_moving_channels(self, channel):
        return 'reference', channel

    def prepare_watcher_for_substep(self, *args, **kwargs):
        pass

    def get_moving_image(self, channel):
        return FakeAsset(self.folder / 'moving.tif')

    def get_fixed_image(self, channel):
        return FakeAsset(self.folder / 'fixed.tif')

    def get_elx_asset(self, asset_type, channel):
        return FakeAsset(self.folder / 'aligned' / f'{asset_type}.txt')


def is_registered(channel_cfg):
    return True


@pytest.fixture()
def alignments(monkeypatch):
    """The landmarks files and weights of each alignment run"""
    runs = []

    def align_from_dict(align_parameters, landmarks_files, landmarks_weights=None):
        runs.append({'landmarks_files': landmarks_files, 'landmarks_weights': landmarks_weights})

    monkeypatch.setattr(registration_orchestrator, 'needs_registration', is_registered)
    monkeypatch.setattr(registration_orchestrator.elastix, 'align_from_dict', align_from_dict)
    return runs



def test_elastix_is_given_the_weights_of_the_percents_of_the_config(tmp_path, alignments):
    processor = FakeRegistrationProcessor(tmp_path, landmarks_weights=[50, 50])
    select_landmarks(tmp_path)

    processor.align_channel(CHANNEL)

    assert alignments[0]['landmarks_weights'] == [elastix_landmarks_weight(50)] * 2
    assert 0 < elastix_landmarks_weight(50) < LANDMARKS_WEIGHT_MAX


def test_the_weight_of_elastix_grows_geometrically_from_none_at_0_to_the_maximum_at_100_percent():
    assert elastix_landmarks_weight(0) == 0
    assert elastix_landmarks_weight(100) == pytest.approx(LANDMARKS_WEIGHT_MAX)
    weights = {percent: elastix_landmarks_weight(percent) for percent in (25, 50, 75)}
    assert weights[50] ** 2 == pytest.approx(weights[25] * weights[75])  # The same ratio for the same step
    assert elastix_landmarks_weight(1) > LANDMARKS_WEIGHT_MIN


def select_landmarks(folder):
    (folder / 'aligned').mkdir()
    for asset_type in ('fixed_landmarks', 'moving_landmarks'):
        (folder / 'aligned' / f'{asset_type}.txt').write_text('point\n0\n')


def test_landmarks_weights_without_landmarks_are_reported_before_aligning(tmp_path, alignments):
    processor = FakeRegistrationProcessor(tmp_path, landmarks_weights=[50, 50])

    with pytest.raises(AssetNotFoundError, match='landmarks are not selected'):
        processor.align_channel(CHANNEL)
    assert alignments == []


def test_selected_landmarks_are_given_to_the_alignment(tmp_path, alignments):
    select_landmarks(tmp_path)
    processor = FakeRegistrationProcessor(tmp_path, landmarks_weights=[50, 50])

    processor.align_channel(CHANNEL)

    assert alignments[0]['landmarks_files']['fixed_landmarks_path'] == tmp_path / 'aligned' / 'fixed_landmarks.txt'


def test_landmarks_weights_for_a_subset_of_the_steps_are_reported(tmp_path, alignments):
    select_landmarks(tmp_path)
    processor = FakeRegistrationProcessor(tmp_path, landmarks_weights=[50, 0])

    with pytest.raises(ClearMapNotImplementedError, match='subset of the registration steps'):
        processor.align_channel(CHANNEL)


def test_without_landmarks_weights_the_alignment_runs_without_landmarks(tmp_path, alignments):
    processor = FakeRegistrationProcessor(tmp_path, landmarks_weights=[0, 0])

    processor.align_channel(CHANNEL)

    assert alignments[0]['landmarks_files'] == {'moving_landmarks_path': '', 'fixed_landmarks_path': ''}
