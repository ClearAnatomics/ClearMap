"""
The upgrade of a 2.1 experiment renames its files and its elastix folders to the names of ClearMap 3. The renaming
step is run here on a folder named as 2.1 named it, with the channels already named (the user was asked) and the
verification of the renames accepted as it is, which is the only part that shows a dialog.
"""
from pathlib import Path

import pytest

from ClearMap.config.convert_config_versions import ExperimentUpgrade_2_1_to_3_0

SAMPLE_ID = 'S1'
OLD_FILES = ('stitched.npy', 'resampled.tif', 'resampled_autofluorescence.tif')
OLD_ELASTIX_FOLDERS = ('elastix_auto_to_reference', 'elastix_resampled_to_auto')
NEW_CHANNELS = {'raw': ('nuclei', 'nuclei'), 'autofluorescence': ('autofluorescence', 'autofluorescence')}


def upgrader_of_a_2_1_experiment(folder: Path, *, with_sample_id_prefix: bool, aligned: bool):
    folder.mkdir()
    prefix = f'{SAMPLE_ID}_' if with_sample_id_prefix else ''
    for old_file in OLD_FILES:
        (folder / f'{prefix}{old_file}').write_text('')
    if aligned:
        for old_folder in OLD_ELASTIX_FOLDERS:
            (folder / f'{prefix}{old_folder}').mkdir()
            (folder / f'{prefix}{old_folder}' / 'result.0.tif').write_text('')

    upgrader = ExperimentUpgrade_2_1_to_3_0(folder, app=None)
    upgrader.use_id_as_prefix = with_sample_id_prefix
    upgrader.sample_id = SAMPLE_ID
    upgrader.new_channels = NEW_CHANNELS
    upgrader.sample_cfg = {'channels': {new_name: {'data_type': data_type}
                                        for new_name, data_type in NEW_CHANNELS.values()}}
    upgrader.verify_renaming = list  # Accept all the renames, no dialog
    return upgrader


def test_the_files_of_a_sample_which_was_not_aligned_are_renamed(tmp_path):
    upgrader = upgrader_of_a_2_1_experiment(tmp_path / 'experiment', with_sample_id_prefix=False, aligned=False)

    upgrader._rename_assets()

    assert sorted(path.name for path in upgrader.root.iterdir()) == ['autofluorescence_resampled.tif',
                                                                     'nuclei_resampled.tif', 'nuclei_stitched.npy']


@pytest.mark.parametrize('with_sample_id_prefix', [False, True])
def test_the_elastix_folders_are_named_by_their_channels_whatever_the_sample_id_prefix(tmp_path,
                                                                                         with_sample_id_prefix):
    upgrader = upgrader_of_a_2_1_experiment(tmp_path / 'experiment', with_sample_id_prefix=with_sample_id_prefix,
                                            aligned=True)

    upgrader._rename_assets()

    # Never prefixed by the sample id: see the asset 'aligned'
    assert sorted(path.name for path in upgrader.root.iterdir() if path.is_dir()) == ['atlas_to_autofluorescence',
                                                                                     'autofluorescence_to_nuclei']


@pytest.mark.parametrize('with_sample_id_prefix', [False, True])
def test_the_files_are_renamed_by_channel_and_keep_one_sample_id_prefix(tmp_path, with_sample_id_prefix):
    upgrader = upgrader_of_a_2_1_experiment(tmp_path / 'experiment', with_sample_id_prefix=with_sample_id_prefix,
                                            aligned=True)
    prefix = f'{SAMPLE_ID}_' if with_sample_id_prefix else ''

    upgrader._rename_assets()

    assert sorted(path.name for path in upgrader.root.iterdir() if path.is_file()) == [
        f'{prefix}autofluorescence_resampled.tif', f'{prefix}nuclei_resampled.tif', f'{prefix}nuclei_stitched.npy']
