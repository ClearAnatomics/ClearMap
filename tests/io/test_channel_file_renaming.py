"""The files of a renamed channel follow it: planned from the workspace, left alone when not known"""
from pathlib import Path

import pytest

from ClearMap.IO.FileUtils import apply_renames
from ClearMap.IO.channel_file_renaming import plan_channel_file_renames
from ClearMap.IO.workspace2 import Workspace2
from ClearMap.IO.workspace_asset import Asset

SAMPLE_ID = 'S1'


def make_workspace(directory: Path, channels=('fos', 'nuc'), pairs=(), **kwargs) -> Workspace2:
    workspace = Workspace2(directory, sample_id=SAMPLE_ID, **kwargs)
    for channel in channels:
        for pipeline in ('stitching', 'CellMap', 'registration'):
            workspace.ensure_pipeline(pipeline, channel, sample_id=SAMPLE_ID, channel_content_type='cells',
                                      create_channel=True)
    for pair in pairs:
        workspace.ensure_pipeline('Colocalization', pair, sample_id=SAMPLE_ID, channel_content_type='compound',
                                  create_channel=True)
    return workspace


def touch(directory: Path, *names: str) -> None:
    for name in names:
        path = directory / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(name)


def names_in(directory: Path) -> list[str]:
    return sorted(str(path.relative_to(directory)) for path in directory.rglob('*') if path.is_file())


def targets(plan) -> dict[str, str]:
    return {rename.source.name: rename.target.name for rename in plan.renames}


@pytest.fixture()
def sample(tmp_path):
    return tmp_path


def test_the_files_of_a_renamed_channel_are_renamed_whatever_their_asset_and_extension(sample):
    workspace = make_workspace(sample)
    touch(sample, 'S1_fos_stitched.npy', 'S1_fos_cells.feather', 'S1_fos_cells_raw.feather', 'S1_fos_cells_shape.npy')

    plan = plan_channel_file_renames(workspace, {'fos': 'gfp'}, ())

    assert targets(plan) == {'S1_fos_stitched.npy': 'S1_gfp_stitched.npy',
                             'S1_fos_cells.feather': 'S1_gfp_cells.feather',
                             'S1_fos_cells_raw.feather': 'S1_gfp_cells_raw.feather',
                             'S1_fos_cells_shape.npy': 'S1_gfp_cells_shape.npy'}
    assert len(plan.renames) == 4  # A file belongs to one asset family only: cells_raw is not planned twice


def test_the_dynamic_sub_types_and_the_debug_files_follow_their_channel(sample):
    workspace = make_workspace(sample)
    touch(sample, 'S1_fos_density_counts_weighted_size.tif', 'debug_S1_fos_cells.feather')

    plan = plan_channel_file_renames(workspace, {'fos': 'gfp'}, ())

    assert targets(plan) == {'S1_fos_density_counts_weighted_size.tif': 'S1_gfp_density_counts_weighted_size.tif',
                             'debug_S1_fos_cells.feather': 'debug_S1_gfp_cells.feather'}


def test_the_files_of_the_other_channels_and_unknown_files_are_left_alone(sample):
    workspace = make_workspace(sample)
    touch(sample, 'S1_nuc_stitched.npy', 'S1_fos2_cells.feather', 'notes_fos.txt', 'S1_fos_unknown_thing.bin',
          'S1_fos_cellsX.feather', 'S1_fos_cells2.feather')  # (Not the asset: only its name starts alike)

    plan = plan_channel_file_renames(workspace, {'fos': 'gfp'}, ())

    assert plan.renames == []


def test_a_channel_whose_name_is_the_beginning_of_another_does_not_take_its_files(sample):
    workspace = make_workspace(sample, channels=('a', 'a_b'))
    touch(sample, 'S1_a_cells.feather', 'S1_a_b_cells.feather')

    plan = plan_channel_file_renames(workspace, {'a': 'z'}, ())

    assert targets(plan) == {'S1_a_cells.feather': 'S1_z_cells.feather'}


def test_the_files_of_a_pair_follow_the_renamed_part_in_place(sample):
    workspace = make_workspace(sample, pairs=[('fos', 'nuc')])
    touch(sample, 'S1_fos-nuc_colocalization.feather', 'S1_fos-nuc_colocalization_report.feather')

    plan = plan_channel_file_renames(workspace, {'nuc': 'dapi'}, ())

    assert targets(plan) == {'S1_fos-nuc_colocalization.feather': 'S1_fos-dapi_colocalization.feather',
                             'S1_fos-nuc_colocalization_report.feather': 'S1_fos-dapi_colocalization_report.feather'}


def test_both_parts_of_a_pair_can_be_renamed_at_once(sample):
    workspace = make_workspace(sample, pairs=[('fos', 'nuc')])
    touch(sample, 'S1_fos-nuc_colocalization.feather')

    plan = plan_channel_file_renames(workspace, {'fos': 'gfp', 'nuc': 'dapi'}, ())

    assert targets(plan) == {'S1_fos-nuc_colocalization.feather': 'S1_gfp-dapi_colocalization.feather'}


def test_two_channels_which_swap_their_names_swap_their_files(sample):
    workspace = make_workspace(sample)
    touch(sample, 'S1_fos_cells.feather', 'S1_nuc_cells.feather')

    apply_renames(plan_channel_file_renames(workspace, {'fos': 'nuc', 'nuc': 'fos'}, ()).renames)

    assert (sample / 'S1_nuc_cells.feather').read_text() == 'S1_fos_cells.feather'
    assert (sample / 'S1_fos_cells.feather').read_text() == 'S1_nuc_cells.feather'


def test_the_registration_folders_and_their_landmarks_follow_the_channels(sample):
    workspace = make_workspace(sample, channels=('auto', 'fos'))
    touch(sample, 'atlas_to_auto/auto_landmarks.pts', 'atlas_to_auto/atlas_landmarks.pts',
          'atlas_to_auto/result.0.mhd', 'auto_to_fos/fos_landmarks.pts', 'auto_to_fos/auto_landmarks.pts')

    plan = plan_channel_file_renames(workspace, {'auto': 'reference', 'fos': 'gfp'},
                                     [('atlas', 'auto'), ('auto', 'fos')])
    apply_renames(plan.renames)

    assert names_in(sample) == ['atlas_to_reference/atlas_landmarks.pts', 'atlas_to_reference/reference_landmarks.pts',
                                'atlas_to_reference/result.0.mhd', 'reference_to_gfp/gfp_landmarks.pts',
                                'reference_to_gfp/reference_landmarks.pts']


def test_a_registration_folder_of_unrenamed_channels_is_left_alone(sample):
    workspace = make_workspace(sample, channels=('auto', 'fos'))
    touch(sample, 'atlas_to_auto/auto_landmarks.pts')

    plan = plan_channel_file_renames(workspace, {'fos': 'gfp'}, [('atlas', 'auto')])

    assert plan.renames == []


def test_what_is_outside_the_folder_of_the_sample_is_reported_and_left_alone(sample, tmp_path_factory):
    elsewhere = tmp_path_factory.mktemp('elsewhere')
    workspace = make_workspace(sample, resource_type_to_folder={'results': str(elsewhere)})
    touch(elsewhere, 'S1_fos_cells.feather')
    touch(sample, 'S1_fos_stitched.npy')

    plan = plan_channel_file_renames(workspace, {'fos': 'gfp'}, ())

    assert targets(plan) == {'S1_fos_stitched.npy': 'S1_gfp_stitched.npy'}
    assert [(item.path.name, item.reason) for item in plan.not_renamed] == [
        ('S1_fos_cells.feather', 'outside the folder of the sample')]


def test_nothing_is_planned_when_no_name_changes(sample):
    workspace = make_workspace(sample)
    touch(sample, 'S1_fos_cells.feather')

    plan = plan_channel_file_renames(workspace, {'fos': 'fos'}, ())

    assert not plan


def test_raw_data_is_never_renamed_and_is_reported_once_whatever_its_number_of_tiles(sample):
    workspace = make_workspace(sample)
    raw = Asset(sample, workspace.asset_types['raw'], workspace['fos'].channel_spec,
                expression='fos_raw/Z<Z,4>.tif', sample_id=SAMPLE_ID)
    workspace.add_asset(raw)
    touch(sample, 'fos_raw/Z0000.tif', 'fos_raw/Z0001.tif', 'fos_raw/Z0002.tif')

    plan = plan_channel_file_renames(workspace, {'fos': 'gfp'}, ())

    assert plan.renames == []
    assert [(item.path, item.reason) for item in plan.not_renamed] == [(sample / 'fos_raw', 'raw data is never renamed')]


def test_a_registration_folder_which_is_not_a_pair_of_the_config_is_left_alone(sample):
    workspace = make_workspace(sample, channels=('auto', 'fos'))
    touch(sample, 'fos_to_auto/fos_landmarks.pts')  # E.g. from an earlier configuration: the config now says auto_to_fos

    plan = plan_channel_file_renames(workspace, {'fos': 'gfp'}, [('auto', 'fos')])

    assert plan.renames == []


def test_the_atlas_may_be_on_either_side_of_a_registration(sample):
    workspace = make_workspace(sample, channels=('auto',))
    touch(sample, 'auto_to_atlas/auto_landmarks.pts')

    plan = plan_channel_file_renames(workspace, {'auto': 'reference'}, [('auto', 'atlas')])
    apply_renames(plan.renames)

    assert names_in(sample) == ['reference_to_atlas/reference_landmarks.pts']
