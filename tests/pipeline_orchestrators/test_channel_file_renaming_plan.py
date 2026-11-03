"""
The files to rename with a channel, as the experiment plans them (headless, with the config adjusters of the
application): from its workspace and its registration config, under the old names.
"""
from pathlib import Path

from ClearMap.IO.FileUtils import apply_renames

OLD_NAME, NEW_NAME, REFERENCE_NAME = 'cfos', 'fos', 'auto'


def _rename(experiment, old: str, new: str) -> None:
    experiment.apply_ui_patch({'sample': {'$rename': {'channels': {old: new}}}})


def _write_cells_of(experiment, channel: str):
    path = experiment.sample_manager.workspace.get('cells', channel=channel, asset_sub_type='raw').path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()
    return path


def test_the_renamed_files_are_the_ones_the_pipeline_looks_for_under_the_new_name(experiment, channel_editor):
    channel_editor.add(OLD_NAME, 'nuclei', with_path=True)
    old_cells = _write_cells_of(experiment, OLD_NAME)

    plan = experiment.plan_channel_file_renames({OLD_NAME: NEW_NAME})
    _rename(experiment, OLD_NAME, NEW_NAME)
    apply_renames(plan.renames)

    new_cells = experiment.sample_manager.workspace.get('cells', channel=NEW_NAME, asset_sub_type='raw').path
    assert not old_cells.exists() and new_cells.exists()


def test_the_registration_folders_of_the_renamed_channel_are_renamed(experiment, channel_editor):
    channel_editor.add(REFERENCE_NAME, 'autofluorescence', with_path=True)
    channel_editor.add(OLD_NAME, 'nuclei', with_path=True)
    registration = experiment.get_config_view()['registration']['channels'][OLD_NAME]
    assert registration['align_with'] == REFERENCE_NAME  # The default: the reference moves onto the channel
    workspace = experiment.sample_manager.workspace
    elastix_folder = Path(workspace.directory) / workspace.resource_type_to_folder.get('elastix', '')
    registration_folder = elastix_folder / f'{REFERENCE_NAME}_to_{OLD_NAME}'
    registration_folder.mkdir(parents=True)

    plan = experiment.plan_channel_file_renames({OLD_NAME: NEW_NAME})

    assert [(rename.source, rename.target) for rename in plan.renames] == [
        (registration_folder, registration_folder.with_name(f'{REFERENCE_NAME}_to_{NEW_NAME}'))]


def test_nothing_is_planned_before_the_workspace_exists(experiment):
    experiment.sample_manager.workspace = None

    assert not experiment.plan_channel_file_renames({OLD_NAME: NEW_NAME})
