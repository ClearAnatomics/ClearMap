"""
Which temporary folder a run uses: CLEARMAP_TMP (the folder of this run, e.g. the node-local one of a cluster job), else
the temp_folder of the machine config (shared by the runs), else the folder chosen in a dialog. Only the last one is
saved in the machine config; with none, the run stops instead of filling the system /tmp (ClearMap writes hundreds of GB).
"""
import os
import tempfile

import pytest

from ClearMap.config.early_boot import MachineConfig
from ClearMap.Utils.exceptions import ClearMapConfigError


@pytest.fixture
def machine_cfg_path(tmp_path, monkeypatch):
    path = tmp_path / 'machine.yml'
    monkeypatch.setenv('CLEARMAP_MACHINE_CFG', str(path))
    MachineConfig()._ensure_cfg_exists()
    monkeypatch.delenv('CLEARMAP_TMP', raising=False)
    # Restored by monkeypatch at the end: patch_tmp changes the process
    monkeypatch.setattr(tempfile, 'tempdir', tempfile.tempdir)
    for name in ('TMP', 'TEMP', 'TMPDIR'):
        monkeypatch.setenv(name, os.environ.get(name, '/tmp'))
    return path


@pytest.fixture
def machine_config(machine_cfg_path):
    config = MachineConfig()
    config.min_temp_folder_gb = 0  # The disk of the test machine is none of our business here
    return config


def patch_tmp(machine_config):
    return machine_config.patch_tmp(allow_prompt=False)


def test_the_environment_variable_comes_before_the_temp_folder_of_the_machine_config(
        machine_config, machine_cfg_path, tmp_path, monkeypatch):
    shared, node_local = tmp_path / 'shared', tmp_path / 'node_local'
    machine_cfg_path.write_text(f"temp_folder: '{shared}'\n")
    monkeypatch.setenv('CLEARMAP_TMP', str(node_local))

    assert patch_tmp(machine_config) == str(node_local)
    assert tempfile.gettempdir() == str(node_local)
    assert os.environ['TMPDIR'] == str(node_local)


def test_the_folder_of_the_environment_is_not_saved_in_the_machine_config(
        machine_config, machine_cfg_path, tmp_path, monkeypatch):
    before = machine_cfg_path.read_text()
    monkeypatch.setenv('CLEARMAP_TMP', str(tmp_path / 'node_local'))

    patch_tmp(machine_config)

    assert machine_cfg_path.read_text() == before


def test_the_temp_folder_of_the_machine_config_is_used_and_the_file_is_not_rewritten(
        machine_config, machine_cfg_path, tmp_path):
    shared = tmp_path / 'shared'
    machine_cfg_path.write_text(f"temp_folder: '{shared}'  # shared by the jobs\n")
    before = machine_cfg_path.read_text()

    assert patch_tmp(machine_config) == str(shared)
    assert machine_cfg_path.read_text() == before


def test_an_unusable_folder_of_the_environment_gives_way_to_the_machine_config(
        machine_config, machine_cfg_path, tmp_path, monkeypatch):
    shared = tmp_path / 'shared'
    machine_cfg_path.write_text(f"temp_folder: '{shared}'\n")
    (tmp_path / 'a_file').write_text('not a folder')
    monkeypatch.setenv('CLEARMAP_TMP', str(tmp_path / 'a_file' / 'tmp'))

    assert patch_tmp(machine_config) == str(shared)


def test_without_a_folder_headless_stops_and_does_not_fall_back_to_the_system_tmp(
        machine_config, machine_cfg_path):
    tempdir_before = tempfile.tempdir

    with pytest.raises(ClearMapConfigError, match='CLEARMAP_TMP'):
        patch_tmp(machine_config)
    assert tempfile.tempdir == tempdir_before


def test_the_error_names_the_unusable_folder_of_the_environment(machine_config, tmp_path, monkeypatch):
    (tmp_path / 'a_file').write_text('not a folder')
    unusable = str(tmp_path / 'a_file' / 'tmp')
    monkeypatch.setenv('CLEARMAP_TMP', unusable)

    with pytest.raises(ClearMapConfigError, match='CLEARMAP_TMP') as error_info:
        patch_tmp(machine_config)
    assert unusable in str(error_info.value)
