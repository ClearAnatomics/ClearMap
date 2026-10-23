"""
The early boot reads the machine config without a YAML library and writes the temp folder in it: the file stays the
one of the package (its comments too), with the temp folder, and is valid YAML for the full config loader.
A first boot used to rewrite every line from its own parse, which kept the inline comments in the values:
atlas_cache_folder: ""~/.clearmap/atlas_cache"  # ...", which the YAML loader refuses.
"""
import yaml

import pytest

from ClearMap.config.config_handler import ConfigHandler
from ClearMap.config.early_boot import MachineConfig
from ClearMap.config.utils import parse_flat_yaml, set_flat_yaml_value


@pytest.fixture()
def machine_cfg_path(tmp_path, monkeypatch):
    path = tmp_path / 'machine.yml'
    monkeypatch.setenv('CLEARMAP_MACHINE_CFG', str(path))
    MachineConfig()._ensure_cfg_exists()  # The copy of the package one
    return path


def test_the_package_machine_config_reads_as_with_a_yaml_loader():
    text = ConfigHandler.get_default_path('machine', from_package=True).read_text()

    assert parse_flat_yaml(text) == yaml.safe_load(text)


def test_setting_the_temp_folder_changes_only_that_line(machine_cfg_path):
    package_text = machine_cfg_path.read_text()

    MachineConfig()._patch_temp_dir_cfg('/data/tmp')

    lines_before, lines_after = package_text.splitlines(), machine_cfg_path.read_text().splitlines()
    changed = [(before, after) for before, after in zip(lines_before, lines_after) if before != after]
    assert len(lines_after) == len(lines_before)
    assert [after for _, after in changed] == ["temp_folder: '/data/tmp'"]
    assert yaml.safe_load(machine_cfg_path.read_text()) == {**yaml.safe_load(package_text), 'temp_folder': '/data/tmp'}


def test_the_temp_folder_is_set_again_at_each_boot_without_damage(machine_cfg_path):
    for _ in range(3):
        MachineConfig()._patch_temp_dir_cfg('/data/tmp')

    assert yaml.safe_load(machine_cfg_path.read_text())['temp_folder'] == '/data/tmp'
    assert MachineConfig().get('atlas_cache_folder') == '~/.clearmap/atlas_cache'
    assert MachineConfig().get('particle_plot_z_sphere_radius') == 3


@pytest.mark.parametrize('folder', [r'C:\Users\me\tmp', "/data/it's tmp", '/data/#tmp', ''])
def test_a_value_with_backslashes_quotes_or_hashes_or_empty_reads_back_the_same(tmp_path, folder):
    path = tmp_path / 'machine.yml'
    path.write_text('verbosity: "info"  # how much to print\ntemp_folder: null\n')

    set_flat_yaml_value(path, 'temp_folder', folder)

    assert yaml.safe_load(path.read_text())['temp_folder'] == folder
    assert parse_flat_yaml(path.read_text())['temp_folder'] == folder
    assert path.read_text().startswith('verbosity: "info"  # how much to print\n')


def test_a_missing_key_is_added(tmp_path):
    path = tmp_path / 'machine.yml'
    path.write_text('verbosity: "info"\n')

    set_flat_yaml_value(path, 'temp_folder', '/data/tmp')

    assert yaml.safe_load(path.read_text()) == {'verbosity': 'info', 'temp_folder': '/data/tmp'}
