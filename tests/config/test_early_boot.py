"""The machine config is created before anything else, from the package one"""
from ClearMap.config.config_handler import ConfigHandler
from ClearMap.config.early_boot import MachineConfig


def _package_machine_cfg_text() -> str:
    return ConfigHandler.get_default_path('machine', from_package=True).read_text()


def test_missing_machine_config_is_a_copy_of_the_package_one(tmp_path, monkeypatch):
    user_path = tmp_path / 'user' / 'machine.yml'
    monkeypatch.setenv('CLEARMAP_MACHINE_CFG', str(user_path))

    path = MachineConfig()._ensure_cfg_exists()

    assert path == user_path
    assert user_path.read_text() == _package_machine_cfg_text()


def test_existing_machine_config_is_kept(tmp_path, monkeypatch):
    user_path = tmp_path / 'machine.yml'
    user_path.write_text('verbosity: quiet\n')
    monkeypatch.setenv('CLEARMAP_MACHINE_CFG', str(user_path))

    MachineConfig()._ensure_cfg_exists()

    assert user_path.read_text() == 'verbosity: quiet\n'
