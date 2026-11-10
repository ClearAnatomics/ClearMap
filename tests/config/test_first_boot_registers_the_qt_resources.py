"""
The GUI boot registers the Qt resources (icons and images of the help) with Qt, after the temporary folder is set, with
the compiler of the loaded Qt: it no longer runs ``pyrcc5``, which PyQt6 has not (the boot failed with a
FileNotFoundError on a PyQt6-only installation). The other steps of the boot are faked: they have their own tests.
"""
from pathlib import Path

from ClearMap.config import early_boot, update_config
from ClearMap.gui import qt_resources

BOOT_STEPS = []


class FakeMachineConfig:
    _version = '3.2.0'

    def patch_tmp(self, allow_prompt):
        BOOT_STEPS.append('tmp folder patched')
        return Path('/tmp')


def _register_resources():
    BOOT_STEPS.append('resources registered')


def _is_not_the_first_run(version_tag):
    return False


def _update_default_config():
    BOOT_STEPS.append('config updated')


def test_the_boot_registers_the_resources_once_after_the_temporary_folder_is_set(monkeypatch):
    BOOT_STEPS.clear()
    monkeypatch.setattr(early_boot, 'MachineConfig', FakeMachineConfig)
    monkeypatch.setattr(early_boot, '_is_first_run', _is_not_the_first_run)
    monkeypatch.setattr(update_config, 'update_default_config', _update_default_config)
    monkeypatch.setattr(qt_resources, 'register_resources', _register_resources)

    early_boot.first_boot(allow_prompt=False)

    assert BOOT_STEPS == ['tmp folder patched', 'resources registered', 'config updated']
