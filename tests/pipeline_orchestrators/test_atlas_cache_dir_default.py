"""
The reoriented / cropped atlases are kept in the folder of the machine config 'atlas_cache_folder'. When it is empty
(the default) they are in the ClearMap config folder, wherever that is (CLEARMAP_CFG_DIR), not always in the home.
A folder set explicitly, including the former default, is used as it is.
"""
from pathlib import Path
from types import SimpleNamespace

from ClearMap import Settings as settings
from ClearMap.pipeline_orchestrators.registration_orchestrator import RegistrationProcessor


def atlas_cache_dir(atlas_cache_folder):
    orchestrator = SimpleNamespace(machine_config={'atlas_cache_folder': atlas_cache_folder})
    return RegistrationProcessor.atlas_cache_dir.fget(orchestrator)


def test_without_a_folder_the_atlases_are_cached_in_the_clearmap_config_folder(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, 'clearmap_cfg_dir', tmp_path / 'clearmap_cfg')

    assert atlas_cache_dir(None) == tmp_path / 'clearmap_cfg' / 'atlas_cache'
    assert atlas_cache_dir('') == tmp_path / 'clearmap_cfg' / 'atlas_cache'


def test_a_folder_set_in_the_machine_config_is_used_with_the_home_expanded(tmp_path, monkeypatch):
    monkeypatch.setenv('HOME', str(tmp_path))
    monkeypatch.setenv('USERPROFILE', str(tmp_path))

    assert atlas_cache_dir('~/my_atlases') == tmp_path / 'my_atlases'
    assert atlas_cache_dir('/data/atlases') == Path('/data/atlases')


def test_the_former_default_folder_of_an_existing_machine_config_is_still_the_same_one(tmp_path, monkeypatch):
    monkeypatch.setenv('HOME', str(tmp_path))
    monkeypatch.setenv('USERPROFILE', str(tmp_path))

    assert atlas_cache_dir('~/.clearmap/atlas_cache') == tmp_path / '.clearmap' / 'atlas_cache'
