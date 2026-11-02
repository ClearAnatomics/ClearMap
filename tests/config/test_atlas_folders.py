"""
Where the atlases are kept: decompressed from the archives shipped with ClearMap (a cache folder, since the program
folder may be read-only), and reoriented / cropped (the machine config 'atlas_cache_folder', by default in the
ClearMap config folder).
"""
from pathlib import Path

from ClearMap import Settings as settings
from ClearMap.config import atlas


def test_the_atlases_decompressed_from_the_archives_are_in_the_cache_folder(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, 'clearmap_cache_dir', tmp_path / 'cache')

    assert atlas.get_decompressed_atlases_dir() == tmp_path / 'cache' / 'atlases'


def test_without_a_folder_in_the_machine_config_the_atlas_cache_is_in_the_clearmap_config_folder(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, 'clearmap_cfg_dir', tmp_path / 'clearmap_cfg')

    assert atlas.get_atlas_cache_dir(None) == tmp_path / 'clearmap_cfg' / 'atlas_cache'
    assert atlas.get_atlas_cache_dir('') == tmp_path / 'clearmap_cfg' / 'atlas_cache'


def test_a_folder_of_the_machine_config_is_used_with_the_home_expanded(tmp_path, monkeypatch):
    monkeypatch.setenv('HOME', str(tmp_path))
    monkeypatch.setenv('USERPROFILE', str(tmp_path))

    assert atlas.get_atlas_cache_dir('~/my_atlases') == tmp_path / 'my_atlases'
    assert atlas.get_atlas_cache_dir('/data/atlases') == Path('/data/atlases')
