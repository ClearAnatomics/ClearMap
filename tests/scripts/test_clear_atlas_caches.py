"""
The atlas caches that clearmap-utils clears are the atlases derived from the shipped ones (reoriented, cropped),
wherever ClearMap puts them: the atlas folder (former versions), the folder of the decompressed atlases, the atlas
cache folder. The atlases shipped (decompressed) are kept: they are not derived.
"""
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from ClearMap import Settings as settings
from ClearMap.Alignment.Annotation import get_atlas_filepath
from ClearMap.Scripts import clearmap_utils

DERIVED = 'ABA_25um_annotation_1_2_3_None.tif'
SHIPPED = 'ABA_25um_annotation.tif'
ADDED_ATLAS = ('/atlases/LAMBADA_P7_25um_reference.tif', (1, 2, 3), None)  # an atlas added to ClearMap
STEPS = (slice(10, 200, 2), slice(20, 300, 2), slice(5, 100, 2))  # no None in the name
WITH_STEPS = ('/atlases/ABA_25um_hemispheres.tif', (3, -2, 1), STEPS)
OTHER_DERIVED = [Path(get_atlas_filepath(path, orientation=orientation, slicing=slicing)).name
                 for path, orientation, slicing in (ADDED_ATLAS, WITH_STEPS)]
OTHER_FILES = ['LAMBADA_P7_25um_annotation.tif', 'ABA_annotation.json', 'ABA_25um_annotation.tif.lzma']


class FakeMachineConfig:
    atlas_cache_folder = None

    def get(self, key, default=None):
        assert key == 'atlas_cache_folder'
        return self.atlas_cache_folder


@pytest.fixture
def folders(tmp_path, monkeypatch):
    atlas_dir, cache_dir, atlas_cache_dir = tmp_path / 'Atlas', tmp_path / 'cache', tmp_path / 'atlas_cache'
    for folder in (atlas_dir, cache_dir / 'atlases', atlas_cache_dir):
        folder.mkdir(parents=True)
    monkeypatch.setattr(settings, 'atlas_folder', str(atlas_dir))
    monkeypatch.setattr(settings, 'clearmap_cache_dir', cache_dir)
    FakeMachineConfig.atlas_cache_folder = str(atlas_cache_dir)
    monkeypatch.setattr(clearmap_utils, 'MachineConfig', FakeMachineConfig)
    return atlas_dir, cache_dir / 'atlases', atlas_cache_dir


def test_the_derived_atlases_of_every_folder_are_removed_and_the_shipped_ones_kept(folders):
    kept = [SHIPPED, *OTHER_FILES]
    for folder in folders:
        for name in [DERIVED, *OTHER_DERIVED, *kept]:
            (folder / name).write_bytes(b'content')

    clearmap_utils.clear_atlas_caches()

    for folder in folders:
        assert sorted(path.name for path in folder.iterdir()) == sorted(kept)


def test_the_folders_of_the_atlases_which_do_not_exist_yet_are_skipped(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, 'atlas_folder', str(tmp_path / 'no_atlas_folder'))
    monkeypatch.setattr(settings, 'clearmap_cache_dir', tmp_path / 'no_cache')
    FakeMachineConfig.atlas_cache_folder = None
    monkeypatch.setattr(settings, 'clearmap_cfg_dir', tmp_path / 'no_cfg')
    monkeypatch.setattr(clearmap_utils, 'MachineConfig', FakeMachineConfig)

    clearmap_utils.clear_atlas_caches()


def test_the_verbose_mode_says_how_many_files_it_deletes(folders, monkeypatch, capsys):
    monkeypatch.setitem(sys.modules, 'tqdm', SimpleNamespace(tqdm=list))  # the progress bar is not under test
    for folder in folders:
        (folder / DERIVED).write_bytes(b'derived')

    clearmap_utils.clear_atlas_caches(verbose=True)

    assert 'Found 3 files, deleting' in capsys.readouterr().out
    assert not any((folder / DERIVED).exists() for folder in folders)
