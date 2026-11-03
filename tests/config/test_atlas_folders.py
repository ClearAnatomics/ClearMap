"""
Where the atlases are kept: decompressed from the archives shipped with ClearMap (a cache folder, since the program
folder may be read-only), and reoriented / cropped (the machine config 'atlas_cache_folder', by default in the
ClearMap config folder).
"""
from pathlib import Path

import pytest

from ClearMap import Settings as settings
from ClearMap.Alignment.Annotation import get_atlas_filepath
from ClearMap.config import atlas

FULL_CROP = (slice(None),) * 3
CROP = (slice(10, 200), slice(20, 300), slice(5, 100))
CROP_WITH_STEPS = (slice(10, 200, 2), slice(20, 300, 2), slice(5, 100, 2))  # no None in the name


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


@pytest.mark.parametrize('base_name', ['ABA_25um', 'ABA_25um_2017', 'LAMBADA_P7_25um'])
@pytest.mark.parametrize('component', atlas.ATLAS_COMPONENT_NAMES)
@pytest.mark.parametrize('orientation', [(1, 2, 3), (-1, 2, 3), (3, -2, 1)])
@pytest.mark.parametrize('slicing', [None, FULL_CROP, CROP, CROP_WITH_STEPS])
def test_the_name_of_an_atlas_derived_by_clearmap_is_recognised(base_name, component, orientation, slicing):
    name = Path(get_atlas_filepath(f'/atlases/{base_name}_{component}.tif', orientation=orientation,
                                   slicing=slicing)).name

    assert atlas.DERIVED_ATLAS_NAME_RE.match(name), name


@pytest.mark.parametrize('name', ['ABA_25um_annotation.tif', 'LAMBADA_P7_25um_distance_to_surface.tif',
                                  'ABA_25um_annotation.tif.lzma', 'ABA_annotation.json', 'regions_lambada.csv',
                                  'my_annotation_v2.tif', 'annotation_1_2_3_None.tif'])
def test_the_name_of_a_shipped_atlas_or_of_another_file_is_not_taken_for_a_derived_atlas(name):
    assert not atlas.DERIVED_ATLAS_NAME_RE.match(name)
