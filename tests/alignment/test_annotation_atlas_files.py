"""
The atlases ship compressed (1 GB once decompressed). They are decompressed when first used, in the cache folder
when the atlas folder holds only the archives, never when the module is imported.
"""
import importlib.util
import lzma

import pytest

from ClearMap import Settings as settings
from ClearMap.Alignment import Annotation
from ClearMap.IO import FileUtils as fu
from ClearMap.Utils.exceptions import AssetNotFoundError

BASE_NAME = 'ABA_25um'
COMPONENTS = ('annotation', 'hemispheres', 'reference', 'distance_to_surface')


def never_called(*args, **kwargs):
    raise AssertionError('an atlas was decompressed while importing the module')


@pytest.fixture
def folders(tmp_path, monkeypatch):
    atlas_dir, cache_dir = tmp_path / 'program' / 'Atlas', tmp_path / 'user' / 'cache'
    atlas_dir.mkdir(parents=True)
    monkeypatch.setattr(settings, 'atlas_folder', str(atlas_dir))
    monkeypatch.setattr(settings, 'clearmap_cache_dir', cache_dir)
    Annotation.get_default_atlas_files.cache_clear()
    yield atlas_dir, cache_dir
    Annotation.get_default_atlas_files.cache_clear()


def ship_archives(atlas_dir):
    for component in COMPONENTS:
        archive = atlas_dir / f'{BASE_NAME}_{component}.tif.lzma'
        archive.write_bytes(lzma.compress(f'content of {component}'.encode()))


def test_importing_the_module_decompresses_nothing(monkeypatch):
    monkeypatch.setattr(fu, 'uncompress', never_called)
    monkeypatch.setattr(fu, 'decompress_file', never_called)
    monkeypatch.setattr(fu, 'find_or_build', never_called)
    # a copy under another name: the module the other tests use is left alone
    spec = importlib.util.spec_from_file_location('annotation_imported_alone', Annotation.__file__)
    module = importlib.util.module_from_spec(spec)

    spec.loader.exec_module(module)


def test_the_archives_are_decompressed_in_the_cache_and_the_atlas_folder_is_not_written(folders):
    atlas_dir, cache_dir = folders
    ship_archives(atlas_dir)
    names_before = sorted(path.name for path in atlas_dir.iterdir())

    paths = Annotation.decompress_atlases(BASE_NAME)

    assert list(paths) == list(COMPONENTS)
    for component, path in paths.items():
        assert path == str(cache_dir / Annotation.ATLAS_CACHE_SUBFOLDER_NAME / f'{BASE_NAME}_{component}.tif')
        assert open(path, 'rb').read() == f'content of {component}'.encode()
    assert sorted(path.name for path in atlas_dir.iterdir()) == names_before


def test_an_atlas_already_decompressed_in_the_atlas_folder_is_used_where_it_is(folders):
    atlas_dir, cache_dir = folders
    for component in COMPONENTS:
        (atlas_dir / f'{BASE_NAME}_{component}.tif').write_bytes(b'prepared')

    paths = Annotation.decompress_atlases(BASE_NAME)

    assert paths['annotation'] == str(atlas_dir / f'{BASE_NAME}_annotation.tif')
    assert not cache_dir.exists()


def test_an_atlas_without_archive_nor_file_is_reported_with_its_name_and_folder(folders):
    atlas_dir, _ = folders

    with pytest.raises(AssetNotFoundError, match=f'{BASE_NAME}_annotation.tif.*{atlas_dir.name}'):
        Annotation.decompress_atlases(BASE_NAME)


def test_the_default_atlas_files_are_still_module_attributes_and_decompressed_once(folders):
    atlas_dir, _ = folders
    ship_archives(atlas_dir)

    assert Annotation.get_default_atlas_files() is Annotation.get_default_atlas_files()
    assert Annotation.default_annotation_file == Annotation.get_default_atlas_files()['annotation']
    assert Annotation.default_hemispheres_file == Annotation.get_default_atlas_files()['hemispheres']
    assert Annotation.default_reference_file.endswith(f'{BASE_NAME}_reference.tif')
    assert Annotation.default_distance_to_surface_file.endswith(f'{BASE_NAME}_distance_to_surface.tif')


def test_an_unknown_module_attribute_is_still_an_attribute_error():
    with pytest.raises(AttributeError, match='no_such_atlas_file'):
        Annotation.no_such_atlas_file
