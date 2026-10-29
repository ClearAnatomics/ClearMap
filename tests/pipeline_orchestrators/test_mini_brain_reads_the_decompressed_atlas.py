"""
The atlas is decompressed in the cache folder when the atlas folder holds only the archives, so the mini brain
reads the annotation where it is, not in the atlas folder.
"""
import lzma

import numpy as np

from ClearMap import Settings as settings
from ClearMap.pipeline_orchestrators import registration_orchestrator as registration

BASE_NAME = 'ABA_25um'


class FakeTifSource:
    """What the mini brain reads: an array, made from the path it was asked to read."""
    paths = []

    def __init__(self, path):
        self.paths.append(path)
        self.array = np.arange(4 * 4 * 4, dtype=float).reshape(4, 4, 4)


def test_the_mini_brain_is_made_from_the_annotation_decompressed_in_the_cache(tmp_path, monkeypatch):
    atlas_dir, cache_dir = tmp_path / 'program' / 'Atlas', tmp_path / 'user' / 'cache'
    atlas_dir.mkdir(parents=True)
    for component in ('annotation', 'hemispheres', 'reference', 'distance_to_surface'):
        (atlas_dir / f'{BASE_NAME}_{component}.tif.lzma').write_bytes(lzma.compress(b'atlas'))
    monkeypatch.setattr(settings, 'atlas_folder', str(atlas_dir))
    monkeypatch.setattr(settings, 'clearmap_cache_dir', cache_dir)
    monkeypatch.setattr(registration, 'TifSource', FakeTifSource)
    registration.setup_mini_brain.cache_clear()
    FakeTifSource.paths.clear()

    scaling, mini_brain = registration.setup_mini_brain(BASE_NAME, (2, 2, 2))

    assert FakeTifSource.paths == [str(cache_dir / 'atlases' / f'{BASE_NAME}_annotation.tif')]
    assert mini_brain.shape == (2, 2, 2)
    registration.setup_mini_brain.cache_clear()
