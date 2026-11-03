"""
ClearMap derives files from the package (decompressed atlases, look-up tables) in a cache folder, by default
``~/.clearmap/cache``. The tests must neither read nor write the home of whoever runs them, so the cache folder
is a folder of the pytest cache of the repository: the derived files (about 1 GB of atlases) are built once there,
and ``pytest --cache-clear`` removes them.
"""
import tempfile
from pathlib import Path

import pytest

from ClearMap import Settings as settings

CACHE_FOLDER_NAME = 'clearmap_cache'  # In the pytest cache (.pytest_cache/d/)

_cache_dir_before_the_tests = None


# Not at the start of a session: the pytest cache (config.cache) must exist, and this is before the collection,
# which imports modules that read the default atlas.
@pytest.hookimpl(trylast=True)
def pytest_configure(config):
    global _cache_dir_before_the_tests
    _cache_dir_before_the_tests = settings.clearmap_cache_dir
    if getattr(config, 'cache', None) is not None:
        settings.clearmap_cache_dir = Path(config.cache.mkdir(CACHE_FOLDER_NAME))
    else:  # -p no:cacheprovider
        settings.clearmap_cache_dir = Path(tempfile.mkdtemp(prefix='clearmap_test_cache_'))


def pytest_unconfigure(config):
    settings.clearmap_cache_dir = _cache_dir_before_the_tests
