"""The cache folder of ClearMap is not the one of the user while the tests run (see conftest.py)."""
from pathlib import Path

from ClearMap import Settings as settings


def test_the_cache_folder_of_the_tests_is_not_in_the_home_of_the_user():
    home_cache = Path('~/.clearmap/cache').expanduser()

    assert settings.clearmap_cache_dir != home_cache
    assert home_cache not in settings.clearmap_cache_dir.parents
