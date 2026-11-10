"""A file is newer than another when it was modified after it: the rule to compile again what is out of date."""
import os

import pytest

from ClearMap.Utils.path_utils import is_newer


@pytest.fixture()
def old_and_new(tmp_path):
    old, new = tmp_path / 'old.txt', tmp_path / 'new.txt'
    old.write_text('old')
    new.write_text('new')
    os.utime(old, (1_000_000_000, 1_000_000_000))
    os.utime(new, (1_000_000_100, 1_000_000_100))
    return old, new


def test_a_file_modified_later_is_newer(old_and_new):
    old, new = old_and_new

    assert is_newer(new, old)
    assert not is_newer(old, new)


def test_a_file_is_not_newer_than_itself_nor_than_one_modified_at_the_same_time(old_and_new, tmp_path):
    old, _ = old_and_new
    same = tmp_path / 'same.txt'
    same.write_text('same')
    os.utime(same, (1_000_000_000, 1_000_000_000))

    assert not is_newer(old, old) and not is_newer(same, old)


def test_paths_are_accepted_as_strings(old_and_new):
    old, new = old_and_new

    assert is_newer(str(new), str(old))
