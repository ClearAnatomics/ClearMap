import pytest
from packaging.version import Version

from ClearMap.Utils.exceptions import ClearMapValueError
from ClearMap.Utils.versions import as_version, compare_versions


@pytest.mark.parametrize('written', ['3.1', '3.1.0', (3, 1), (3, 1, 0), Version('3.1'), 3.1])
def test_every_way_to_write_a_version_gives_the_same_version(written):
    assert as_version(written) == Version('3.1')


@pytest.mark.parametrize('not_a_version', [None, '', 'latest', ('a', 1)])
def test_what_is_not_a_version_is_refused_with_what_was_given(not_a_version):
    with pytest.raises(ClearMapValueError, match=repr(not_a_version).replace('(', r'\(').replace(')', r'\)')):
        as_version(not_a_version)


@pytest.mark.parametrize('left, right, expected', [
    ('3.9', '3.10', -1),  # As strings, "3.9" > "3.10"
    ('3.10', '3.9', 1),
    ('3.1', '3.1.0', 0),
    ((3, 1), '3.1.0', 0),
    ((3, 1), (3, 1, 0), 0),  # As tuples, (3, 1) < (3, 1, 0)
    ('3.2.0.dev0', '3.2.0', -1),
    ('2.1', (3, 0), -1),
    (Version('3.1.1'), '3.1', 1),
])
def test_versions_are_compared_as_versions(left, right, expected):
    assert compare_versions(left, right) == expected


@pytest.mark.parametrize('left, right, expected', [
    ('3.1.1', (3, 1), 0),  # A patch release of the same major.minor
    ('3.1.0', '3.1', 0),
    ('3.2.0.dev0', (3, 2), 0),  # A pre-release of the same major.minor
    ('3', (3, 0), 0),  # A missing number is 0
    ('3.0.9', (3, 1), -1),
    ('3.2.0.dev0', (3, 1), 1),
])
def test_comparing_major_and_minor_ignores_the_patch_and_the_pre_release(left, right, expected):
    assert compare_versions(left, right, parts=2) == expected


def test_comparing_only_the_major_version():
    assert compare_versions('3.9', '3.1', parts=1) == 0
    assert compare_versions('2.9', '3.1', parts=1) == -1
