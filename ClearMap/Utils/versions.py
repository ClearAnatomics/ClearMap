"""
Versions are compared as versions, never as strings ("3.10" is after "3.9", "3.1" is "3.1.0") or as tuples of
different lengths ((3, 1) is not (3, 1, 0)). Every comparison of versions in ClearMap goes through here.
"""
from typing import Union

from packaging.version import Version, InvalidVersion

from ClearMap.Utils.exceptions import ClearMapValueError

VersionLike = Union[str, tuple, Version]


def as_version(version: VersionLike) -> Version:
    """
    The version, whatever way it is written.

    Parameters
    ----------
    version : str | tuple | Version
        "3.1", "3.2.0.dev0", (3, 1) or a Version.

    Raises
    ------
    ClearMapValueError
        If it is not a version (None, "", "latest").
    """
    if isinstance(version, Version):
        return version
    text = '.'.join(str(number) for number in version) if isinstance(version, tuple) else str(version)
    try:
        return Version(text)
    except InvalidVersion as err:
        raise ClearMapValueError(f'Expected a version like "3.1", (3, 1) or "3.2.0.dev0", got {version!r}') from err


def compare_versions(left: VersionLike, right: VersionLike, *, parts: int | None = None) -> int:
    """
    Whether ``left`` is before (-1), the same as (0) or after (1) ``right``.

    Parameters
    ----------
    left, right : str | tuple | Version
        The versions to compare, each written in any of the forms that ``as_version`` takes.
    parts : int | None
        Compare only the first ``parts`` numbers (2 for major.minor): the patch number and the pre-release tag are
        ignored, and a missing number is 0. By default, the versions are compared whole ("3.2.0.dev0" is before
        "3.2.0").
    """
    left, right = as_version(left), as_version(right)
    if parts is not None:
        left, right = (_first_numbers(version, parts) for version in (left, right))
    return (left > right) - (left < right)


def _first_numbers(version: Version, parts: int) -> tuple:
    return (version.release + (0,) * parts)[:parts]
