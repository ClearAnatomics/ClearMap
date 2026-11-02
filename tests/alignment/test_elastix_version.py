"""
The elastix version is read from ``elastix --version`` for the stock releases ("5.3.1", three numbers) as well as for
the patched build shipped with ClearMap ("4.900"). Reading it as a float crashed on "5.3.1", and with it the binary
point files of ``transform_points``.
"""
import numpy as np
import pytest
from packaging.version import Version

from ClearMap.Alignment import Elastix as elastix
from ClearMap.Utils.exceptions import ClearMapRuntimeError

STOCK_ELASTIX_VERSION_OUTPUT = 'elastix version: 5.3.1\n'
PATCHED_ELASTIX_VERSION_OUTPUT = 'elastix version: 4.900\n'


@pytest.fixture()
def fake_elastix(tmp_path, monkeypatch):
    """Configure ClearMap to use an elastix executable that prints the given ``--version`` output"""
    def configure(version_output):
        executable = tmp_path / 'elastix'
        executable.write_text(f'#!/bin/sh\nprintf %s "{version_output}"\n')
        executable.chmod(0o755)
        monkeypatch.setattr(elastix, 'elastix_binary', str(executable))
        monkeypatch.setattr(elastix, 'initialized', True)
    return configure


@pytest.mark.parametrize('version_output, expected_version', [
    (STOCK_ELASTIX_VERSION_OUTPUT, Version('5.3.1')),
    (PATCHED_ELASTIX_VERSION_OUTPUT, Version('4.900')),
    ('elastix version: 5.2\n', Version('5.2')),
])
def test_the_elastix_version_is_read_as_a_version(fake_elastix, version_output, expected_version):
    fake_elastix(version_output)

    assert elastix.get_elastix_version() == expected_version


@pytest.mark.parametrize('version_output', ['something else\n', ''])
def test_an_unreadable_version_names_the_executable_and_its_output(fake_elastix, version_output):
    fake_elastix(version_output)

    with pytest.raises(ClearMapRuntimeError, match='expected a line like'):
        elastix.get_elastix_version()


@pytest.mark.parametrize('version_output', [STOCK_ELASTIX_VERSION_OUTPUT, PATCHED_ELASTIX_VERSION_OUTPUT])
def test_binary_point_files_can_be_written_whatever_the_elastix_version(fake_elastix, tmp_path, version_output):
    fake_elastix(version_output)
    point_file = tmp_path / 'points.bin'
    points = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

    elastix.write_points(str(point_file), points, binary=True)

    header = np.frombuffer(point_file.read_bytes()[:16], dtype=np.int64)
    assert header.tolist() == [0, len(points)]  # the index flag (0: physical coordinates), then the number of points
    np.testing.assert_array_equal(np.frombuffer(point_file.read_bytes()[16:], dtype=np.double).reshape(-1, 3), points)


@pytest.mark.parametrize('version_output', [STOCK_ELASTIX_VERSION_OUTPUT, PATCHED_ELASTIX_VERSION_OUTPUT])
def test_binary_point_files_read_back_the_points_whatever_the_elastix_version(fake_elastix, tmp_path, version_output):
    fake_elastix(version_output)
    point_file = tmp_path / 'points.bin'
    points = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    elastix.write_points(str(point_file), points, binary=True)

    np.testing.assert_array_equal(elastix.read_points(str(point_file), binary=True), points)
