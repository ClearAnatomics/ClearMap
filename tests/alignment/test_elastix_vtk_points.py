"""
With elastix >= 5.3, transformix answers a binary legacy VTK point file with a binary VTK point file (elastix pull
request #1253), so a stock elastix transforms millions of points without text files. ClearMap writes and reads that
format with numpy, and chooses the point format from the configured elastix in one place, point_file_extension.
"""
import functools

import numpy as np
import pytest
from packaging.version import Version

from ClearMap.Alignment import Elastix as elastix
from ClearMap.Utils.exceptions import ClearMapIoException, ClearMapValueError

POINTS = np.array([[1.5, -2.25, 1e4], [4.0, 5.0, 6.0]])
# The header of the outputpoints.vtk of transformix 5.3.1 for an input of 0 points: ITK writes no POINTS line
ITK_EMPTY_MESH_HEADER = b'# vtk DataFile Version 2.0\nFile written by itkPolyDataMeshIO\nBINARY\nDATASET POLYDATA\n'


@pytest.fixture()
def elastix_version(monkeypatch):
    def configure(version, system='Linux'):
        monkeypatch.setattr(elastix, 'get_elastix_version', functools.partial(Version, version))
        monkeypatch.setattr(elastix.platform, 'system', functools.partial(str, system))
    return configure


def test_a_vtk_point_file_is_a_legacy_binary_vtk_header_then_big_endian_doubles(tmp_path):
    point_file = tmp_path / 'points.vtk'

    elastix.write_vtk_points(str(point_file), POINTS)

    header = (b'# vtk DataFile Version 2.0\nClearMap points\nBINARY\nDATASET POLYDATA\nPOINTS 2 double\n')
    content = point_file.read_bytes()
    assert content.startswith(header)
    np.testing.assert_array_equal(np.frombuffer(content[len(header):], dtype='>f8').reshape(-1, 3), POINTS)


@pytest.mark.parametrize('points', [POINTS, np.zeros((0, 3))])
def test_a_vtk_point_file_reads_back_the_points(tmp_path, points):
    point_file = tmp_path / 'points.vtk'
    elastix.write_vtk_points(str(point_file), points)

    np.testing.assert_array_equal(elastix.read_vtk_points(str(point_file)), points)


def test_a_vtk_point_file_without_points_line_holds_no_points(tmp_path):
    point_file = tmp_path / 'outputpoints.vtk'
    point_file.write_bytes(ITK_EMPTY_MESH_HEADER)

    assert elastix.read_vtk_points(str(point_file)).shape == (0, 3)


def test_float_coordinates_are_read_as_well(tmp_path):
    point_file = tmp_path / 'outputpoints.vtk'
    point_file.write_bytes(ITK_EMPTY_MESH_HEADER + b'POINTS 2 float\n' + POINTS.astype('>f4').tobytes())

    np.testing.assert_array_equal(elastix.read_vtk_points(str(point_file)), POINTS.astype(np.float32))


def test_a_truncated_vtk_point_file_is_refused(tmp_path):
    point_file = tmp_path / 'outputpoints.vtk'
    point_file.write_bytes(ITK_EMPTY_MESH_HEADER + b'POINTS 3 double\n' + POINTS.astype('>f8').tobytes())

    with pytest.raises(ClearMapIoException, match='announces 3 points but holds 2'):
        elastix.read_vtk_points(str(point_file))


def test_an_unknown_coordinate_type_is_refused(tmp_path):
    point_file = tmp_path / 'outputpoints.vtk'
    point_file.write_bytes(ITK_EMPTY_MESH_HEADER + b'POINTS 2 int\n' + bytes(24))

    with pytest.raises(ClearMapValueError, match='int'):
        elastix.read_vtk_points(str(point_file))


@pytest.mark.parametrize('version, system, binary, indices, expected_extension', [
    ('5.3.1', 'Linux', True, False, 'vtk'),
    ('5.3.0', 'Darwin', True, False, 'vtk'),
    ('5.3.1', 'Linux', True, True, 'txt'),  # VTK point files hold physical coordinates only
    ('4.900', 'Linux', True, False, 'bin'),  # the patched build shipped with ClearMap
    ('4.900', 'Linux', True, True, 'bin'),
    ('4.900', 'Darwin', True, False, 'txt'),  # no patched build there
    ('5.2', 'Darwin', True, False, 'txt'),
    ('5.3.1', 'Linux', False, False, 'txt'),
])
def test_the_point_format_follows_the_configured_elastix(elastix_version, version, system, binary, indices,
                                                         expected_extension):
    elastix_version(version, system)

    assert elastix.point_file_extension(binary=binary, indices=indices) == expected_extension


def test_points_go_through_transformix_as_vtk_with_elastix_5_3(elastix_version, transformix_runs,
                                                                transform_directory, temp_folder):
    elastix_version('5.3.1')

    transformed = elastix.transform_points(POINTS, transform_directory=str(transform_directory))

    np.testing.assert_array_equal(transformed, POINTS)
    assert transformix_runs()[0]['-def'].endswith('.vtk')


def test_an_existing_vtk_point_file_is_given_to_transformix_as_is(elastix_version, transformix_runs,
                                                                   transform_directory, temp_folder, tmp_path):
    elastix_version('5.3.1')
    point_file = tmp_path / 'my_points.vtk'
    elastix.write_vtk_points(str(point_file), POINTS)

    transformed = elastix.transform_points(str(point_file), transform_directory=str(transform_directory))

    np.testing.assert_array_equal(transformed, POINTS)
    assert transformix_runs()[0]['-def'] == str(point_file)


def test_a_temp_file_of_another_format_is_refused(elastix_version, transformix_runs, transform_directory,
                                                  temp_folder, tmp_path):
    elastix_version('5.3.1')

    with pytest.raises(ClearMapValueError, match=r'must end with "\.vtk"'):
        elastix.transform_points(POINTS, transform_directory=str(transform_directory),
                                 temp_file=str(tmp_path / 'points.bin'))
    assert list(temp_folder.iterdir()) == []
