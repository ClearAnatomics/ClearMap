"""
The elastix parameter files of a result directory are rewritten in place (absolute paths of the initial transforms,
the metric of an inverse transform): next to themselves, keeping their permissions, never through the temp folder
(#97: they became private, chmod 600; #138: "Invalid cross-device link" when the data is on another disk).
"""
import os
import stat
import tempfile

import pytest

from ClearMap.Alignment import Elastix as elastix


@pytest.fixture()
def result_directory(tmp_path):
    (tmp_path / 'TransformParameters.0.txt').write_text('(InitialTransformParametersFileName "NoInitialTransform")\n'
                                                        '(Metric "AdvancedMattesMutualInformation")\n')
    (tmp_path / 'TransformParameters.1.txt').write_text('(InitialTransformParametersFileName '
                                                        '"TransformParameters.0.txt")\n(Size 10 20 30)\n')
    for transform_file in tmp_path.iterdir():
        transform_file.chmod(0o664)
    return tmp_path


@pytest.fixture()
def no_temp_folder(monkeypatch):
    """The temp folder may be on another file system: using it is refused here"""
    def refused(*args, **kwargs):
        raise AssertionError('The parameter files must not go through the temp folder')
    monkeypatch.setattr(tempfile, 'mkstemp', refused)
    monkeypatch.setattr(tempfile, 'NamedTemporaryFile', refused)


def _mode(path):
    return stat.S_IMODE(os.stat(path).st_mode)


def test_the_initial_transforms_get_absolute_paths_in_place_keeping_the_permissions(result_directory, no_temp_folder):
    elastix.set_path_transform_files(str(result_directory))

    second = (result_directory / 'TransformParameters.1.txt').read_text()
    assert second == (f'(InitialTransformParametersFileName "{result_directory / "TransformParameters.0.txt"}")\n'
                      f'(Size 10 20 30)\n')
    assert [_mode(path) for path in sorted(result_directory.iterdir())] == [0o664, 0o664]
    assert sorted(path.name for path in result_directory.iterdir()) == ['TransformParameters.0.txt',
                                                                         'TransformParameters.1.txt']


def test_the_metric_is_replaced_in_place_keeping_the_permissions(result_directory, no_temp_folder):
    parameter_file = result_directory / 'TransformParameters.0.txt'

    elastix.set_metric_parameter_file(str(parameter_file), 'DisplacementMagnitudePenalty')

    assert '(Metric "DisplacementMagnitudePenalty")\n' in parameter_file.read_text()
    assert _mode(parameter_file) == 0o664
    assert [path.name for path in result_directory.iterdir() if path.name.startswith('.')] == []  # No temporary file
