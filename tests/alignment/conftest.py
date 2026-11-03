"""
Fixtures to run the transformix functions of :mod:`ClearMap.Alignment.Elastix` with a fake transformix: an identity
transform that copies its input points, writes empty result images and records its arguments.
"""
import json
import sys

import pytest

from ClearMap.Alignment import Elastix as elastix

FAKE_TRANSFORMIX_LOG_VARIABLE = 'FAKE_TRANSFORMIX_LOG'
FAKE_TRANSFORMIX_SLEEP_VARIABLE = 'FAKE_TRANSFORMIX_SLEEP_SECONDS'

FAKE_TRANSFORMIX_SOURCE = f'''#!{sys.executable}
"""Identity transformix: copies its input points, writes empty result images, and records its arguments"""
import json, os, shutil, sys, time
from pathlib import Path

arguments = dict(zip(sys.argv[1::2], sys.argv[2::2]))
with open(os.environ['{FAKE_TRANSFORMIX_LOG_VARIABLE}'], 'a') as log:
    log.write(json.dumps(arguments) + '\\n')
time.sleep(float(os.environ.get('{FAKE_TRANSFORMIX_SLEEP_VARIABLE}', 0)))  # time for concurrent runs to overlap
output_directory = Path(arguments['-out'])
if arguments.get('-def', 'all') != 'all':
    shutil.copyfile(arguments['-def'], output_directory / ('outputpoints' + Path(arguments['-def']).suffix))
else:
    (output_directory / 'result.mhd').touch()
'''


@pytest.fixture()
def temp_folder(tmp_path, monkeypatch):
    folder = tmp_path / 'temp'
    folder.mkdir()
    monkeypatch.setattr(elastix.tempfile, 'tempdir', str(folder))
    return folder


@pytest.fixture()
def transformix_runs(tmp_path, monkeypatch):
    """Use the fake transformix; returns a function giving the arguments of each run so far"""
    executable = tmp_path / 'transformix'
    executable.write_text(FAKE_TRANSFORMIX_SOURCE)
    executable.chmod(0o755)
    log_path = tmp_path / 'transformix_runs.jsonl'
    monkeypatch.setenv(FAKE_TRANSFORMIX_LOG_VARIABLE, str(log_path))
    monkeypatch.setattr(elastix, 'transformix_binary', str(executable))
    monkeypatch.setattr(elastix, 'initialized', True)

    def runs():
        return [json.loads(line) for line in log_path.read_text().splitlines()]
    return runs


@pytest.fixture()
def transform_directory(tmp_path):
    directory = tmp_path / 'transform'
    directory.mkdir()
    (directory / 'TransformParameters.0.txt').write_text('(InitialTransformParametersFileName "NoInitialTransform")\n')
    return directory
