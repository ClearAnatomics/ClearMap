"""
``python -m ClearMap.run`` runs a pipeline of an experiment folder headless: no plot (they need a display), a status
the scheduler can read, and a pipeline imported only when it is run.
"""
import collections
import os
import shutil
import subprocess
import sys
import tempfile
import types

import pytest

from ClearMap import run
from ClearMap.Utils.exceptions import ClearMapFileNotFoundError
from ClearMap.Utils.utilities import ExitCode

PIPELINE_NAME = 'tube_map'
MODULE_OF_THE_PIPELINE = run.PIPELINE_MODULES[PIPELINE_NAME]


class RecordingPipelineModule(types.ModuleType):
    def __init__(self):
        super().__init__(MODULE_OF_THE_PIPELINE)
        self.calls = []

    def main(self, src_directory, show_plots=True):
        self.calls.append((src_directory, show_plots))


DiskUsage = collections.namedtuple('DiskUsage', 'total used free')
ONE_TIB = 1024 ** 4


@pytest.fixture(autouse=True)
def temporary_folder_of_the_run(tmp_path, monkeypatch):
    """The run needs a temporary folder: this one, with room, and a machine config of the test, not the user's"""
    folder = tmp_path / 'tmp_of_the_run'
    monkeypatch.setenv('CLEARMAP_TMP', str(folder))
    monkeypatch.setenv('CLEARMAP_MACHINE_CFG', str(tmp_path / 'machine.yml'))
    monkeypatch.setattr(shutil, 'disk_usage', lambda path: DiskUsage(total=ONE_TIB, used=0, free=ONE_TIB))
    # The run patches the process: monkeypatch puts them back
    monkeypatch.setattr(tempfile, 'tempdir', tempfile.tempdir)
    for name in ('TMP', 'TEMP', 'TMPDIR'):
        monkeypatch.setenv(name, os.environ.get(name, '/tmp'))
    return folder


@pytest.fixture
def pipeline_module(monkeypatch):
    module = RecordingPipelineModule()
    monkeypatch.setitem(sys.modules, MODULE_OF_THE_PIPELINE, module)
    return module


def test_runs_the_pipeline_on_the_resolved_folder_without_plots(pipeline_module, tmp_path):
    status = run.main([PIPELINE_NAME, str(tmp_path)])

    assert status == ExitCode.SUCCESS
    assert pipeline_module.calls == [(str(tmp_path.resolve()), False)]


def test_check_only_does_not_run_the_pipeline(pipeline_module, tmp_path, capsys):
    status = run.main([PIPELINE_NAME, str(tmp_path), '--check-only'])

    assert status == ExitCode.SUCCESS
    assert pipeline_module.calls == []
    assert 'ready to run' in capsys.readouterr().out


def test_a_missing_folder_fails_with_a_status_and_a_message(pipeline_module, tmp_path, capsys):
    status = run.main([PIPELINE_NAME, str(tmp_path / 'missing')])

    assert status == ExitCode.FAILURE
    assert pipeline_module.calls == []
    assert 'does not exist' in capsys.readouterr().err


def test_a_failing_pipeline_gives_a_status_and_the_traceback(monkeypatch, tmp_path, capsys):
    class FailingModule(RecordingPipelineModule):
        def main(self, src_directory, show_plots=True):
            raise RuntimeError('out of memory')

    monkeypatch.setitem(sys.modules, MODULE_OF_THE_PIPELINE, FailingModule())

    status = run.main([PIPELINE_NAME, str(tmp_path)])

    assert status == ExitCode.FAILURE
    error_output = capsys.readouterr().err
    assert 'Traceback' in error_output and 'out of memory' in error_output


def test_the_temporary_folder_is_set_before_the_pipeline_is_imported_and_printed(
        pipeline_module, temporary_folder_of_the_run, tmp_path, monkeypatch, capsys):
    temporary_folders_at_import = []

    def import_module(name):
        temporary_folders_at_import.append(tempfile.gettempdir())
        return pipeline_module

    monkeypatch.setattr(run.importlib, 'import_module', import_module)

    status = run.main([PIPELINE_NAME, str(tmp_path)])

    assert status == ExitCode.SUCCESS
    assert temporary_folders_at_import == [str(temporary_folder_of_the_run)]
    assert f'Temporary folder: {temporary_folder_of_the_run} (1024 GB free)' in capsys.readouterr().out


@pytest.mark.parametrize('extra_arguments', [[], ['--check-only']])
def test_without_a_temporary_folder_the_run_stops_before_the_pipeline_and_says_how_to_set_it(
        extra_arguments, pipeline_module, tmp_path, monkeypatch, capsys):
    monkeypatch.delenv('CLEARMAP_TMP')

    status = run.main([PIPELINE_NAME, str(tmp_path), *extra_arguments])

    assert status == ExitCode.FAILURE
    assert pipeline_module.calls == []
    assert 'CLEARMAP_TMP' in capsys.readouterr().err


def test_an_unknown_pipeline_is_a_usage_error(tmp_path):
    with pytest.raises(SystemExit) as exit_info:
        run.main(['no_such_pipeline', str(tmp_path)])

    assert exit_info.value.code == ExitCode.USAGE_ERROR


def test_check_experiment_folder_raises_a_clearmap_exception(tmp_path):
    with pytest.raises(ClearMapFileNotFoundError):
        run.check_experiment_folder(tmp_path / 'missing')


def test_importing_the_runner_does_not_import_the_pipelines():
    """In a new interpreter: in this one, any test may already have imported them"""
    pipeline_modules = list(run.PIPELINE_MODULES.values())
    script = ('import sys, ClearMap.run\n'
              f'print([name for name in {pipeline_modules!r} if name in sys.modules])')

    result = subprocess.run([sys.executable, '-c', script], capture_output=True, text=True, check=True)

    assert result.stdout.strip().splitlines()[-1] == '[]'


@pytest.mark.parametrize('arguments, expected_status', [
    (['tube_map', '{folder}', '--check-only'], ExitCode.SUCCESS),
    (['tube_map', '{folder}/missing'], ExitCode.FAILURE),
    (['no_such_pipeline', '{folder}'], ExitCode.USAGE_ERROR),
])
def test_the_process_exits_with_the_status_the_scheduler_reads(arguments, expected_status, tmp_path):
    """The status of the process, not only the value main() returns: a scheduler only sees the former"""
    command = [sys.executable, '-m', 'ClearMap.run'] + [argument.format(folder=tmp_path) for argument in arguments]
    env = {**os.environ, 'HOME': str(tmp_path), 'CLEARMAP_TMP': str(tmp_path / 'tmp_of_the_run'),
           'CLEARMAP_MACHINE_CFG': str(tmp_path / 'machine.yml')}

    result = subprocess.run(command, capture_output=True, text=True, check=False, env=env)

    assert result.returncode == expected_status
