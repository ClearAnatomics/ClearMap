"""
The queries to the driver run nvidia-smi directly (no shell), so that a path with a space or a quote in it is
a path and not a broken command. They are tested with a real program named nvidia-smi found through the PATH.
"""
import os
import stat
import time

import pytest

from ClearMap.Utils import nvidia_utils

FAKE_NVIDIA_SMI = """#!/bin/sh
echo "$@" > "$(dirname "$0")/arguments.txt"
echo "1024, 24576, 7"
"""
WAIT_FOR_BACKGROUND_QUERY_S = 5


@pytest.fixture
def fake_nvidia_smi(tmp_path, monkeypatch):
    """A nvidia-smi which records its arguments and answers like a GPU with 1024 of 24576 MiB used"""
    bin_dir = tmp_path / 'bin'
    bin_dir.mkdir()
    program = bin_dir / 'nvidia-smi'
    program.write_text(FAKE_NVIDIA_SMI)
    program.chmod(program.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv('PATH', f'{bin_dir}{os.pathsep}{os.environ["PATH"]}')
    return bin_dir


def _arguments_received(fake_dir):
    return (fake_dir / 'arguments.txt').read_text().split()


def test_a_query_asks_for_one_property_without_units_by_default(fake_nvidia_smi):
    nvidia_utils.smi_query('memory.free')

    assert _arguments_received(fake_nvidia_smi) == ['--query-gpu=memory.free', '--format=noheader,csv,nounits']


def test_a_query_keeps_the_units_when_asked(fake_nvidia_smi):
    nvidia_utils.smi_query('utilization.gpu', units=True)

    assert _arguments_received(fake_nvidia_smi) == ['--query-gpu=utilization.gpu', '--format=noheader,csv']


def test_the_gpu_parameters_are_written_to_a_file_whose_path_has_spaces(fake_nvidia_smi, tmp_path):
    dest_file = tmp_path / 'a folder with spaces' / "gpu 'params'.txt"
    dest_file.parent.mkdir()

    nvidia_utils.gpu_params(dest_file)

    deadline = time.monotonic() + WAIT_FOR_BACKGROUND_QUERY_S
    while not dest_file.exists() or not dest_file.read_text():  # The query runs in the background
        assert time.monotonic() < deadline, f'{dest_file} was not written within {WAIT_FOR_BACKGROUND_QUERY_S} s'
        time.sleep(0.05)
    assert dest_file.read_text().strip() == '1024, 24576, 7'
