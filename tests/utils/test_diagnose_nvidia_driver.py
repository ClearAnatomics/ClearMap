"""
A broken NVIDIA driver (out of sync with the kernel after an update, until a reboot) makes the viewers and CUDA fail
with many unrelated messages. nvidia-smi tells it, so it is asked at start. A machine without nvidia-smi is fine.
"""
import subprocess

import pytest

from ClearMap.Utils import nvidia_utils

DRIVER_NOT_REACHED = ("NVIDIA-SMI has failed because it couldn't communicate with the NVIDIA driver. "
                      "Make sure that the latest NVIDIA driver is installed and running.")
VERSION_MISMATCH = 'Failed to initialize NVML: Driver/library version mismatch'


def _fake_nvidia_smi(monkeypatch, *, returncode=0, stdout='', stderr='', exception=None):
    """Put a nvidia-smi on the PATH which answers like this"""
    def run(command, **kwargs):
        assert command == [nvidia_utils.NVIDIA_SMI_EXECUTABLE, '-L']
        assert kwargs['timeout'] == nvidia_utils.NVIDIA_SMI_TIMEOUT_S  # A broken driver can hang
        if exception is not None:
            raise exception
        return subprocess.CompletedProcess(command, returncode, stdout, stderr)
    monkeypatch.setattr(nvidia_utils.shutil, 'which', lambda name: f'/usr/bin/{name}')
    monkeypatch.setattr(nvidia_utils.subprocess, 'run', run)


def test_a_machine_without_nvidia_smi_has_nothing_to_diagnose(monkeypatch):
    monkeypatch.setattr(nvidia_utils.shutil, 'which', lambda name: None)

    assert nvidia_utils.diagnose_nvidia_driver() is None


def test_a_driver_which_lists_its_gpus_is_fine(monkeypatch):
    _fake_nvidia_smi(monkeypatch, stdout='GPU 0: NVIDIA GeForce RTX 3090 (UUID: GPU-1234)\n')

    assert nvidia_utils.diagnose_nvidia_driver() is None


@pytest.mark.parametrize('output', [DRIVER_NOT_REACHED, VERSION_MISMATCH])
def test_a_driver_which_does_not_answer_is_reported_with_what_nvidia_smi_said_and_the_reboot(monkeypatch, output):
    _fake_nvidia_smi(monkeypatch, returncode=9, stdout=output + '\n')

    message = nvidia_utils.diagnose_nvidia_driver()

    assert output in message
    assert 'Reboot' in message


def test_the_error_stream_of_nvidia_smi_is_read_too(monkeypatch):
    _fake_nvidia_smi(monkeypatch, returncode=18, stderr=VERSION_MISMATCH + '\n')

    assert VERSION_MISMATCH in nvidia_utils.diagnose_nvidia_driver()


def test_a_failure_without_any_output_reports_the_exit_status(monkeypatch):
    _fake_nvidia_smi(monkeypatch, returncode=255)

    assert 'exit status 255' in nvidia_utils.diagnose_nvidia_driver()


def test_a_driver_which_hangs_is_reported(monkeypatch):
    _fake_nvidia_smi(monkeypatch, exception=subprocess.TimeoutExpired('nvidia-smi', nvidia_utils.NVIDIA_SMI_TIMEOUT_S))

    assert 'did not answer' in nvidia_utils.diagnose_nvidia_driver()


def test_a_nvidia_smi_which_cannot_be_run_is_reported(monkeypatch):
    _fake_nvidia_smi(monkeypatch, exception=PermissionError('Permission denied'))

    assert 'could not be run' in nvidia_utils.diagnose_nvidia_driver()
