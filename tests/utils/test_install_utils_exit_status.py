"""
install_gui.sh reads the exit status of PytorchVersionManager.assert_cuda and check_pytorch, and runs install_utils in a
temporary environment where numpy and psutil are not installed (only python, packaging, pyyaml and configobj are).
"""
import sys
import types

import pytest

from ClearMap.Utils import install_utils
from ClearMap.Utils.utilities import ExitCode
from ClearMap.Utils.install_utils import PytorchVersionManager


class FakeCuda:
    def __init__(self, available):
        self.available = available

    def is_available(self):
        return self.available


def fake_torch(cuda_available):
    module = types.ModuleType('torch')
    module.cuda = FakeCuda(cuda_available)
    return module


def nvidia_smi_lists_a_gpu(command, shell):
    return b'GPU 0: NVIDIA RTX (UUID: GPU-0)'


def exit_status_of(function):
    with pytest.raises(SystemExit) as exit_info:
        function()
    return exit_info.value.code


def test_install_utils_imports_in_the_bootstrap_environment(import_in_bootstrap_environment):
    result = import_in_bootstrap_environment('ClearMap.Utils.install_utils')

    assert result.returncode == 0, result.stderr


def test_check_pytorch_succeeds_when_cuda_is_available(monkeypatch):
    monkeypatch.setitem(sys.modules, 'torch', fake_torch(cuda_available=True))

    assert exit_status_of(PytorchVersionManager.check_pytorch) == ExitCode.SUCCESS


def test_check_pytorch_fails_when_cuda_is_not_available(monkeypatch):
    monkeypatch.setitem(sys.modules, 'torch', fake_torch(cuda_available=False))

    assert exit_status_of(PytorchVersionManager.check_pytorch) == ExitCode.FAILURE


def test_assert_cuda_succeeds_when_nvidia_smi_lists_a_gpu(monkeypatch):
    monkeypatch.setattr(install_utils.subprocess, 'check_output', nvidia_smi_lists_a_gpu)

    assert exit_status_of(PytorchVersionManager.assert_cuda) == ExitCode.SUCCESS
