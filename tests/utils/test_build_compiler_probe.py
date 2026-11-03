"""
``setup.py`` asks the compiler whether it knows an option (``-flarge-source-files``, GCC only) before using it.

The probe used to run ``gcc`` whatever compiler builds: without ``gcc`` on the PATH (a conda toolchain names it
``x86_64-conda-linux-gnu-gcc``, a module may load another one), or with a compiler that does not understand
``--help=common`` (clang), ``setup.py`` stopped on an exception before anything was built. Now it asks the compiler
that distutils will use (``CC``...) and, whatever happens, an option the compiler is not seen to list is left out.
"""
import stat
import sys
import time
from pathlib import Path

import pytest


OPTION = '-flarge-source-files'
SHORT_PROBE_TIMEOUT_SECONDS = 0.3
HANGING_SECONDS = 10
MAXIMUM_WAIT_SECONDS = HANGING_SECONDS / 2

pytestmark = pytest.mark.skipif(sys.platform == 'win32',
                                reason='The compilers here are shell scripts (the probe is Linux only)')


def _fake_compiler(directory: Path, help_text: str, exit_code: int = 0) -> Path:
    path = directory / 'fake-cc'
    path.write_text(f'#!/bin/sh\ncat <<"END"\n{help_text}\nEND\nexit {exit_code}\n')
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


def test_an_option_the_compiler_lists_is_found(setup_script, tmp_path):
    compiler = _fake_compiler(tmp_path, f'The following options are language-independent:\n  {OPTION}  Some help.')
    assert setup_script.compiler_lists_option([str(compiler)], OPTION)


def test_an_option_the_compiler_does_not_list_is_left_out(setup_script, tmp_path):
    compiler = _fake_compiler(tmp_path, '  -fopenmp  Some help.')
    assert not setup_script.compiler_lists_option([str(compiler)], OPTION)


def test_a_compiler_that_rejects_the_help_argument_is_left_out(setup_script, tmp_path):
    compiler = _fake_compiler(tmp_path, f'error: unknown argument: --help=common\n{OPTION}', exit_code=1)
    assert not setup_script.compiler_lists_option([str(compiler)], OPTION)


def test_a_compiler_that_is_not_there_is_not_an_error(setup_script, tmp_path):
    assert not setup_script.compiler_lists_option([str(tmp_path / 'no-such-compiler')], OPTION)


def test_no_compiler_is_not_an_error(setup_script):
    assert not setup_script.compiler_lists_option([], OPTION)


def test_a_compiler_that_hangs_is_left_out(setup_script, tmp_path, monkeypatch):
    compiler = tmp_path / 'hanging-cc'
    compiler.write_text(f'#!/bin/sh\nexec sleep {HANGING_SECONDS}\n')
    compiler.chmod(compiler.stat().st_mode | stat.S_IXUSR)
    monkeypatch.setattr(setup_script, 'PROBE_TIMEOUT_SECONDS', SHORT_PROBE_TIMEOUT_SECONDS)
    start = time.monotonic()
    assert not setup_script.compiler_lists_option([str(compiler)], OPTION)
    assert time.monotonic() - start < MAXIMUM_WAIT_SECONDS, 'The probe waited for the compiler instead of giving up'


def test_the_compiler_to_ask_is_the_one_distutils_builds_with(setup_script, tmp_path, monkeypatch):
    compiler = _fake_compiler(tmp_path, OPTION)
    monkeypatch.setenv('CC', str(compiler))
    assert setup_script.build_compiler_command()[0] == str(compiler)


@pytest.mark.parametrize('compiler_help, option_is_used', [(OPTION, True), ('  -fopenmp  Some help.', False)])
def test_the_arguments_of_a_linux_build_follow_the_compiler_that_builds(setup_script, tmp_path, monkeypatch,
                                                                        compiler_help, option_is_used):
    compiler = _fake_compiler(tmp_path, compiler_help)
    monkeypatch.setenv('CC', str(compiler))
    monkeypatch.setattr(setup_script, 'os_name', 'linux')
    assert (OPTION in setup_script.default_compile_arguments()) is option_is_used


def test_a_linux_build_goes_on_without_the_compiler_the_probe_asks(setup_script, tmp_path, monkeypatch):
    monkeypatch.setenv('CC', str(tmp_path / 'no-such-compiler'))
    monkeypatch.setattr(setup_script, 'os_name', 'linux')
    assert OPTION not in setup_script.default_compile_arguments()
