"""
initialize_elastix takes the elastix folder as argument (as the message of check_elastix_initialized advises) or,
without argument, from the settings. With an argument it always raised "Cannot find elastix path!".
"""
import functools
import os

import pytest

from ClearMap.Alignment import Elastix as elastix
from ClearMap.Utils.exceptions import ClearMapRuntimeError, ClearMapValueError


@pytest.fixture()
def restored_elastix_state(monkeypatch):
    """initialize_elastix sets module globals and the library path variable: restore them after the test"""
    for name in ('elastix_binary', 'transformix_binary', 'elastix_lib', 'initialized'):
        monkeypatch.setattr(elastix, name, getattr(elastix, name))
    for variable in ('LD_LIBRARY_PATH', 'DYLD_LIBRARY_PATH'):
        if variable in os.environ:
            monkeypatch.setenv(variable, os.environ[variable])
        else:
            monkeypatch.delenv(variable, raising=False)


@pytest.fixture()
def elastix_folder(tmp_path):
    """The layout of an elastix release: bin/elastix, bin/transformix, lib/"""
    (tmp_path / 'bin').mkdir()
    (tmp_path / 'lib').mkdir()
    for executable in ('elastix', 'transformix'):
        (tmp_path / 'bin' / executable).touch()
    return tmp_path


def test_an_elastix_folder_given_as_argument_is_used(restored_elastix_state, elastix_folder):
    assert elastix.initialize_elastix(str(elastix_folder)) == str(elastix_folder)

    assert elastix.elastix_binary == str(elastix_folder / 'bin' / 'elastix')
    assert elastix.transformix_binary == str(elastix_folder / 'bin' / 'transformix')
    assert elastix.elastix_lib == str(elastix_folder / 'lib')


def test_without_argument_the_elastix_folder_of_the_settings_is_used(restored_elastix_state, elastix_folder,
                                                                      monkeypatch):
    monkeypatch.setattr(elastix.settings, 'elastix_path', str(elastix_folder))

    assert elastix.initialize_elastix() == str(elastix_folder)


def test_without_any_elastix_folder_the_error_says_where_to_set_it(restored_elastix_state, monkeypatch):
    monkeypatch.setattr(elastix.settings, 'elastix_path', None)

    with pytest.raises(ClearMapRuntimeError, match='elastix_path'):
        elastix.initialize_elastix()


def pretend_to_run_on(monkeypatch, system):
    """Make ``platform.system()`` return ``system`` ('Windows', 'Linux', 'Darwin') for the rest of the test"""
    monkeypatch.setattr(elastix.platform, 'system', functools.partial(str, system))


def make_elastix_folder(root, *, executables_folder, executable_suffix='', library_folder=None):
    folders = {root / executables_folder, root / (library_folder or executables_folder)}
    for folder in folders:
        folder.mkdir(parents=True, exist_ok=True)
    for executable in ('elastix', 'transformix'):
        (root / executables_folder / f'{executable}{executable_suffix}').touch()
    return root


@pytest.fixture()
def no_library_path_variable(monkeypatch):
    for variable in ('LD_LIBRARY_PATH', 'DYLD_LIBRARY_PATH'):
        monkeypatch.delenv(variable, raising=False)


@pytest.mark.parametrize('executables_folder', ['', 'bin'])
def test_the_windows_executables_are_found_with_their_exe_suffix_at_the_root_or_in_bin(
        restored_elastix_state, no_library_path_variable, monkeypatch, tmp_path, executables_folder):
    pretend_to_run_on(monkeypatch, 'Windows')
    make_elastix_folder(tmp_path, executables_folder=executables_folder, executable_suffix='.exe')

    elastix.initialize_elastix(str(tmp_path))

    assert elastix.elastix_binary == str(tmp_path / executables_folder / 'elastix.exe')
    assert elastix.transformix_binary == str(tmp_path / executables_folder / 'transformix.exe')
    assert elastix.elastix_lib == str(tmp_path / executables_folder)


def test_on_windows_no_library_path_variable_is_set(restored_elastix_state, no_library_path_variable, monkeypatch,
                                                    tmp_path):
    pretend_to_run_on(monkeypatch, 'Windows')
    make_elastix_folder(tmp_path, executables_folder='', executable_suffix='.exe')

    elastix.initialize_elastix(str(tmp_path))

    assert 'LD_LIBRARY_PATH' not in os.environ and 'DYLD_LIBRARY_PATH' not in os.environ


@pytest.mark.parametrize('system, variable', [('Linux', 'LD_LIBRARY_PATH'), ('Darwin', 'DYLD_LIBRARY_PATH')])
def test_on_linux_and_macos_the_library_folder_is_added_to_the_library_path_variable(
        restored_elastix_state, no_library_path_variable, monkeypatch, tmp_path, system, variable):
    pretend_to_run_on(monkeypatch, system)
    make_elastix_folder(tmp_path, executables_folder='bin', library_folder='lib')

    elastix.initialize_elastix(str(tmp_path))

    assert os.environ[variable] == str(tmp_path / 'lib')


def test_an_unknown_operating_system_is_refused(restored_elastix_state, monkeypatch, tmp_path):
    pretend_to_run_on(monkeypatch, 'Plan9')
    make_elastix_folder(tmp_path, executables_folder='bin', library_folder='lib')

    with pytest.raises(ClearMapValueError, match='plan9'):
        elastix.initialize_elastix(str(tmp_path))


def test_a_missing_windows_executable_is_reported_with_its_exe_name_and_the_folders_searched(
        restored_elastix_state, monkeypatch, tmp_path):
    pretend_to_run_on(monkeypatch, 'Windows')
    make_elastix_folder(tmp_path, executables_folder='bin')  # the executables of a Linux release: no .exe

    with pytest.raises(ClearMapRuntimeError, match=r'elastix\.exe in .*bin.* or '):
        elastix.initialize_elastix(str(tmp_path))
