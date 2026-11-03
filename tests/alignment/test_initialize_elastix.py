"""
initialize_elastix takes the elastix folder as argument (as the message of check_elastix_initialized advises) or,
without argument, from the settings. With an argument it always raised "Cannot find elastix path!".
"""
import os

import pytest

from ClearMap.Alignment import Elastix as elastix
from ClearMap.Utils.exceptions import ClearMapRuntimeError


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
