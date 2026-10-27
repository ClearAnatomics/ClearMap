"""
``setup.py`` builds nothing when it is imported, only when it is run (``python setup.py``, ``pip install``: setuptools
and pip run it as ``__main__``), so that what it decides (the compiler arguments, whether to use OpenMP) can be tested.
It used to compile every extension, and call ``setup()``, as soon as it was imported, and imported numpy and Cython
at the top.
"""
import runpy
import subprocess
import sys
import types

import pytest

from conftest import SETUP_SCRIPT  # pytest puts this directory on the path

REPOSITORY_ROOT = SETUP_SCRIPT.parent
LOAD_THE_SETUP_SCRIPT_AND_REPORT = f"""
import importlib.util, sys
import setuptools

def setup_must_not_be_called(**_):
    raise AssertionError('setup() was called when setup.py was imported')

setuptools.setup = setup_must_not_be_called
arguments_before = list(sys.argv)
spec = importlib.util.spec_from_file_location('clearmap_setup_script', {str(SETUP_SCRIPT)!r})
spec.loader.exec_module(importlib.util.module_from_spec(spec))
assert sys.argv == arguments_before, 'The arguments of the command line were changed'
print(sorted(name for name in ('numpy', 'Cython') if name in sys.modules))
"""


def test_importing_the_setup_script_builds_nothing_and_needs_neither_numpy_nor_cython():
    completed = subprocess.run([sys.executable, '-c', LOAD_THE_SETUP_SCRIPT_AND_REPORT], capture_output=True,
                               text=True, check=False, timeout=60)
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == '[]', 'numpy or Cython was imported with the script'


def cythonize_nothing(extensions, **_):
    return extensions


@pytest.fixture
def setup_calls(monkeypatch):
    """What ``setup()`` receives when the script is run, with Cython replaced by a function that compiles nothing"""
    pytest.importorskip('numpy', reason='The extensions need the headers of numpy')
    import setuptools
    calls = []

    def record_the_call(**keywords):
        calls.append(keywords)

    cython = types.ModuleType('Cython')
    cython_build = types.ModuleType('Cython.Build')
    cython_build.cythonize = cythonize_nothing
    cython.Build = cython_build
    monkeypatch.setitem(sys.modules, 'Cython', cython)
    monkeypatch.setitem(sys.modules, 'Cython.Build', cython_build)
    monkeypatch.setattr(setuptools, 'setup', record_the_call)
    monkeypatch.setattr(sys, 'argv', ['setup.py', 'build_ext'])
    monkeypatch.chdir(REPOSITORY_ROOT)
    return calls


def test_running_the_setup_script_calls_setup_once_with_the_extensions_and_the_version(setup_calls):
    runpy.run_path(str(SETUP_SCRIPT), run_name='__main__')
    assert len(setup_calls) == 1
    keywords, = setup_calls
    assert keywords['name'] == 'ClearMap'
    assert keywords['ext_modules'], 'No extension to build'
    assert all(extension.sources[0].endswith('.pyx') for extension in keywords['ext_modules'])
    assert keywords['version'].count('.') == 2


@pytest.mark.parametrize('system, cpp_version_line, expected', [
    ('linux', None, True),
    ('windows', None, True),
    ('darwin', 'Apple clang version 15.0.0', False),
    ('darwin', 'g++ (Homebrew GCC 13.2.0) 13.2.0', True),
])
def test_openmp_is_used_where_the_compiler_has_it(setup_script, monkeypatch, system, cpp_version_line, expected):
    def say_the_version_of_cpp(*_, **__):
        return f'{cpp_version_line}\nmore\n'.encode('ascii')

    monkeypatch.setattr(setup_script, 'os_name', system)
    monkeypatch.setattr(sys, 'argv', ['setup.py'])
    monkeypatch.setattr(subprocess, 'check_output', say_the_version_of_cpp)
    assert setup_script.openmp_is_used() is expected


def test_an_unknown_system_is_refused(setup_script, monkeypatch):
    monkeypatch.setattr(setup_script, 'os_name', 'plan9')
    monkeypatch.setattr(sys, 'argv', ['setup.py'])
    with pytest.raises(ValueError, match='plan9'):
        setup_script.openmp_is_used()


def test_no_openmp_turns_it_off_and_is_not_left_for_setuptools(setup_script, monkeypatch):
    monkeypatch.setattr(sys, 'argv', ['setup.py', 'build_ext', '--no-openmp', '--inplace'])
    assert setup_script.openmp_is_used() is False
    assert sys.argv == ['setup.py', 'build_ext', '--inplace']
