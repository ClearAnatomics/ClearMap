"""
setup.py rebuilds every extension when they were last built with another Python, NumPy or Cython.

setuptools and Cython skip an extension whose compiled file is newer than its source. A checkout used by two
environments (3.1 with NumPy 1, 3.2 with NumPy 2) so kept the modules compiled against NumPy 1, which failed at import
under NumPy 2: "numpy.core.multiarray failed to import (auto-generated because you didn't call 'numpy.import_array()'
...)". Only ``python setup.py build_ext --inplace --force`` repaired it.
"""
import sys
import types

import pytest
from setuptools.dist import Distribution

from conftest import SETUP_SCRIPT  # pytest puts this directory on the path

CYTHON_VERSION = '3.1.0'


def setup_nothing(**_):
    """setuptools.setup, replaced: the tests decide nothing about the installation"""


def compile_nothing(_command):
    """build_ext.run of setuptools, replaced: nothing is compiled"""
NUMPY_1_SIGNATURE_PYTHON_311 = {'python': '3.11', 'numpy': '1.26', 'cython': CYTHON_VERSION[:3]}


@pytest.fixture
def run_main(setup_script, monkeypatch, tmp_path):
    """Run main() of setup.py with a fake Cython and setup(); returns the keyword arguments given to cythonize"""
    pytest.importorskip('numpy', reason='The extensions need the headers of numpy')
    cythonize_calls = []

    def record_cythonize(extensions, **keywords):
        cythonize_calls.append(keywords)
        return extensions

    cython = types.ModuleType('Cython')
    cython.__version__ = CYTHON_VERSION
    cython_build = types.ModuleType('Cython.Build')
    cython_build.cythonize = record_cythonize
    cython.Build = cython_build
    monkeypatch.setitem(sys.modules, 'Cython', cython)
    monkeypatch.setitem(sys.modules, 'Cython.Build', cython_build)
    monkeypatch.setattr(setup_script, 'setup', setup_nothing)
    monkeypatch.setattr(setup_script, 'BUILD_SIGNATURE_PATH', tmp_path / 'build' / 'clearmap_build_signature.json')
    monkeypatch.setattr(setup_script, 'FORCE', False)
    monkeypatch.setattr(setup_script.parallel_build_ext, 'force_rebuild', False)
    monkeypatch.setattr(setup_script.parallel_build_ext, 'build_signature', None)
    monkeypatch.setattr(sys, 'argv', ['setup.py', 'build_ext'])
    monkeypatch.chdir(SETUP_SCRIPT.parent)

    def run():
        setup_script.main()
        return cythonize_calls[-1]
    return run


def test_extensions_built_for_numpy_1_are_all_rebuilt_for_another_numpy(setup_script, run_main):
    setup_script.record_build_signature(setup_script.BUILD_SIGNATURE_PATH, NUMPY_1_SIGNATURE_PYTHON_311)
    if setup_script.current_build_signature() == NUMPY_1_SIGNATURE_PYTHON_311:
        pytest.skip('this environment is the one of the record')

    assert run_main()['force'] is True
    assert setup_script.parallel_build_ext.force_rebuild is True


def test_extensions_built_in_this_environment_are_not_rebuilt(setup_script, run_main):
    setup_script.record_build_signature(setup_script.BUILD_SIGNATURE_PATH, setup_script.current_build_signature())

    assert run_main()['force'] is False
    assert setup_script.parallel_build_ext.force_rebuild is False


def test_without_a_record_of_the_last_build_every_extension_is_rebuilt(setup_script, run_main):
    assert run_main()['force'] is True


def test_a_damaged_record_is_read_as_no_record(setup_script, tmp_path):
    path = tmp_path / 'clearmap_build_signature.json'
    path.write_text('{"python": "3.1', encoding='utf-8')

    assert setup_script.recorded_build_signature(path) is None


def test_a_forced_rebuild_reaches_build_ext(setup_script, monkeypatch):
    monkeypatch.setattr(setup_script.parallel_build_ext, 'force_rebuild', True)
    command = setup_script.parallel_build_ext(Distribution({'name': 'probe'}))
    command.initialize_options()

    command.finalize_options()

    assert command.force


def test_build_ext_records_what_it_built_with_once_it_has_built(setup_script, monkeypatch, tmp_path):
    signature = {'python': '3.12', 'numpy': '2.3', 'cython': '3.1'}
    monkeypatch.setattr(setup_script, 'BUILD_SIGNATURE_PATH', tmp_path / 'build' / 'clearmap_build_signature.json')
    monkeypatch.setattr(setup_script.parallel_build_ext, 'build_signature', signature)
    monkeypatch.setattr(setup_script._build_ext, 'run', compile_nothing)
    command = setup_script.parallel_build_ext(Distribution({'name': 'probe'}))

    command.run()

    assert setup_script.recorded_build_signature(setup_script.BUILD_SIGNATURE_PATH) == signature
