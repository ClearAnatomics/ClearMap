"""
The Qt binding of the process is chosen once, before any Qt import, and qtpy and pyqtgraph are told the same one.

qtpy prefers PyQt5 and pyqtgraph PyQt6: with both installed, they would each load their own. The user chooses with
QT_API (qtpy's variable); otherwise PyQt6 if installed, else PyQt5; a process without Qt is left alone.
"""
import os
import subprocess
import sys

import pytest

from ClearMap.gui import qt_binding
from ClearMap.Utils.exceptions import ClearMapValueError
from ClearMap.gui.qt_binding import PYQTGRAPH_QT_LIB_ENV_VAR, QT_API_ENV_VAR, choose_qt_binding


@pytest.fixture()
def environment(monkeypatch):
    """No binding requested, and the installed bindings set by the test"""
    for name in (QT_API_ENV_VAR, PYQTGRAPH_QT_LIB_ENV_VAR):
        monkeypatch.setenv(name, '')  # Records the variable, so that the value choose_qt_binding sets is undone
        monkeypatch.delenv(name)

    def install(*module_names):
        monkeypatch.setattr(qt_binding, '_is_installed', frozenset(module_names).__contains__)
    return install


@pytest.mark.parametrize('installed, chosen, pyqtgraph_lib', [(('PyQt5', 'PyQt6'), 'pyqt6', 'PyQt6'),
                                                               (('PyQt6',), 'pyqt6', 'PyQt6'),
                                                               (('PyQt5',), 'pyqt5', 'PyQt5')])
def test_without_a_request_pyqt6_is_preferred_and_qtpy_and_pyqtgraph_get_the_same(environment, monkeypatch,
                                                                                  installed, chosen, pyqtgraph_lib):
    environment(*installed)

    assert choose_qt_binding() == chosen
    assert (os.environ[QT_API_ENV_VAR], os.environ[PYQTGRAPH_QT_LIB_ENV_VAR]) == \
        (chosen, pyqtgraph_lib)


def test_the_binding_requested_with_qt_api_wins(environment, monkeypatch):
    environment('PyQt5', 'PyQt6')
    monkeypatch.setenv(QT_API_ENV_VAR, 'PyQt5')  # qtpy accepts any case

    assert choose_qt_binding() == 'pyqt5'
    assert os.environ[PYQTGRAPH_QT_LIB_ENV_VAR] == 'PyQt5'


def test_a_process_without_qt_is_left_alone(environment):
    environment()

    assert choose_qt_binding() is None
    assert QT_API_ENV_VAR not in os.environ and PYQTGRAPH_QT_LIB_ENV_VAR not in os.environ


@pytest.mark.parametrize('requested', ['pyside6', 'pyside2', 'qt7'])
def test_a_binding_that_clearmap_does_not_support_is_refused(environment, monkeypatch, requested):
    environment('PyQt6')
    monkeypatch.setenv(QT_API_ENV_VAR, requested)

    with pytest.raises(ClearMapValueError, match=requested):
        choose_qt_binding()


def test_pyqtgraph_set_to_another_binding_than_qtpy_is_refused(environment, monkeypatch):
    environment('PyQt5', 'PyQt6')
    monkeypatch.setenv(QT_API_ENV_VAR, 'pyqt6')
    monkeypatch.setenv(PYQTGRAPH_QT_LIB_ENV_VAR, 'PyQt5')

    with pytest.raises(ClearMapValueError, match='disagrees'):
        choose_qt_binding()


def test_an_imported_binding_counts_as_installed_even_without_a_module_spec(monkeypatch):
    monkeypatch.setitem(qt_binding.sys.modules, 'PyQt6', type(qt_binding)('PyQt6'))  # A module without __spec__

    assert qt_binding._is_installed('PyQt6')


def _environment_after_importing(module_name, tmp_path):
    """QT_API in a new interpreter (in this one ClearMap is already imported) where PyQt6 is installed"""
    (tmp_path / 'PyQt6').mkdir()
    (tmp_path / 'PyQt6' / '__init__.py').touch()  # Found by find_spec, never imported
    environment = {name: value for name, value in os.environ.items()
                   if name not in (QT_API_ENV_VAR, PYQTGRAPH_QT_LIB_ENV_VAR)}
    environment['PYTHONPATH'] = os.pathsep.join(filter(None, [str(tmp_path), os.environ.get('PYTHONPATH')]))
    code = f'import os, {module_name}; print(os.environ.get({QT_API_ENV_VAR!r}))'
    result = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, env=environment,
                            check=True)
    return result.stdout.strip()


@pytest.mark.parametrize('package', ['ClearMap.gui', 'ClearMap.Visualization.Qt'])
def test_importing_a_package_that_may_import_qt_chooses_the_binding(tmp_path, package):
    assert _environment_after_importing(package, tmp_path) == 'pyqt6'


def test_importing_clearmap_leaves_the_qt_binding_to_the_packages_that_import_qt(tmp_path):
    assert _environment_after_importing('ClearMap', tmp_path) == 'None'
