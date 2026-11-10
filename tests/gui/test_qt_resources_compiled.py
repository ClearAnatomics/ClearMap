"""
The Qt resources of the GUI are compiled with the ``rcc`` of the Qt which is loaded, then registered at run time:
PyQt6 has no ``pyrcc``. The ``rcc`` is looked for in the folders of the loaded Qt only, never on the PATH (the 5.15 of a
Linux distribution next to a conda Qt 6.11 would compile for the wrong version). The compiled file is named after the
major version of Qt and compiled again only when older than its sources.

A fake ``rcc`` (an executable script) stands for the real one: the sandbox and the CI have no Qt tools.
"""
import enum
import os
import stat
import sys

import pytest
import qtpy

from ClearMap.Utils.exceptions import ClearMapRuntimeError, MissingRequirementException
from ClearMap.gui import qt_resources

pytestmark = pytest.mark.skipif(sys.platform == 'win32', reason='the fake rcc is a shell script')

QRC = '''<RCC><qresource prefix="/icons"><file>icons/a.png</file><file>icons/b.png</file></qresource></RCC>'''
FAKE_RCC = '''#!/bin/sh
echo "$@" >> "{log}"
[ -n "{fail}" ] && echo "{fail}" >&2 && exit 3
while [ $# -gt 0 ]; do [ "$1" = "-o" ] && out="$2"; shift; done
echo compiled > "$out"
'''


class FakeLibraryPath(enum.Enum):
    LibraryExecutablesPath = 'libexec'
    BinariesPath = 'bin'


class FakeQt6LibraryInfo:
    """QLibraryInfo of Qt 6: path() of a LibraryPath"""
    LibraryPath = FakeLibraryPath
    folders = {}

    @classmethod
    def path(cls, which):
        return cls.folders[which.value]


class FakeQt5LibraryInfo:
    """QLibraryInfo of Qt 5: location() of a constant, no path()"""
    BinariesPath, LibraryExecutablesPath = 'bin', 'libexec'
    folders = {}

    @classmethod
    def location(cls, which):
        return cls.folders[which]


def _install_rcc(folder, *, log, fail=''):
    folder.mkdir(parents=True, exist_ok=True)
    rcc = folder / 'rcc'
    rcc.write_text(FAKE_RCC.format(log=log, fail=fail))
    rcc.chmod(rcc.stat().st_mode | stat.S_IXUSR)
    return rcc


@pytest.fixture()
def qt6(tmp_path, monkeypatch):
    """A loaded Qt 6.11.1 whose tools are in tmp_path/libexec (rcc) and tmp_path/bin"""
    FakeQt6LibraryInfo.folders = {'libexec': str(tmp_path / 'libexec'), 'bin': str(tmp_path / 'bin')}
    monkeypatch.setattr(qt_resources, 'QLibraryInfo', FakeQt6LibraryInfo)
    monkeypatch.setattr(qtpy, 'QT_VERSION', '6.11.1', raising=False)
    monkeypatch.setattr(qt_resources, '_registered_files', set())
    return tmp_path


@pytest.fixture()
def resources_folder(tmp_path):
    folder = tmp_path / 'creator'
    (folder / 'icons').mkdir(parents=True)
    (folder / 'icons' / 'a.png').write_bytes(b'a')
    (folder / 'icons' / 'b.png').write_bytes(b'b')
    (folder / 'icons.qrc').write_text(QRC)
    return folder


def test_rcc_is_looked_for_in_the_libexec_folder_of_qt6_before_its_bin_folder(qt6):
    _install_rcc(qt6 / 'bin', log=qt6 / 'log')
    expected = _install_rcc(qt6 / 'libexec', log=qt6 / 'log')

    assert qt_resources.find_rcc() == expected


def test_rcc_is_looked_for_in_the_bin_folder_of_qt5_before_its_libexec_folder(tmp_path, monkeypatch):
    FakeQt5LibraryInfo.folders = {'bin': str(tmp_path / 'bin'), 'libexec': str(tmp_path / 'libexec')}
    monkeypatch.setattr(qt_resources, 'QLibraryInfo', FakeQt5LibraryInfo)
    _install_rcc(tmp_path / 'libexec', log=tmp_path / 'log')
    expected = _install_rcc(tmp_path / 'bin', log=tmp_path / 'log')

    assert qt_resources.find_rcc() == expected


def test_an_rcc_on_the_path_is_not_the_rcc_of_the_loaded_qt(qt6, monkeypatch):
    _install_rcc(qt6 / 'on_the_path', log=qt6 / 'log')
    monkeypatch.setenv('PATH', os.pathsep.join([str(qt6 / 'on_the_path'), os.environ.get('PATH', '')]))

    with pytest.raises(MissingRequirementException, match=r'rcc.*not found|not found'):
        qt_resources.find_rcc()


def test_the_error_of_a_missing_rcc_names_the_folders_looked_in(qt6):
    with pytest.raises(MissingRequirementException) as error:
        qt_resources.find_rcc()

    assert error.value.missing_items == [str(qt6 / 'libexec' / 'rcc'), str(qt6 / 'bin' / 'rcc')]


def test_the_compiled_file_is_named_after_the_major_version_of_qt(qt6, monkeypatch):
    assert qt_resources.compiled_resource_path('creator/icons.qrc').name == 'icons.qt6.rcc'
    monkeypatch.setattr(qtpy, 'QT_VERSION', '5.15.8')
    assert qt_resources.compiled_resource_path('creator/icons.qrc').name == 'icons.qt5.rcc'


def test_the_resources_are_compiled_to_a_binary_file_by_the_rcc_of_the_loaded_qt(qt6, resources_folder):
    _install_rcc(qt6 / 'libexec', log=qt6 / 'log')

    compiled = qt_resources.compile_resources(resources_folder / 'icons.qrc')

    assert compiled == resources_folder / 'icons.qt6.rcc' and compiled.exists()
    assert (qt6 / 'log').read_text().split() == ['-binary', str(resources_folder / 'icons.qrc'), '-o', str(compiled)]


def test_up_to_date_resources_are_not_compiled_again_but_a_newer_image_is(qt6, resources_folder):
    _install_rcc(qt6 / 'libexec', log=qt6 / 'log')
    compiled = qt_resources.compile_resources(resources_folder / 'icons.qrc')
    qt_resources.compile_resources(resources_folder / 'icons.qrc')
    assert len((qt6 / 'log').read_text().splitlines()) == 1

    newer = compiled.stat().st_mtime + 10
    os.utime(resources_folder / 'icons' / 'b.png', (newer, newer))
    qt_resources.compile_resources(resources_folder / 'icons.qrc')

    assert len((qt6 / 'log').read_text().splitlines()) == 2


def test_a_file_listed_by_the_qrc_which_does_not_exist_is_named(qt6, resources_folder):
    _install_rcc(qt6 / 'libexec', log=qt6 / 'log')
    (resources_folder / 'icons' / 'b.png').unlink()

    with pytest.raises(MissingRequirementException) as error:
        qt_resources.compile_resources(resources_folder / 'icons.qrc')

    assert error.value.missing_items == [str(resources_folder / 'icons' / 'b.png')]


def test_a_failing_rcc_says_what_it_said(qt6, resources_folder):
    _install_rcc(qt6 / 'libexec', log=qt6 / 'log', fail='invalid qrc')

    with pytest.raises(ClearMapRuntimeError, match='exit code 3.*invalid qrc'):
        qt_resources.compile_resources(resources_folder / 'icons.qrc')


class FakeResource:
    """QResource: keeps the files registered; refuses what is in ``refused``"""
    registered = []
    refused = ()

    @classmethod
    def registerResource(cls, path):
        cls.registered.append(path)
        return path not in cls.refused


@pytest.fixture()
def fake_resource(monkeypatch):
    monkeypatch.setattr(FakeResource, 'registered', [])
    monkeypatch.setattr(FakeResource, 'refused', ())
    monkeypatch.setattr(qt_resources, 'QResource', FakeResource)
    return FakeResource


def test_every_qrc_of_the_folder_is_registered_once(qt6, resources_folder, fake_resource):
    _install_rcc(qt6 / 'libexec', log=qt6 / 'log')
    (resources_folder / 'graphics_resources.qrc').write_text(QRC)

    qt_resources.register_resources(folder=resources_folder)
    qt_resources.register_resources(folder=resources_folder)

    assert fake_resource.registered == [str(resources_folder / 'graphics_resources.qt6.rcc'),
                                        str(resources_folder / 'icons.qt6.rcc')]


def test_a_file_refused_by_qt_is_reported_with_the_way_out(qt6, resources_folder, fake_resource):
    _install_rcc(qt6 / 'libexec', log=qt6 / 'log')
    fake_resource.refused = (str(resources_folder / 'icons.qt6.rcc'),)

    with pytest.raises(ClearMapRuntimeError, match='delete it'):
        qt_resources.register_resources(folder=resources_folder)
