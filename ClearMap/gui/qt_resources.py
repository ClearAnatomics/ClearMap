"""
qt_resources
============

The images of the GUI built in Qt resources (the ``.qrc`` files of ``ClearMap/gui/creator``), registered with Qt.

PyQt6 has no resource compiler (no ``pyrcc``). The ``.qrc`` files are compiled instead by the ``rcc`` that comes with
the Qt which is loaded, into binary resource files that Qt registers at run time (``QResource.registerResource``).
The designer files and their ``:/icons/...`` paths are unchanged, with PyQt5 and PyQt6.

The compiled file is next to its ``.qrc`` and named after the major version of Qt (``icons.qt6.rcc``), so that
switching between PyQt5 and PyQt6 never reuses a file compiled by the other Qt. It is compiled again when it is older
than the ``.qrc`` or than any of the files it lists. It is not tracked (see ``.gitignore``).

Only the folders of the loaded Qt are searched for ``rcc``, never the ``PATH``: another Qt on the system (e.g. the
5.15 of the distribution next to a conda Qt 6) would compile resources for the wrong version.
"""
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

import qtpy
from qtpy.QtCore import QLibraryInfo, QResource

from ClearMap.Utils.exceptions import ClearMapRuntimeError, MissingRequirementException
from ClearMap.Utils.path_utils import is_newer
from ClearMap.Utils.versions import as_version

CREATOR_FOLDER = Path(__file__).parent / 'creator'
RCC_EXECUTABLE_NAMES = ('rcc', 'rcc.exe')

_registered_files = set()


def qt_major_version():
    return as_version(qtpy.QT_VERSION).major


def _qt_tool_folders():
    """The folders where the loaded Qt keeps its tools: the one of ``rcc`` for the Qt of the binding first"""
    if hasattr(QLibraryInfo, 'path'):  # Qt 6: rcc is in the libexec folder (conda-forge: lib/qt6)
        library_path = QLibraryInfo.LibraryPath
        locations = [QLibraryInfo.path(library_path.LibraryExecutablesPath),
                     QLibraryInfo.path(library_path.BinariesPath)]
    else:  # Qt 5: rcc is in the binaries folder
        locations = [QLibraryInfo.location(QLibraryInfo.BinariesPath),
                     QLibraryInfo.location(QLibraryInfo.LibraryExecutablesPath)]
    return [Path(location) for location in locations if location]


def find_rcc():
    """
    The resource compiler of the Qt which is loaded.

    Returns
    -------
    pathlib.Path

    Raises
    ------
    MissingRequirementException
        If the folders of the loaded Qt have no ``rcc``.
    """
    folders = _qt_tool_folders()
    for folder in folders:
        for name in RCC_EXECUTABLE_NAMES:
            if (folder / name).is_file():
                return folder / name
    raise MissingRequirementException(
        f'The resource compiler (rcc) of the Qt which is loaded ({qtpy.API}, Qt {qtpy.QT_VERSION}) was not found. '
        f'Install the tools of this Qt (with conda, the Qt package of the same version ships rcc); '
        f'the PATH is not searched, as it may hold the rcc of another Qt version',
        missing_items=[str(folder / RCC_EXECUTABLE_NAMES[0]) for folder in folders])


def resource_files(qrc_path):
    """
    The files listed by a ``.qrc`` file.

    Returns
    -------
    dict
        ``{':/<prefix>/<file>': pathlib.Path of the file}``, the resource paths being the ones of the designer files.
    """
    qrc_path = Path(qrc_path)
    resources = {}
    for qresource in ET.parse(qrc_path).getroot().iter('qresource'):
        prefix = qresource.get('prefix', '/').strip('/')
        for file in qresource.iter('file'):
            file_name = file.text.strip()
            resource_path = ':/' + '/'.join(part for part in (prefix, file_name) if part)
            resources[resource_path] = qrc_path.parent / file_name
    return resources


def compiled_resource_path(qrc_path):
    qrc_path = Path(qrc_path)
    return qrc_path.with_name(f'{qrc_path.stem}.qt{qt_major_version()}.rcc')


def _needs_compiling(qrc_path, compiled_path):
    sources = [qrc_path, *resource_files(qrc_path).values()]
    missing = [str(source) for source in sources if not source.is_file()]
    if missing:
        raise MissingRequirementException(f'{qrc_path} lists files which do not exist', missing_items=missing)
    return not compiled_path.exists() or any(is_newer(source, compiled_path) for source in sources)


def compile_resources(qrc_path):
    """
    Compile a ``.qrc`` file to the binary resource file of the loaded Qt, unless it is up to date.

    Returns
    -------
    pathlib.Path
        The compiled file.

    Raises
    ------
    MissingRequirementException
        If there is no ``rcc`` for the loaded Qt, or the ``.qrc`` lists files which do not exist.
    ClearMapRuntimeError
        If ``rcc`` fails.
    """
    qrc_path = Path(qrc_path)
    compiled_path = compiled_resource_path(qrc_path)
    if _needs_compiling(qrc_path, compiled_path):
        command = [str(find_rcc()), '-binary', str(qrc_path), '-o', str(compiled_path)]
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode != 0:
            raise ClearMapRuntimeError(f'rcc failed on {qrc_path} (exit code {result.returncode}): '
                                       f'{result.stderr.strip()}', command=' '.join(command))
    return compiled_path


def register_resources(*, folder=CREATOR_FOLDER):
    """
    Compile (if needed) and register with Qt the resources of every ``.qrc`` file of a folder.

    Needs no QApplication. Registering the same file twice is skipped.

    Raises
    ------
    ClearMapRuntimeError
        If Qt refuses a compiled file.
    """
    for qrc_path in sorted(Path(folder).glob('*.qrc')):
        compiled_path = compile_resources(qrc_path)
        if compiled_path in _registered_files:
            continue
        if not QResource.registerResource(str(compiled_path)):
            raise ClearMapRuntimeError(f'Qt could not register the resources {compiled_path}; '
                                       f'delete it to compile it again')
        _registered_files.add(compiled_path)
