"""
The images of the GUI built in Qt resources (.qrc, compiled by rcc at boot) exist, and every resource path of
the designer files (':/<prefix>/<file>') is one of them (#135: the figure of the rigid stitcher help was missing,
and referenced at a path no resource had; rcc stops on a missing file).
"""
import re
from pathlib import Path

import pytest

import ClearMap.gui
from ClearMap.gui.qt_resources import resource_files

CREATOR_FOLDER = Path(ClearMap.gui.__file__).parent / 'creator'
QRC_FILES = sorted(CREATOR_FOLDER.glob('*.qrc'))


def _resources():
    """{':/<prefix>/<file>': path on disk} of all the qrc files"""
    return {resource_path: file for qrc in QRC_FILES for resource_path, file in resource_files(qrc).items()}


@pytest.mark.parametrize('resource_path, file_path', sorted(_resources().items()))
def test_every_file_of_the_resources_exists(resource_path, file_path):
    assert file_path.is_file(), f'{resource_path}: {file_path} is missing'


def test_every_resource_of_the_designer_files_is_in_the_resources():
    resources = _resources()
    used = {(ui.name, path) for ui in CREATOR_FOLDER.glob('*.ui')
            for path in re.findall(r'(?<![A-Za-z]):/[^"<&;\s]+', ui.read_text(encoding='utf-8'))}

    assert sorted((ui, path) for ui, path in used if path not in resources) == []
