"""
The images of the GUI built in Qt resources (.qrc, compiled by pyrcc5 at boot) exist, and every resource path of
the designer files (':/<prefix>/<file>') is one of them (#135: the figure of the rigid stitcher help was missing,
and referenced at a path no resource had; pyrcc5 stops on a missing file).
"""
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

import ClearMap.gui

CREATOR_FOLDER = Path(ClearMap.gui.__file__).parent / 'creator'
QRC_FILES = sorted(CREATOR_FOLDER.glob('*.qrc'))


def _resources():
    """{':/<prefix>/<file>': path on disk} of all the qrc files"""
    resources = {}
    for qrc in QRC_FILES:
        for qresource in ET.parse(qrc).getroot().iter('qresource'):
            prefix = qresource.get('prefix', '/').strip('/')
            for file in qresource.iter('file'):
                resource_path = ':/' + '/'.join(part for part in (prefix, file.text.strip()) if part)
                resources[resource_path] = qrc.parent / file.text.strip()
    return resources


@pytest.mark.parametrize('resource_path, file_path', sorted(_resources().items()))
def test_every_file_of_the_resources_exists(resource_path, file_path):
    assert file_path.is_file(), f'{resource_path}: {file_path} is missing'


def test_every_resource_of_the_designer_files_is_in_the_resources():
    resources = _resources()
    used = {(ui.name, path) for ui in CREATOR_FOLDER.glob('*.ui')
            for path in re.findall(r'(?<![A-Za-z]):/[^"<&;\s]+', ui.read_text(encoding='utf-8'))}

    assert sorted((ui, path) for ui, path in used if path not in resources) == []
