"""
The loader of the designer files reads them without their ``<resources>`` element: its ``<include location="icons.qrc"/>``
made it import a generated module (``icons_rc``) that ClearMap no longer has, now that the resources are compiled and
registered by gui/qt_resources. Nothing else of the file changes (checked on every designer file of the GUI).
"""
import xml.etree.ElementTree as ET
from collections import Counter
from io import StringIO
from pathlib import Path

import pytest

import ClearMap.gui
from ClearMap.gui.gui_utils_base import ui_without_resource_includes

CREATOR_FOLDER = Path(ClearMap.gui.__file__).parent / 'creator'
UI_FILES = sorted(CREATOR_FOLDER.glob('*.ui'))
RESOURCES_TAGS = ('resources', 'include')


def _elements_other_than_resources(root):
    return Counter((element.tag, tuple(sorted(element.attrib.items())), (element.text or '').strip())
                   for element in root.iter() if element.tag not in RESOURCES_TAGS)


def test_there_are_designer_files_with_resources_to_check():
    assert any(ET.parse(ui_file).getroot().findall('resources') for ui_file in UI_FILES)


@pytest.mark.parametrize('ui_file', UI_FILES, ids=lambda ui_file: ui_file.name)
def test_a_designer_file_loses_its_resources_and_nothing_else(ui_file):
    original = ET.parse(ui_file).getroot()

    read = ET.parse(ui_without_resource_includes(ui_file)).getroot()

    assert read.findall('resources') == [] and not [e for e in read.iter('include')]
    assert _elements_other_than_resources(read) == _elements_other_than_resources(original)


def test_the_file_on_disk_keeps_its_resources_for_designer():
    with_resources = next(ui_file for ui_file in UI_FILES if ET.parse(ui_file).getroot().findall('resources'))

    ui_without_resource_includes(with_resources)

    assert ET.parse(with_resources).getroot().findall('resources')


def test_a_file_like_object_is_read_too():
    ui_text = '<ui version="4.0"><class>W</class><resources><include location="a.qrc"/></resources></ui>'

    assert ui_without_resource_includes(StringIO(ui_text)).read() == '<ui version="4.0"><class>W</class></ui>'
