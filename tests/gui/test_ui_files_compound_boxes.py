"""
The compound boxes of the designer files (singlets, doublets, triplets) are QFrames promoted to NumberListBox, which
the file declares as a custom widget of ClearMap.gui.compound_boxes, with as many spin boxes as their name says, which
NumberListBox can order. Read from the XML: building them into widgets is checked by
test_designer_files_build_with_the_qt_binding.
"""
import importlib
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

import ClearMap.gui

CREATOR_FOLDER = Path(ClearMap.gui.__file__).parent / 'creator'
UI_FILES = sorted(CREATOR_FOLDER.glob('*.ui'))
BOX_CLASS = 'NumberListBox'
BOX_MODULE = 'ClearMap.gui.compound_boxes'
BOX_NAME = re.compile(r'(Singlet|Doublet|Triplet)$')
N_SPIN_BOXES = {'Singlet': 1, 'Doublet': 2, 'Triplet': 3}
SPIN_BOX_CLASSES = ('QSpinBox', 'QDoubleSpinBox')
ORDERED_SPIN_BOX_NAME = re.compile(r'_\d+$')


def _boxes(ui_file):
    """The widgets whose name says they are a singlet, a doublet or a triplet"""
    return [widget for widget in ET.parse(ui_file).getroot().iter('widget') if BOX_NAME.search(widget.get('name'))]


def _spin_boxes(box):
    return [widget.get('name') for widget in box.iter('widget') if widget.get('class') in SPIN_BOX_CLASSES]


def _all_boxes():
    return [(ui_file.name, box) for ui_file in UI_FILES for box in _boxes(ui_file)]


def _file_name(ui_file):
    return ui_file.name


def _id(file_and_box):
    return f'{file_and_box[0]}:{file_and_box[1].get("name")}'


def test_the_designer_files_have_list_boxes():
    assert len(_all_boxes()) >= 38


@pytest.mark.parametrize('file_and_box', _all_boxes(), ids=_id)
def test_a_list_box_is_a_promoted_number_list_box(file_and_box):
    assert file_and_box[1].get('class') == BOX_CLASS


@pytest.mark.parametrize('file_and_box', _all_boxes(), ids=_id)
def test_a_list_box_has_as_many_spin_boxes_as_its_name_says(file_and_box):
    _, box = file_and_box
    kind = BOX_NAME.search(box.get('name')).group(1)

    assert len(_spin_boxes(box)) == N_SPIN_BOXES[kind]


@pytest.mark.parametrize('file_and_box', _all_boxes(), ids=_id)
def test_the_order_of_the_spin_boxes_of_a_list_box_can_be_told_from_their_names(file_and_box):
    _, box = file_and_box
    names = _spin_boxes(box)
    min_max = len(names) == 2 and {name.lower()[-3:] for name in names} == {'min', 'max'}

    assert min_max or all(ORDERED_SPIN_BOX_NAME.search(name) for name in names), names


@pytest.mark.parametrize('ui_file', [ui_file for ui_file in UI_FILES if _boxes(ui_file)], ids=_file_name)
def test_a_file_with_list_boxes_declares_the_promoted_class(ui_file):
    declared = [(widget.findtext('class'), widget.findtext('extends'), widget.findtext('header'))
                for widget in ET.parse(ui_file).getroot().iter('customwidget')]

    assert (BOX_CLASS, 'QFrame', BOX_MODULE) in declared


def test_the_declared_module_provides_the_class():
    assert hasattr(importlib.import_module(BOX_MODULE), BOX_CLASS)


def test_no_designer_file_has_a_plain_frame_named_like_a_list_box():
    assert not [(file, box.get('name')) for file, box in _all_boxes() if box.get('class') == 'QFrame']
