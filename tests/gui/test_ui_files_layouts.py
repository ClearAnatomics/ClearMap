"""
The layouts of the .ui files, read as XML (no Qt): what the designer wrote is what Qt lays out, and Qt only warns
("Cell (4, 1) already occupied") when two widgets are put in the same cell, then draws them over each other.
"""
import xml.etree.ElementTree as ElementTree
from pathlib import Path

import pytest

CREATOR_DIR = Path(__file__).resolve().parents[2] / 'ClearMap' / 'gui' / 'creator'
UI_FILES = sorted(CREATOR_DIR.glob('*.ui'))
CELL_LAYOUT_CLASSES = ('QGridLayout', 'QFormLayout')


def find_layout(ui_file_name: str, layout_name: str):
    root = ElementTree.parse(CREATOR_DIR / ui_file_name).getroot()
    layouts = [layout for layout in root.iter('layout') if layout.get('name') == layout_name]
    assert len(layouts) == 1, f'expected one layout named "{layout_name}" in {ui_file_name}, found {len(layouts)}'
    return layouts[0]


def occupied_twice(layout) -> list[str]:
    """The cells of a grid or form layout that more than one widget is put in, with the widgets"""
    owners = {}
    for item in layout.findall('item'):
        row, column = int(item.get('row', 0)), int(item.get('column', 0))
        row_span, column_span = int(item.get('rowspan', 1)), int(item.get('colspan', 1))
        widget_name = item[0].get('name')
        for cell_row in range(row, row + row_span):
            for cell_column in range(column, column + column_span):
                owners.setdefault((cell_row, cell_column), []).append(widget_name)
    return [f'{layout.get("name")} cell {cell}: {", ".join(names)}' for cell, names in owners.items() if len(names) > 1]


@pytest.mark.parametrize('ui_file', UI_FILES, ids=lambda path: path.name)
def test_no_two_widgets_of_a_ui_file_are_put_in_the_same_cell_of_a_grid_or_form_layout(ui_file):
    root = ElementTree.parse(ui_file).getroot()

    overlaps = [overlap for layout in root.iter('layout') if layout.get('class') in CELL_LAYOUT_CLASSES
                for overlap in occupied_twice(layout)]

    assert not overlaps, overlaps
