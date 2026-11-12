"""
The rich-text tooltips of the designer files mark warnings and notes with bold or italic, not with a fixed colour:
the tooltip background follows the theme (dark with qdarkstyle, the desktop's with the system theme), and a yellow
or light blue text is unreadable on a light one.
"""
import html
import re
from pathlib import Path

import pytest

import ClearMap.gui

DESIGNER_FILES = sorted((Path(ClearMap.gui.__file__).parent / 'creator').glob('*.ui'))
TOOLTIP_PATTERN = re.compile(r'<property name="(?:toolTip|whatsThis|statusTip)">\s*<string[^>]*>(.*?)</string>', re.S)
TEXT_COLOUR_PATTERN = re.compile(r'(?<![-\w])color\s*:\s*[^;"]+')


@pytest.mark.parametrize('designer_file', DESIGNER_FILES, ids=lambda path: path.name)
def test_the_tooltips_of_the_designer_file_set_no_text_colour(designer_file):
    tooltips = [html.unescape(text) for text in TOOLTIP_PATTERN.findall(designer_file.read_text(encoding='utf-8'))]

    assert [match for tooltip in tooltips for match in TEXT_COLOUR_PATTERN.findall(tooltip)] == []
