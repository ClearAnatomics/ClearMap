"""
The combo box of the preferences offers the themes of gui.style.THEMES, in that order: the names are written once.
"""
import xml.etree.ElementTree as ET
from pathlib import Path

import ClearMap.gui
from ClearMap.gui import style

PREFERENCES_UI = Path(ClearMap.gui.__file__).parent / 'creator' / 'preferences_editor.ui'


def test_the_preferences_offer_the_themes_of_the_style_module():
    combo_box = next(widget for widget in ET.parse(PREFERENCES_UI).getroot().iter('widget')
                     if widget.get('name') == 'themeComboBox')
    combo_box_items = [item.findtext('property/string') for item in combo_box.findall('item')]

    assert tuple(combo_box_items) == style.THEMES
