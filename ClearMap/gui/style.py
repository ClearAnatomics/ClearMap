# -*- coding: utf-8 -*-
"""
style
=====

A set of constants for the style of the widgets in the UI (to customise qdarkstyle)
"""

__author__ = 'Charly Rousseau <charly.rousseau@icm-institute.org>'
__license__ = 'GPLv3 - GNU General Public License v3 (see LICENSE.txt)'
__copyright__ = 'Copyright © 2022 by Charly Rousseau'
__webpage__ = 'https://idisco.info'
__download__ = 'https://www.github.com/ChristophKirst/ClearMap2'

from qdarkstyle import DarkPalette

QDARKSTYLE_BACKGROUND = '#2E3436'
WARNING_YELLOW = '#FADF11'
DARK_BACKGROUND = '#282D2F'
VERY_LIGHT_BACKGROUND = '#727A7E'
PLOT_3D_BG = '#1A1D1E'
BTN_STYLE_SHEET = f"""
QPushButton {{
    background-color: {DarkPalette.COLOR_BACKGROUND_5};
    color: white;
    border: 2px white solid;
    border-radius: 4px;
    padding: 2px;
}}"""

TOOLTIP_STYLE_SHEET = f"""
QToolTip {{
    background-color: {DarkPalette.COLOR_BACKGROUND_2};
    color: {DarkPalette.COLOR_TEXT_2};
    border: 2px white solid;
    border-width: 2px;
    padding: 2px;
    border-radius: 3px;
    opacity: 200;
}}"""

# QComboBox::down-arrow {{
#     image: url(/usr/share/icons/crystalsvg/16x16/actions/1downarrow.png);
# }}


#
COMBOBOX_STYLE_SHEET = f"""
QComboBox {{
    border: 1px solid {DarkPalette.COLOR_BACKGROUND_5};
    border-radius: 3px;
    padding: 1px 2px 1px 3px;
    min-width: 2em;
}}

/* QComboBox gets the "on" state when the popup is open */
QComboBox:!editable:on, QComboBox::drop-down:editable:on {{
    background: {DarkPalette.COLOR_BACKGROUND_3};
}}

QComboBox:editable {{
    background: {DarkPalette.COLOR_BACKGROUND_3};
}}

QComboBox:!editable, QComboBox::drop-down:editable {{
     background: {DarkPalette.COLOR_BACKGROUND_3};
}}

QComboBox:on {{ /* shift the text when the popup opens */
     padding-top: 3px;
     padding-left: 4px;
}}

QComboBox::down-arrow:on {{ /* shift the arrow when popup is open */
    top: 1px;
    left: 1px;
}}

QComboBox::drop-down {{
    subcontrol-origin: padding;
    subcontrol-position: top right;
    width: 15px;

    border-left-width: 1px;
    border-left-color: darkgray;
    border-left-style: solid; /* just a single line */
    border-top-right-radius: 3px; /* same radius as the QComboBox */
    border-bottom-right-radius: 3px;
}}

QComboBox QAbstractItemView {{
    border: 2px solid {DarkPalette.COLOR_ACCENT_2};
    selection-background-color: pink;
}}
"""
