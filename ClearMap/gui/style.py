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
DARK_BACKGROUND = '#282D2F'
VERY_LIGHT_BACKGROUND = '#727A7E'
PLOT_3D_BG = '#1A1D1E'
BTN_STYLE_SHEET = \
    'QPushButton {'\
    'background-color: #455364; '\
    'color: #E0E1E3;'\
    'border-radius: 4px;'\
    'padding: 2px;'\
    '}'\
    'QPushButton:pressed {'\
    'background-color: #259AE9; }'
# 'QPushButton {'
# 'outline: none;'
# 'QPushButton:pressed {'\
# 'background-color: #60798B; '
# 'border: 2px #259AE9;'

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
