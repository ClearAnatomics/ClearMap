# -*- coding: utf-8 -*-
"""
Color
=====

This sub-package provides tools for creating colors and color maps.
""" 
__author__ = 'Christoph Kirst <christoph.kirst.ck@gmail.com>'
__license__ = 'GPLv3 - GNU General Public License v3 (see LICENSE)'
__copyright__ = 'Copyright © 2023 by Christoph Kirst'


from .Color import color, hex_to_rgb, rgb_to_hex, colormap, orientation_to_boys, orientation_to_rgb, colormaps, \
    lighter, darker, rgb_LUT, rgb_to_LUT, write_LUT, write_PAL, rand_cmap

from .orientation_colormap import orientation_color, orientation_color_cached
