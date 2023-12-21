# -*- coding: utf-8 -*-
"""
gui_utils
=========

Various utility functions specific to the graphical interface
"""

import inspect
import os
from math import sqrt, ceil

import warnings

import matplotlib
import numpy as np
import skimage.io
from PyQt5 import QtGui
from PyQt5.QtGui import QColor
from matplotlib.colors import hsv_to_rgb

from ClearMap.gui.pyuic_utils import loadUiType

warnings.filterwarnings('ignore', category=RuntimeWarning, module='ClearMap.gui.gui_utils')  # For surface_project

__author__ = 'Charly Rousseau <charly.rousseau@icm-institute.org>'
__license__ = 'GPLv3 - GNU General Public License v3 (see LICENSE.txt)'
__copyright__ = 'Copyright © 2022 by Charly Rousseau'
__webpage__ = 'https://idisco.info'
__download__ = 'https://www.github.com/ChristophKirst/ClearMap2'


UI_FOLDER = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))


def np_to_qpixmap(img_array, alpha):
    img_array = img_array.copy().astype(np.float64)
    img_array -= img_array.min()  # normalise 0-1
    img_array /= img_array.max()
    # img = np.uint8(matplotlib.cm.Blues_r(img_array)*255)
    img = np.uint8(matplotlib.cm.copper(img_array)*255)

    if alpha.dtype == bool:
        alpha = alpha.astype(np.uint8)
        alpha[alpha != 0] = 255  # reestablish full range

    r = img[:, :, 0]
    g = img[:, :, 1]
    b = img[:, :, 2]
    img = np.dstack((r, g, b, alpha))

    tmp_path = '/tmp/test.png'  # OPTIMISE: use tempfile
    skimage.io.imsave(tmp_path, img)
    img = QtGui.QPixmap(tmp_path)
    os.remove(tmp_path)
    return img


# def __np_to_qpixmap(img_array, fmt=QtGui.QImage.Format_Indexed8):
#     img = QtGui.QImage(
#         img_array.data.tobytes(),
#         img_array.shape[0],
#         img_array.shape[1],
#         img_array.strides[0],
#         fmt
#     )
#     return QtGui.QPixmap.fromImage(img)


def html_to_ansi(msg):  #  WARNING: Will not work correctly with colours
    codes_dict = {
        '<nobr>': '',
        '</nobr>': '',
        '<br>': '\n',
        '</em>': '\033[0m',
        '<em>': '\033[3m'
    }
    for k, v in codes_dict.items():
        msg = msg.replace(k, v)
    return msg


def html_to_plain_text(msg):
    codes_dict = {
        '<nobr>': '',
        '</nobr>': '',
        '<br>': '\n',
        '</em>': '',
        '<em>': ''
    }
    for k, v in codes_dict.items():
        msg = msg.replace(k, v)
    return msg


def compute_grid(nb):
    sqr = round(sqrt(nb))
    col = ceil(nb / sqr)
    row = ceil(nb / col)
    return row, col


def project(img, axis, invalid_val=np.nan):
    mask = img != 0
    return np.where(mask.any(axis=axis), mask.argmax(axis=axis), invalid_val)


def surface_project(img):
    proj = project(img, 2, invalid_val=0.0)
    proj -= np.min(proj[np.nonzero(proj)])
    proj[proj < 0] = np.nan  # or -np.inf
    mask = ~np.isnan(proj).T
    proj *= 255.0 / np.nanmax(proj)
    proj = 255 - proj  # invert
    proj = proj.astype(np.uint8).T
    return mask, proj


def format_long_nb_to_str(nb):
    out = ''
    for idx, c in enumerate(str(nb)[::-1]):
        tick = "'" if (idx and not idx % 3) else ""
        out += tick+c
    out = out[::-1]
    return out


class TmpDebug(object):  # FIXME: move as part of workspace
    def __init__(self, workspace):
        self.workspace = workspace

    def __enter__(self):
        self.workspace.debug = True
        return self.workspace

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.workspace.debug = False


def get_current_res(app):
    screen = app.primaryScreen()
    size = np.array((screen.size().width(), screen.size().height()))

    hd_res = np.array((1920, 1080))
    four_k_res = np.array((3840, 2160))

    if (size < hd_res).any():
        res = 'sd'
    elif (size < four_k_res).any():
        res = 'hd'
    else:
        res = '4k'
    return res


def create_clearmap_widget(ui_name, patch_parent_class, window_title=None):
    widget_class, _ = loadUiType(os.path.join(UI_FOLDER, 'creator', ui_name), from_imports=True,
                                 import_from='ClearMap.gui.creator', patch_parent_class=patch_parent_class)
    widget = widget_class()
    if window_title is not None:
        widget.setWindowTitle(window_title)
    widget.setupUi()
    return widget


def pseudo_random_rgb_array(n_samples):
    if n_samples == 0:
        return None
    hues = np.random.rand(n_samples)
    saturations = np.random.rand(n_samples) / 2 + 0.5
    values = np.random.rand(n_samples) / 2 + 0.5
    hsvs = np.vstack((hues, saturations, values))
    rgbs = np.apply_along_axis(hsv_to_rgb, 0, hsvs)
    return rgbs.T


def get_random_color():
    return QColor(*np.random.randint(0, 255, 3))


def get_pseudo_random_color():
    """
    Return a pseudo random colour. The hue is random but
    The saturation and the value are kept in the upper half interval

    Returns
    -------

    """
    rand_gen = np.random.default_rng()
    hsv = (rand_gen.uniform(0, 1),
           rand_gen.uniform(0.5, 1),
           rand_gen.uniform(0.5, 1))
    rgb = hsv_to_rgb(hsv)
    return rgb


def is_dark(color):
    """

    Parameters
    ----------
    color QColor

    Returns
    -------


    """
    return color.getHsl()[2] < 128


def clear_layout(layout):
    """
    Clears all widgets in the layout

    Parameters
    ----------
    layout

    Returns
    -------

    """
    for i in range(layout.count(), -1, -1):
        item = layout.takeAt(i)
        if item is not None:
            widg = item.widget()
            widg.setParent(None)
            widg.deleteLater()
