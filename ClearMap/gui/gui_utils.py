import os
from math import sqrt, ceil

import matplotlib
import numpy as np
import skimage.io
from PyQt5 import QtGui

from ClearMap.gui.widgets import RedCross

QDARKSTYLE_BACKGROUND = '#2E3436'
DARK_BACKGROUND = '#282D2F'


def np_to_qpixmap(img_array, alpha):
    img_array = img_array.copy().astype(np.float64)
    img_array -= img_array.min()  # normalise 0-1
    img_array /= img_array.max()
    # img = np.uint8(matplotlib.cm.Blues_r(img_array)*255)
    img = np.uint8(matplotlib.cm.copper(img_array)*255)

    if alpha.dtype == np.bool:
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
    proj = project(img, 2, invalid_val=0.0)  # FIXME: check axis=0
    proj -= np.min(proj[np.nonzero(proj)])
    proj[proj < 0] = np.nan  # or -np.inf
    mask = ~np.isnan(proj).T  # TODO: check .T
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


def link_dataviewers_cursors(dvs):  # TODO: move to DataViewer module
    for i, dv in enumerate(dvs):
        cross = RedCross()
        dv.view.addItem(cross)
        dv.cross = cross
        pals = dvs.copy()
        pals.pop(i)
        dv.pals = pals


class TmpDebug(object):
    def __init__(self, workspace):
        self.workspace = workspace

    def __enter__(self):
        self.workspace.debug = True
        return self.workspace

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.workspace.debug = False
