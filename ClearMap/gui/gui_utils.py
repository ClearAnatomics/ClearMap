import os
from math import sqrt, ceil

import lxml.html
import matplotlib
import numpy as np
import skimage.io
from PyQt5 import QtGui
from PyQt5.QtWidgets import QApplication
from matplotlib.colors import cnames

from ClearMap.gui.widgets import RedCross

QDARKSTYLE_BACKGROUND = '#2E3436'
DARK_BACKGROUND = '#282D2F'

GRAY_COLOR_TABLE = []
BLUE_COLOR_TABLE = []
cmap = matplotlib.cm.get_cmap('GnBu')
for i in range(256):
    col = cmap(i / 256)
    col = [int(c*256) for c in col[:-1]]
    GRAY_COLOR_TABLE.append(QtGui.qRgb(i, i, i))
    BLUE_COLOR_TABLE.append(QtGui.qRgb(*col))


class Printer(object):
    def __init__(self, text_widget, color=None, app=None):
        self.widget = text_widget
        self.color = color
        self.win = self.widget.window()
        if app is None:
            self.app = QApplication.instance()

    def write(self, msg):
        self.widget.append(self._colourise(msg))

    def flush(self):  # To be compatible with ClearMap.ParallelProcessing.ProcessWriting
        pass
        # self.app.processEvents()

    def _colourise(self, msg):
        if self.color is not None:
            try:
                html = lxml.html.fromstring(msg)
                is_html = html.find('.//*') is not None
            except lxml.etree.ParserError:
                is_html = False
            if not is_html:
                colour_msg = '<p style="color:{}">{}</p>'.format(cnames[self.color], msg)
            else:
                colour_msg = msg
        else:
            colour_msg = msg
        return colour_msg


def np_to_qpixmap(img_array, alpha, color_table=GRAY_COLOR_TABLE):
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


def clean_path(path):
    return os.path.normpath(os.path.expanduser(path))


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


def runs_from_pycharm():
    return "PYCHARM_HOSTED" in os.environ


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
    for dv in dvs:
        cross = RedCross()
        dv.view.addItem(cross)
        dv.cross = cross
    dvs[0].pal = dvs[1]  # FIXME: pals is all others (to work with != 2 dvs)
    dvs[1].pal = dvs[0]
