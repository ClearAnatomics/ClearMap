# -*- coding: utf-8 -*-
"""
Utils
=====

Some utility functions to arrang windows and screens.
"""
__author__ = 'Christoph Kirst <christoph.kirst.ck@gmail.com>'
__license__ = 'GPLv3 - GNU General Pulic License v3 (see LICENSE)'
__copyright__ = 'Copyright © 2020 by Christoph Kirst'
__webpage__ = 'https://idisco.info'
__download__ = 'https://www.github.com/ChristophKirst/ClearMap2'


import numpy as np

from PyQt5 import QtGui
from PyQt5.QtCore import QRectF
import pyqtgraph as pg


def screen_geometry(screen=None):
    """Return the geometry of the current screen.

    Arguments
    ---------
    screen : int or None
       The screen for which to return the geometry.

    Returns
    -------
    geometry : tuple
        (Left, Top, width, Height)
    """
    if screen is None:
        screen = -1
    if not pg.QAPP:
       pg.mkQApp()

    g = pg.QAPP.screens()[screen].geometry()
    return g.left(), g.top(), g.width(), g.height()


def tiled_layout(n_windows, origin=None, shape=None, percent=None, screen=None):
    """Generate tiled geometry for windows on a screen.

    Arguments
    ---------
    n_windows : int
       Number number of windows.
    origin : tuple or None
       Optional lower left corner.
    shape : tuple or None
       The shape for all windows. If None use full screen.
    percent : float or None
       Percentage of the given shape o use.
    screen : int or None
        The screen for which to return the geometry.

    Returns
    -------
    geometry : list of tuples
        [(Left, Top, width, Height), ...] for each window.
    """

    if origin is None or shape is None:
       geometry = screen_geometry()
    if origin is None:
       origin = geometry[:2]
    if shape is None:
        shape = geometry[2:]
    width, height = shape
    x0, y0 = origin

    if percent is not None:
        width = int(width / 100.0 * percent)
        height = int(height / 100.0 * percent)

    if n_windows <= 3:
        nx = n_windows
        ny = 1
    else:
        nx = int(np.ceil(np.sqrt(n_windows)))
        ny = int(np.ceil(n_windows*1.0/nx))

    x = np.array(np.linspace(0, width, nx+1), dtype=int)
    y = np.array(np.linspace(0, height, ny+1), dtype=int)

    geo = []
    ix = 0
    iy = 0
    for i in range(n_windows):
        geo.append([x[ix] + x0, y[iy] + y0, x[ix+1]-x[ix], y[iy+1]-y[iy]])
        ix += 1
        if ix == nx:
           ix = 0
           iy += 1

    return geo


class RedCross(pg.GraphicsObject):  # Here to avoid circular imports
    def __init__(self, coords=None, size=[10, 10], parent=None):
        super().__init__(parent)
        self.color = "#FF00007d"
        if coords is None:
            coords = [0, 0]
        self.size = size
        self._rect = QRectF(coords[0]-self.size[0]/2, coords[1]-self.size[1]/2, self.size[0], self.size[1])  # TODO set as fraction of image size
        self.coords = coords
        self.picture = QtGui.QPicture()

    def set_coords(self, coords):
        x, y = coords
        self.coords = coords
        self._rect.setCoords(x-self.size[0]/2, y-self.size[1]/2,
                             x+self.size[0]/2, y+self.size[1]/2)
        self._generate_picture()

    def _generate_picture(self):
        painter = QtGui.QPainter(self.picture)
        pen = pg.mkPen(self.color)
        pen.setWidth(4)
        painter.setPen(pen)
        painter.setBrush(pg.mkBrush(self.color))
        painter.drawLine(self._rect.topLeft(), self._rect.bottomRight())
        painter.drawLine(self._rect.bottomLeft(), self._rect.topRight())
        painter.drawPicture(0, 0, self.picture)
        painter.end()

    def paint(self, painter, option, widget=None):
        painter.drawPicture(0, 0, self.picture)

    def boundingRect(self):  # TODO: check if we need this method
        return QRectF(self.picture.boundingRect())


def link_dataviewers_cursors(dvs):
    for i, dv in enumerate(dvs):
        cursor = RedCross()
        dv.view.addItem(cursor)
        dv.cross = cursor
        pals = dvs.copy()
        pals.pop(i)
        dv.pals = pals

  
############################################################################################################
# ##  Tests
############################################################################################################


def _test():
    import ClearMap.Visualization.Qt.utils as guiu
    from importlib import reload
    reload(guiu)

    print(guiu.screen_geometry())

    w, h = guiu.screen_geometry()

    print(guiu.tiled_layout(3))
