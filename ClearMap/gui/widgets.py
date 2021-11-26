import numpy as np
import pyqtgraph as pg
from PyQt5 import QtGui
from PyQt5.QtCore import QRectF
from ClearMap.Visualization import Plot3d as plot_3d


class RectItem(pg.GraphicsObject):  # Derived from https://stackoverflow.com/a/60012800
    def __init__(self, rect, parent=None):
        super().__init__(parent)
        self._rect = rect
        self.picture = QtGui.QPicture()
        self._generate_picture()
        self.name = 'rect'

    def __str__(self):
        return 'Rect {}, coordinates: {}'.format(self.name, self.rect.getCoords())

    @property
    def rect(self):
        return self._rect

    # TODO: setWidth, setLeft ... that call self.rect.setWidth + self._generate_picture

    def _generate_picture(self):
        painter = QtGui.QPainter(self.picture)
        painter.setPen(pg.mkPen("#FFFF007d"))
        painter.setBrush(pg.mkBrush("#2e34367d"))
        painter.drawRect(self.rect)
        painter.end()

    def paint(self, painter, option, widget=None):
        painter.drawPicture(0, 0, self.picture)

    def boundingRect(self):
        return QRectF(self.picture.boundingRect())


class RedCross(pg.GraphicsObject):
    def __init__(self, coords=None, size=[10, 10], parent=None):
        super().__init__(parent)
        self.color = "#FF00007d"
        if coords is None:
            coords = [0, 0]
        self.size = size
        self._rect = QRectF(coords[0]-self.size[0]/2, coords[1]-self.size[1]/2, self.size[0], self.size[1])  # TODO@ set as fraction of image size
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
        painter.setPen(pg.mkPen(self.color))
        painter.setBrush(pg.mkBrush(self.color))
        painter.drawLine(self._rect.topLeft(), self._rect.bottomRight())
        painter.drawLine(self._rect.bottomLeft(), self._rect.topRight())
        painter.drawPicture(0, 0, self.picture)
        painter.end()

    def paint(self, painter, option, widget=None):
        painter.drawPicture(0, 0, self.picture)

    def boundingRect(self):  # TODO: check if we need this method
        return QRectF(self.picture.boundingRect())


class OrthoViewer(object):
    def __init__(self, img=None, parent=None):
        self.img = img
        self.parent = parent
        if img is not None:
            self.shape = img.shape
        else:
            self.shape = None
        self.rectangles = []

    def setup(self, img, parent=None):
        self.img = img
        self.parent = parent
        self.shape = img.shape
        self.rectangles = []

    @property
    def width(self):
        return self.shape[0]

    @property
    def height(self):
        return self.shape[1]

    @property
    def depth(self):
        return self.shape[2]

    def update_x_min(self, val):
        self._update_rect('x', val, 'min')

    def update_x_max(self, val):
        self._update_rect('x', val, 'max')

    def update_y_min(self, val):
        self._update_rect('y', val, 'min')

    def update_y_max(self, val):
        self._update_rect('y', val, 'max')

    def update_z_min(self, val):
        self._update_rect('z', val, 'min')

    def update_z_max(self, val):
        self._update_rect('z', val, 'max')

    def update_ranges(self, param):
        self.update_x_min(param.crop_x_min)
        self.update_x_max(param.crop_x_max)
        self.update_y_min(param.crop_y_min)
        self.update_y_max(param.crop_y_max)
        self.update_z_min(param.crop_z_min)
        self.update_z_max(param.crop_z_max)

    def get_rect(self, axis, min_or_max):
        axes = ('x', 'y', 'z')
        if axis in axes:
            axis = axes.index(axis)
        idx = axis * 2 + (min_or_max == 'max')
        return self.rectangles[idx]

    def _update_rect(self, axis, val, min_or_max='min'):
        if not self.rectangles:
            return
        rect_itm = self.get_rect(axis, min_or_max)
        if min_or_max == 'min':
            rect_itm.rect.setWidth(val)
        else:
            rect_itm.rect.setLeft(val)
        rect_itm._generate_picture()
        graph = getattr(self.parent, self.parent.graph_names[axis])  # REFACTOR: not the cleanest
        graph.view.update()

    def add_cropping_bars(self):
        self.rectangles = []
        matched_axes = (1, 0, 1)  # TODO: compute
        for i, dv in enumerate(self.dvs):
            min_rect = RectItem(QRectF(0, 0, 0, self.shape[matched_axes[i]]))
            self.rectangles.append(min_rect)
            dv.view.addItem(min_rect)
            max_rect = RectItem(QRectF(self.shape[i], 0, 0, self.shape[matched_axes[i]]))
            self.rectangles.append(max_rect)
            dv.view.addItem(max_rect)

    def plot_orthogonal_views(self, img=None, parent=None):
        if img is None:
            img = self.img.array
        if parent is None:
            parent = self.parent
            if parent is None:
                raise ValueError('Parent not set')
        x = np.copy(img)
        y = np.copy(img).swapaxes(0, 1)
        z = np.copy(img).swapaxes(0, 2)
        dvs = plot_3d.plot([x, y, z], arange=False, lut='white', parent=parent, sync=False)
        self.dvs = dvs
        return dvs
