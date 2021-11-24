import pyqtgraph as pg
from PyQt5 import QtGui
from PyQt5.QtCore import QRectF, QPoint
from PyQt5.QtWidgets import QWidget


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
        if coords is None:
            coords = [0, 0]
        self.size = size
        self._rect = QRectF(coords[0]-self.size[0]/2, coords[1]-self.size[1]/2, self.size[0], self.size[1])  # TODO@ set as fraction of image size
        self.coords = coords
        self.picture = QtGui.QPicture()

    def __to_pts(self, angle):
        return angle * 16

    def set_coords(self, coords):
        x, y = coords
        self.coords = coords
        self._rect.setCoords(x-self.size[0]/2, y-self.size[1]/2,
                             x+self.size[0]/2, y+self.size[1]/2)
        # self._rect.setLeft(self.coords[0])
        # self._rect.setTop(self.coords[1])
        self._generate_picture()
        # self.paint()

    def _generate_picture(self):
        painter = QtGui.QPainter(self.picture)
        painter.setPen(pg.mkPen("#FF00007d"))
        painter.setBrush(pg.mkBrush("#FF00007d"))
        painter.drawLine(self._rect.topLeft(), self._rect.bottomRight())
        painter.drawLine(self._rect.bottomLeft(), self._rect.topRight())
        # painter.drawRect(self._rect)
        painter.drawPicture(0, 0, self.picture)
        painter.end()

    def paint(self, painter, option, widget=None):
        painter.drawPicture(0, 0, self.picture)

    def boundingRect(self):  # TODO: check if we need this method
        return QRectF(self.picture.boundingRect())
