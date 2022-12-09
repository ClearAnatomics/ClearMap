import functools as ft

import numpy as np
import pyqtgraph as pg


class LUTItem(pg.HistogramLUTItem):
    """Lookup table item for the DataViewer"""

    def __init__(self, *args, **kargs):
        pg.HistogramLUTItem.__init__(self, *args, **kargs)
        self.vb.setMaximumWidth(15)
        self.vb.setMinimumWidth(10)

    def imageChanged(self, autoLevel=False, autoRange=False):
        if autoLevel:
            mn, mx = self.quickMinMax(targetSize=500)
            self.region.setRegion([mn, mx])

    def quickMinMax(self, targetSize=1e3):
        """
        Estimate the min/max values of the image data by subsampling.
        """
        data = self.imageItem().image
        while data.size > targetSize:
            ax = np.argmax(data.shape)
            sl = [slice(None)] * data.ndim
            sl[ax] = slice(None, None, 2)
            data = data[tuple(sl)]
        return np.nanmin(data), np.nanmax(data)


class LUTWidget(pg.GraphicsView):
    """Lookup table widget for the DataViewer"""

    def __init__(self, parent=None, *args, **kargs):
        background = kargs.get('background', 'default')
        pg.GraphicsView.__init__(self, parent=parent, useOpenGL=True, background=background)
        self.item = LUTItem(*args, **kargs)
        self.setCentralItem(self.item)
        self.setSizePolicy(pg.QtGui.QSizePolicy.Preferred, pg.QtGui.QSizePolicy.Expanding)  # TODO: add option for sizepolicy
        self.setMinimumWidth(120)

    def sizeHint(self):
        return pg.QtCore.QSize(120, 200)

    def __getattr__(self, attr):
        return getattr(self.item, attr)


class LUT(pg.QtGui.QWidget):
    def __init__(self, image=None, color='red', percentiles=[[-100, 0, 50], [50, 75, 100]],
                 parent=None, *args):
        pg.QtGui.QWidget.__init__(self, parent, *args)

        self.layout = pg.QtGui.QGridLayout(self)
        self.layout.setSpacing(0)
        self.layout.setMargin(0)

        self.lut = LUTWidget(parent=parent, image=image)
        self.layout.addWidget(self.lut, 0, 0, 1, 1)

        self.range_layout = pg.QtGui.QGridLayout()
        self.range_buttons = []
        pre = ['%d', '%d']
        for r in range(2):
            range_buttons_m = []
            for i, p in enumerate(percentiles[r]):
                button = pg.QtGui.QPushButton(pre[r] % (p))
                button.setMaximumWidth(30)
                font = button.font()
                font.setPointSize(6)
                button.setFont(font)
                self.range_layout.addWidget(button, r, i)
                range_buttons_m.append(button)
            self.range_buttons.append(range_buttons_m)

        self.layout.addLayout(self.range_layout, 1, 0, 1, 1)

        self.precentiles = percentiles
        self.percentile_id = [2, 2]

        for m, ab in enumerate(self.range_buttons):
            for p, abm in enumerate(ab):
                abm.clicked.connect(ft.partial(self.updateRegionRange, m, p))

        # default gradient
        if color in pg.graphicsItems.GradientEditorItem.Gradients.keys():
            self.lut.gradient.loadPreset(color)
        else:
            self.lut.gradient.getTick(0).color = pg.QtGui.QColor(0, 0, 0, 0)
            self.lut.gradient.getTick(1).color = pg.QtGui.QColor(color)
            self.lut.gradient.updateGradient()

    def updateRegionRange(self, m, p):
        self.percentile_id[m] = p
        p_min = self.precentiles[0][self.percentile_id[0]]
        p_max = self.precentiles[1][self.percentile_id[1]]
        self.updateRegionPercentile(p_min, p_max)

    def updateRegionPercentile(self, pmin, pmax):
        iitem = self.lut.imageItem()
        if iitem is not None:
            pmax1 = max(0, min(pmax, 100))

            if pmin < 0:
                pmin1 = min(-pmin, 100)
            else:
                pmin1 = min(pmin, 100)

            if pmax1 == 0:
                pmax1 = 1; pmax = 1
            if pmin1 == 0:
                pmin1 = 1; pmin = 1

            r = [float(pmin)/pmin1, float(pmax)/pmax1] * self.quickPercentile(iitem.image, [pmin1, pmax1])
            self.lut.region.setRegion(r)

    def quickPercentile(self, data, percentiles, targetSize=1e3):
        while data.size > targetSize:
            ax = np.argmax(data.shape)
            sl = [slice(None)] * data.ndim
            sl[ax] = slice(None, None, 2)
            sl = tuple(sl)
            data = data[sl]
        if data.dtype == np.bool_:
            return np.nanpercentile(data.astype(np.uint8), percentiles)
        else:
            return np.nanpercentile(data, percentiles)
