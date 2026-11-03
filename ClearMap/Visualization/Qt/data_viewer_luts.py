import functools as ft

import numpy as np
import pyqtgraph as pg
from PyQt5 import QtCore
from PyQt5.QtCore import QSize, QObject
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import QGridLayout, QPushButton, QSizePolicy, QWidget

from ClearMap.Visualization.Color import rand_cmap


class LUTItem(pg.HistogramLUTItem):
    """Lookup table item for the DataViewer"""

    def __init__(self, *args, **kwargs):
        pg.HistogramLUTItem.__init__(self, *args, **kwargs)
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

    def __init__(self, parent=None, *args, **kwargs):
        background = kwargs.get('background', 'default')
        pg.GraphicsView.__init__(self, parent=parent, useOpenGL=True, background=background)
        self.item = LUTItem(*args, **kwargs)
        self.setCentralItem(self.item)
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)  # TODO: add option for size_policy
        self.setMinimumWidth(120)

    def sizeHint(self):
        return QSize(120, 200)

    def __getattr__(self, attr):
        return getattr(self.item, attr)


class LUT(QWidget):
    def __init__(self, image=None, color='red', percentiles=[[-100, 0, 50], [50, 75, 100]],
                 parent=None, *args):
        QWidget.__init__(self, parent, *args)

        self.layout = QGridLayout(self)
        self.layout.setSpacing(0)
        # self.layout.setMargin(0)
        self.layout.setContentsMargins(0, 0, 0, 0)

        self.lut = LUTWidget(parent=parent, image=image)
        self.layout.addWidget(self.lut, 0, 0, 1, 1)

        self.range_layout = QGridLayout()
        self.range_buttons = []
        pre = ['%d', '%d']
        for r in range(2):
            range_buttons_m = []
            for i, p in enumerate(percentiles[r]):
                button = QPushButton(pre[r] % (p))
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
        elif color in pg.colormap.listMaps('matplotlib'):
            colormap = pg.colormap.get(color, source='matplotlib')
            self.lut.gradient.setColorMap(colormap)
        elif color == 'random':
            colormap_values = rand_cmap(int(image.image.max() - image.image.min()), map_type='bright', first_color_black=True, last_color_black=False)
            colormap_values = [pg.mkColor(*[int(c*255) for c in col]) for col in colormap_values]
            colormap = pg.ColorMap(None, colormap_values, mapping=pg.ColorMap.CLIP)
            self.lut.gradient.setColorMap(colormap)
        else:
            self.lut.gradient.getTick(0).color = QColor(0, 0, 0, 0)
            self.lut.gradient.getTick(1).color = QColor(color)
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

    def quickPercentile(self, data, percentiles, target_size=1e3):
        while data.size > target_size:
            ax = np.argmax(data.shape)
            sl = [slice(None)] * data.ndim
            sl[ax] = slice(None, None, 2)
            sl = tuple(sl)
            data = data[sl]
        if data.dtype == np.bool_:
            return np.nanpercentile(data.astype(np.uint8), percentiles)
        else:
            return np.nanpercentile(data, percentiles)


class HighLowLUT(QObject):
    """
    Adds a 2nd ImageItem *on top* of an existing one.
    The overlay keeps its own (fixed) 0‑to‑dtype range, so two absolute
    thresholds never move when the user changes the greyscale levels of
    the underlying slice.
    """

    _WHITE = (255, 255, 255, 255)
    _BLACK = (0, 0, 0, 255)
    _TRANSPARENT = (0, 0, 0, 0)
    _EPS = 1e-6

    def __init__(self, *, view_box: pg.ViewBox,
                 base_item: pg.ImageItem,
                 hist_item: pg.HistogramLUTItem,
                 low: float, high: float,
                 dtype_min: int, dtype_max: int,
                 n_pts: str = 'max',
                 low_color: tuple[int] = (0, 0, 255, 255),
                 high_color: tuple[int, int, int, int] = (255, 0, 0, 255)
                 ):
        """
        view_box   : pg.ViewBox that holds the two ImageItems
        base_item  : the greyscale ImageItem already in the viewer
        hist_item  : its HistogramLUTItem (levels widget) — we listen to its
                     region so we repaint when the user changes brightness
        low, high  : absolute thresholds (spin‑boxes)
        dtype_min/max : full representable range, e.g. 0 / 65535 for uint16
        n_pts     : number of colours in the LUT (or 'max' for max size (i.e. 65535 for uint16))
        """
        if n_pts == 'max':
            n_pts = dtype_max - 1
        super().__init__(base_item)
        self.base = base_item
        self.low, self.high = float(low), float(high)
        self.dtype_min, self.dtype_max = float(dtype_min), float(dtype_max)
        self.n_pts = int(n_pts)
        self.low_color = low_color
        self.high_color = high_color

        # ------------------------------------------------------------------
        # second ImageItem that shares the SAME ndarray slice
        # ------------------------------------------------------------------
        self.top = pg.ImageItem(axisOrder="col-major")
        self.top.setCompositionMode(pg.QtGui.QPainter.CompositionMode_Plus)
        view_box.addItem(self.top)

        # The overlay uses exactly the same 2‑D array object the base layer
        # points to, so scrolling & zooming stay in sync automatically.
        self.top.setImage(self.base.image,
                          levels = (self.dtype_min, self.dtype_max),
                          autoLevels=False)
        # ---------------------------------------------------------------
        #  Keep the overlay in sync every time the base slice is updated
        # ---------------------------------------------------------------
        if hasattr(self.base, "sigImageChanged"):        # pg ≥ 0.11
            self.base.sigImageChanged.connect(self._sync_image)
        else:                                            # pg ≤ 0.10
            self.base.imageChanged.connect(self._sync_image)
        # Update whenever thresholds or brightness window changes
        hist_item.region.sigRegionChanged.connect(self._rebuild)
        self._rebuild()

    def _sync_image(self, *_) -> None:
        """Copy the *current* ndarray of the greyscale layer into the
        overlay so colours always refer to the right slice."""
        self.top.setImage(self.base.image,
                          levels=(self.dtype_min, self.dtype_max),
                          autoLevels=False)

    # ---------------- attach spin‑boxes here ------------------------------
    def set_low (self, v):  self.low  = float(v); self._rebuild()
    def set_high(self, v):  self.high = float(v); self._rebuild()

    # ---------------------------------------------------------------------
    def _rebuild(self):
        lo, hi = sorted((self.low, self.high))
        rng    = self.dtype_max - self.dtype_min

        # positions in 0..1 of absolute thresholds (fixed range!)
        p_lo = (lo - self.dtype_min) / rng
        p_hi = (hi - self.dtype_min) / rng
        p_lo = np.clip(p_lo, 0., 1.);  p_hi = np.clip(p_hi, 0., 1.)
        if p_hi - p_lo < 1e-6:
            p_hi = min(p_lo + 1e-6, 1.0)

        pos  = [0.0, p_lo, p_lo + self._EPS,
                      p_hi - self._EPS, p_hi, 1.0]
        cols = [self.low_color, self.low_color,
                self._BLACK, self._WHITE,
                # self._TRANSPARENT, self._TRANSPARENT,
                self.high_color, self.high_color]

        cmap = pg.ColorMap(pos, cols, mode='byte')
        lut  = cmap.getLookupTable(0.0, 1.0, self.n_pts, alpha=True)

        self.top.setLookupTable(lut, update=True)
