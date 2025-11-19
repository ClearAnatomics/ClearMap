"""
Plot3d
======

Plotting routines based on qt.

Note
----
This module is based on the pyqtgraph package.
"""
__author__ = 'Christoph Kirst <christoph.kirst.ck@gmail.com>'
__license__ = 'GPLv3 - GNU General Public License v3 (see LICENSE)'
__copyright__ = 'Copyright © 2020 by Christoph Kirst'
__webpage__ = 'https://idisco.info'
__download__ = 'https://github.com/ClearAnatomics/ClearMap'

import itertools
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, List, Sequence, TypeAlias, Union, Any

import numpy as np
import pyqtgraph as pg
import functools as ft

from PyQt5.QtCore import QRect
from PyQt5.QtWidgets import QApplication

from ClearMap.Utils.utilities import runs_on_spyder

import ClearMap.Visualization.Qt.DataViewer as dv
import ClearMap.Visualization.Qt.utils as qtu

############################################################################################################
#  Plotting
############################################################################################################

# TODO: figure / windows handler to update data in existing windows


SourceLike: TypeAlias = Union[str, Path, np.ndarray, "ClearMap.IO.source.Source"]
SourceType: TypeAlias = Union[SourceLike, Sequence[SourceLike], Sequence[Sequence[SourceLike]]]

Lut: TypeAlias = Optional[str]
# Per-panel min/max: either a single (low, high) or a list of these for overlays
MinMax: TypeAlias = Optional[Union[Tuple[float, float], List[Tuple[float, float]]]]
# Per-panel max-projection flags: either a single int or per-overlay list
MaxProj: TypeAlias = Optional[Union[int, List[int]]]


@dataclass
class PlotPanel:
    """
    One window in multi_plot.
    images: Any that Plot3d already accepts:
      Path | str | np.ndarray | Source | List[Any] | List[List[Any]] (overlays)
    """
    images: SourceType
    title: Optional[str] = None
    lut: Lut = None
    min_max: MinMax = None  # potentially nested to match overlays
    max_projection: MaxProj = None


def multi_plot_from_panels(panels: Sequence[PlotPanel], *, axis: int | None = None,
                           scale: Tuple[float, float, float] | None = None,
                           invert_y: bool = True, arrange: bool = True, sync: bool = True,
                           screen: int | None = None, to_front: bool = True, parent=None):
    """
    Thin wrapper that converts a list of Panels into a call to multi_plot().
    (same Path->str normalization as Plot3d.plot()).
    """
    sources: List[SourceType]   = []
    titles:  List[Optional[str]]   = []
    luts:    List[Lut]   = []
    min_maxes:    List[MinMax]   = []
    max_projs:    List[MaxProj]   = []

    def _normalize_paths(obj: Any):
        if isinstance(obj, Path):
            return str(obj)
        if isinstance(obj, (list, tuple)):
            return [ _normalize_paths(x) for x in obj ]
        return obj

    for p in panels:
        sources.append(_normalize_paths(p.images))
        titles.append(p.title)
        luts.append(p.lut)
        min_maxes.append(p.min_max)
        max_projs.append(p.max_projection)

    return multi_plot(sources, axis=axis, scale=scale, title=titles, invert_y=invert_y,
                      min_max=min_maxes, max_projection=max_projs, arrange=arrange, screen=screen,
                      lut=luts, parent=parent, sync=sync, to_front=to_front)


def plot(source, axis=None, scale=None, title=None, invert_y=True, min_max=None, screen=None,
         arrange=True, lut=None, max_projection=None, to_front=True, parent=None, sync=True):
    """
    Plot a source as 2d slices.

    Arguments
    ---------
    source : Source, pathlib.Path, list or dict
        The source to plot. If a list is given several synchronized windows are
        generated. If an element in the list is a list of sources those are
        overlayed in different colors in that window.
    axis : int or None
        The axis along which to slice the data.
    scale : tuple of float
        A spatial scale for each axis used for the spatial cursor position.
    title : str or None
        The title of the window.
    invert_y : bool
        If True invert the y axis (as typically done for images).
    min_max : tuple or None
        The minimal and maximal values for each source. If None, determine them from
        the source.
    screen : int or None
        Specify on which screen to open the window.

    Returns
    -------
    plot : DataViewer
      A data viewer class.
    """
    if not isinstance(source, (list, tuple)):
        source = [source]
    if isinstance(source, tuple):
        source = list(source)

    for i, src in enumerate(source):
        if isinstance(src, Path):
            source[i] = str(src)

    data_viewers = multi_plot(source, axis=axis, scale=scale, title=title, invert_y=invert_y,
                              min_max=min_max, max_projection=max_projection, screen=screen,
                              arrange=arrange, lut=lut, to_front=to_front,
                              parent=parent, sync=sync)
    if not runs_on_spyder():
        inst = QApplication.instance()
        # if inst is not None:
        #   inst.exec_()
    return data_viewers


def multi_plot(sources, axis=None, scale=None, title=None, invert_y=True, min_max=None,
               max_projection=None, arrange=True, screen=None, lut='flame', screen_percent=90, parent=None, sync=True, to_front=True):
    """
    Plot a source as 2d slices.

    Arguments
    ---------
    sources : list of sources
        The sources to plot.If an element in the list is a list of sources
        those are overlayed in different colors in that window.
    axis : int or None
        The axis along which to slice the data.
    scale : tuple of float
        A spatial scale for each axis used for the spatial cursor position.
    title : str or None
        The title of the window.
    invert_y : bool
        If True invert the y axis (as typically done for images).
    min_max : tuple or None
        The minimal and maximal values for each source. If None, determine them from
        the source.
    screen : int or None
        Specify on which screen to open the window.

    Returns
    -------
    plots : list of DataViewers
        A list of viewer classes.
    """

    if not isinstance(title, (tuple, list)):
        title = [title] * len(sources)
    if not isinstance(lut, (list, tuple)):
        lut = [lut] * len(sources)
    if min_max is None or np.isscalar(min_max[0]):  # Because it is a list of lists
        min_max = [min_max] * len(sources)
    if not isinstance(max_projection, list):
        max_projection = [max_projection] * len(sources)

    dvs = [dv.DataViewer(source=src, axis=axis, scale=scale, title=title_, invertY=invert_y,
                         minMax=min_max_, max_projection=max_projection_, default_lut=lut_, parent=parent)
           for src, title_, lut_, min_max_, max_projection_ in zip(sources, title, lut, min_max, max_projection)]

    if arrange:
        try:
            geo = qtu.tiled_layout(len(dvs), percent=screen_percent, screen=screen)
            for d, g in zip(dvs, geo):
                # d.setFixedSize(int(0.95 * g[2]), int(0.9 * g[3]))
                d.setGeometry(QRect(*g))
        except:  # FIXME: too broad
            pass

    if sync:
        for d1, d2 in itertools.combinations(dvs, 2):
            synchronize(d1, d2)

    if to_front:
        bring_to_front(dvs)

    #for d in dvs:
    #    d.update();
    
    return dvs


def arrange_plots(plots, screen = None, screen_percent = 90):
    try:
        geo = qtu.tiled_layout(len(plots), percent=screen_percent, screen=screen)

        for d, g in zip(plots, geo):
            d.setGeometry(pg.QtCore.QRect(*g))
    except:
        pass
  

def synchronize(viewer1, viewer2):
    """Synchronize scrolling between two data viewers"""
    def sync_d1_d2_scroll():
        """sync dv1 -> dv2"""
        viewer2.sliceLine.setValue(viewer1.sliceLine.value())

    def sync_d1_d2_button(button, ax):
        viewer2.axis_buttons[ax].setChecked(button.isChecked())

    viewer1.sliceLine.sigPositionChanged.connect(sync_d1_d2_scroll)
    for ax, button in enumerate(viewer1.axis_buttons):
        button.clicked.connect(ft.partial(viewer2.setSliceAxis, ax))
        button.clicked.connect(ft.partial(sync_d1_d2_button, button, ax))

    def sync_d2_d1_scroll():
        """sync dv2 -> dv1"""
        viewer1.sliceLine.setValue(viewer2.sliceLine.value())

    def sync_d2_d1_button(button, ax):
        viewer1.axis_buttons[ax].setChecked(button.isChecked())

    viewer2.sliceLine.sigPositionChanged.connect(sync_d2_d1_scroll)
    for ax, button in enumerate(viewer2.axis_buttons):
        button.clicked.connect(ft.partial(viewer1.setSliceAxis, ax))
        button.clicked.connect(ft.partial(sync_d2_d1_button, button, ax))

    viewer1.view.setXLink(viewer2.view)
    viewer1.view.setYLink(viewer2.view)


def set_source(viewer, source):
    """Set the source data in a viewer.

    Arguments
    ---------
    viewer : DataViewer
        The viewer to set a new source for.
    source : Source
        The source to use in the viewer.

    Returns
    -------
    viewer : DataViewer
        The viewer.
    """
    viewer.setSource(source)
    return viewer


def bring_to_front(plots):
    if not isinstance(plots, list):
        plots = [plots]
    for plot in plots:
        plot.setWindowFlag(pg.Qt.QtCore.Qt.WindowStaysOnTopHint)
        plot.raise_()
        plot.activateWindow()
        plot.show()


def close(plots='all'):
    if plots == 'all':
        pg.Qt.App.closeAllWindows()
    else:
        if not isinstance(plots, list):
            plots = [plots]
        for plot in plots:
            plot.close()


############################################################################################################
# ## Tests
############################################################################################################


def _test():
    import numpy as np
    import ClearMap.Visualization.Qt.Plot3d as p3d

    img1 = np.random.rand(*(100, 80, 30))
    img2 = np.random.rand(*(100, 80, 30)) > 0.5

    p = p3d.plot([img1, img2])  # analysis:ignore
