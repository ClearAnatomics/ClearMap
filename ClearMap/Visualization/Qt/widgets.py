import math
import re

import numpy as np
import pandas as pd

from matplotlib.colors import to_hex
from PyQt5.QtGui import QColor
import pyqtgraph as pg

from ClearMap.gui.gui_utils_images import pseudo_random_rgb_array


def is_valid_hex_color(s):
    """
    To check if the input is a valid hex color triplet despite the lack of # at the start
    """
    return bool(re.fullmatch(r'(?:#[0-9a-fA-F]{6}|[0-9a-fA-F]{6})', s))


class Scatter3D:
    """
    Scatter dataset for display in :class:`DataViewer`.

    Markers whose coordinate along the scroll axis matches the current slice
    are drawn at full size.  When *z_radius* is set, markers from neighbouring
    slices are also drawn at reduced size to convey depth.

    Parameters
    ----------
    coordinates : (N, 3) np.ndarray or pd.DataFrame
        Point positions, or a pre-built DataFrame with columns ``x``, ``y``,
        ``z``, ``colour``, ``symbol`` (and optionally ``pen`` / ``brush``).
        Passing a DataFrame skips the construction logic entirely.
    smarties : bool
        Assign a pseudo-random colour to every point (ignored when *colors*
        is provided).
    colors : array-like or None
        Per-point colours as hex strings or RGB(A) arrays.  Converted to
        hex internally.
    hemispheres : array-like or None
        Integer label per point that controls the marker symbol.  -1 is
        reserved for out-of-hemisphere points, drawn with
        ``out_of_bounds_symbol``.
    z_radius : int or None
        Half-width of the depth window in slices.  Points within
        ±*z_radius* of the current slice are drawn at a size proportional
        to their proximity.  ``None`` / 0 disables depth display.
    marker_size : int
        Base marker diameter in display pixels at the current slice
        (minimum 2).
    """
    def __init__(self, coordinates, smarties=False, colors=None, hemispheres=None, z_radius=None,
                 marker_size=5):
        self.__coordinates = None
        self._axis_indices = {}  # axis -> (order, sorted_keys), built lazily by slab_indices
        self._default_pen = None
        self._default_brush = None
        self._clear_brush = None
        self.__has_hemispheres = hemispheres is not None  # FIXME: this should be renamed to has_different_symbols
        self.z_radius = z_radius
        self.axis = 2
        self.marker_size = max(2, marker_size)
        self.out_of_bounds_symbol = 'x'  # Symbol to use for out of bounds markers

        if isinstance(coordinates, pd.DataFrame):
            self.data = coordinates
            self.symbols = self.data['symbol'].unique().tolist()
            self.__has_colours = self.data['colour'].nunique() > 1
            self.__has_hemispheres = len(self.symbols) > 1
        else:
            self.symbols = ['+', 'p']

            if smarties and colors is None:
                n_samples = coordinates.shape[0]
                colors = pseudo_random_rgb_array(n_samples)
            if colors is not None and not is_valid_hex_color(str(colors[0])):  # Convert to hex if not yet
                if not smarties:
                    colors_dict = {tuple(col): to_hex(col) for col in np.unique(colors, axis=0)}
                    colors_dict[None] = to_hex((1, 0, 0))  # default to red
                    colors = [colors_dict[col] for col in colors]
                else:
                    colors = [to_hex((1, 0, 0) if c is None else c) for c in colors]

            self.__has_colours = colors is not None

            if hemispheres is not None:
                hemispheres_values = np.unique(hemispheres)
                if -1 in hemispheres_values:  # If there are values outside of the hemispheres
                    symbols = [self.out_of_bounds_symbol] + self.symbols
                else:
                    symbols = self.symbols
                # FIXME: id_ is probably unhashable
                self.symbol_map = {id_: symbols[i] for i, id_ in enumerate(hemispheres_values)}

            # colors = colors if colors is None else np.array([QColor( * col.astype(int)) for col in colors]
            self.data = pd.DataFrame({
                'x': coordinates[:, 0],
                'y': coordinates[:, 1],
                'z': coordinates[:, 2],
                'hemisphere': hemispheres,
                'colour': colors
            })  # TODO: could use id instead of colour
            self.data['symbol'] = self.data['hemisphere'].map(self.symbol_map) if self.__has_hemispheres else self.symbols[0]
            self.data['colour'] = self.data['colour'].astype(str)

        if self.__has_colours and not 'pen' in self.data.columns:
            # Finalise DF
            if colors is None:
                colors = self.data['colour'].values
            if colors is not None:
                unique_colors = np.unique(colors)
                self.point_map = pd.DataFrame({
                    'colour': unique_colors,
                    'pen': [pg.mkPen(c) for c in unique_colors],
                    'brush': [pg.mkBrush(c) for c in unique_colors]
                })

            self.data['pen'] = self.data['colour'].map(dict(self.point_map[['colour', 'pen']].values))
            self.data['brush'] = self.data['colour'].map(dict(self.point_map[['colour', 'brush']].values))

    @property
    def coordinates(self):
        if self.__coordinates is None:
            self.__coordinates = self.data[['x', 'y', 'z']].values
        return self.__coordinates

    @property
    def plane_axes(self):
        return [a for a in range(3) if a != self.axis]

    def set_data(self, df):
        # print(self.data['colour'].values, df['colour'])
        if isinstance(df, dict):
            df = pd.DataFrame(df)
        if len(df) and 'colour' in df.columns:
            sample_colour = df['colour'][0]
            if isinstance(sample_colour, np.ndarray):  # TODO: should be iterable
                df['colour'] = [to_hex(c) for c in df['colour']]  # OPTIMISE: see map
            elif isinstance(sample_colour, QColor):
                df['colour'] = [c.name() for c in df['colour']]  # OPTIMISE: see map
            unique_colors = np.unique(df['colour'])
            self.point_map = pd.DataFrame({
                'colour': unique_colors,
                'pen': [pg.mkPen(c) for c in unique_colors],
                'brush': [pg.mkBrush(c) for c in unique_colors]
            })
        if set(self.data.columns) >= set(df.columns):
            for col in set(self.data.columns) - set(df.columns):  # Add if missing
                df[col] = None
            self.data = df
            # print(self.data['colour'].values, df['colour'].values)
            self.__coordinates = None
            self._axis_indices = {}

    @property
    def has_colours(self):
        return self.__has_colours

    @property
    def has_hemispheres(self):
        return self.__has_hemispheres

    def _axis_index(self, axis):
        """
        Row order sorted by the (integer) position along *axis*, and the sorted keys.

        Built once per axis on first use (~1 s for 8M points), then every slab lookup is
        two binary searches instead of a full-array comparison. Non-finite coordinates
        get a sentinel key that no slab request can reach, so they are never drawn.
        """
        cached = self._axis_indices.get(axis)
        if cached is None:
            values = self.coordinates[:, axis]
            finite = np.isfinite(values)
            keys = np.full(len(values), np.iinfo(np.int32).min, dtype=np.int32)
            keys[finite] = np.clip(np.floor(values[finite]), -2 ** 30, 2 ** 30)
            order = np.argsort(keys, kind='stable').astype(np.int32)
            cached = (order, keys[order])
            self._axis_indices[axis] = cached
        return cached

    def is_prepared(self, axis=None):
        """Whether the slice index of *axis* (default: current one) is built, i.e. slab lookups are instant."""
        return (self.axis if axis is None else axis) in self._axis_indices

    def prepare(self, axes=None, progress=None):
        """
        Build the slice index now instead of at the first draw (~1 s per axis for 8M markers).

        Parameters
        ----------
        axes : int or iterable of int or None
            The axes to prepare (default: the current one).
        progress : Callable[[str], None] or None
            Called with a short message before each axis is indexed.
        """
        if axes is None:
            axes = [self.axis]
        elif isinstance(axes, int):
            axes = [axes]
        for axis in axes:
            if not self.is_prepared(axis):
                if progress is not None:
                    progress(f'Indexing markers along axis {axis}')
                self._axis_index(axis)

    def slab_indices(self, lo, hi, axis=None):
        """
        Row indices of the points with ``lo <= floor(coord[axis]) < hi`` (negative positions are never returned).
        Rows are in ascending original order within one slice.
        """
        order, keys = self._axis_index(self.axis if axis is None else axis)
        # Needles must have the keys' dtype: otherwise numpy upcasts the whole (8M) key array on every call
        needles = np.array([max(math.ceil(lo), 0), max(math.ceil(hi), 0)]).clip(0, 2 ** 30).astype(np.int32)
        start, stop = np.searchsorted(keys, needles)
        return order[start:stop]

    def build_draw_data(self, index, *, base_size=None, main_size=None, zoom_factor=1.0,
                        view_rect=None, depth_limit=None):
        """
        Everything needed to draw the markers of the slice *index* (and its neighbours) in a
        single ``ScatterPlotItem.setData`` call.

        Parameters
        ----------
        index : int
            The current slice along ``self.axis``.
        base_size : int or None
            Size of a marker at distance 0, used for the neighbours' sizes (defaults to ``self.marker_size``).
        main_size : int or None
            Size of the markers in the current slice (defaults to *base_size*).
        zoom_factor : float
            Multiplies the neighbours' sizes (zoom-scaled markers).
        view_rect : ((x0, x1), (y0, y1)) or None
            Only markers inside this rectangle (plane coordinates) are returned.
        depth_limit : int or None
            Neighbour markers are omitted if more than this many would be drawn
            (``None``: no limit, ``0``: never draw them).

        Returns
        -------
        data : dict or None
            Keyword arguments for ``setData`` (``pos, size, symbol, pen, brush``), None if there is nothing to draw.
        info : dict
            ``n_main``, ``n_depth`` (drawn), ``n_depth_candidates`` (before applying the limit), ``depth_skipped``.

        Note
        ----
        Markers of the current slice are drawn once (not again as hollow neighbours) and neighbours whose
        size rounds to 0 are dropped.
        """
        axis, plane = self.axis, self.plane_axes
        base_size = self.marker_size if base_size is None else base_size
        main_size = base_size if main_size is None else main_size
        coords = self.coordinates

        def gather(idx):
            c = coords[idx]
            if view_rect is not None and len(idx):
                (x0, x1), (y0, y1) = view_rect
                px, py = c[:, plane[0]], c[:, plane[1]]
                keep = (px >= x0) & (px <= x1) & (py >= y0) & (py <= y1)
                idx, c = idx[keep], c[keep]
            return idx, c

        main_idx, main_c = gather(self.slab_indices(index, index + 1))

        radius = self.z_radius
        depth_idx = np.empty(0, dtype=np.int32)
        depth_c = np.empty((0, 3))
        depth_size = np.empty(0, dtype=int)
        n_candidates = 0
        depth_skipped = False
        if radius and depth_limit != 0:
            idx = np.concatenate([self.slab_indices(index - radius, index),
                                  self.slab_indices(index + 1, index + radius)])
            idx, c = gather(idx)
            dist = np.abs(c[:, axis] - index)
            size = np.round(base_size * ((radius - dist) / radius)).astype(int)
            size = np.round(size * zoom_factor).astype(int)
            keep = size > 0
            depth_idx, depth_c, depth_size = idx[keep], c[keep], size[keep]
            n_candidates = len(depth_idx)
            if depth_limit is not None and n_candidates > depth_limit:
                depth_skipped = True
                depth_idx, depth_c, depth_size = depth_idx[:0], depth_c[:0], depth_size[:0]

        info = {'n_main': len(main_idx), 'n_depth': len(depth_idx),
                'n_depth_candidates': n_candidates, 'depth_skipped': depth_skipped}
        if not (len(main_idx) or len(depth_idx)):
            return None, info

        rows = np.concatenate([main_idx, depth_idx])
        if self.has_colours:
            pens = self.data['pen'].values[rows]
            brushes = self.data['brush'].values[rows]
        else:
            if self._default_pen is None:
                self._default_pen, self._default_brush = pg.mkPen('red'), pg.mkBrush('red')
            pens = np.full(len(rows), self._default_pen, dtype=object)
            brushes = np.full(len(rows), self._default_brush, dtype=object)
        if self._clear_brush is None:
            self._clear_brush = pg.mkBrush((0, 0, 0, 0))
        brushes[len(main_idx):] = self._clear_brush  # neighbours are hollow
        if self.has_hemispheres:
            symbols = self.data['symbol'].values[rows]
        else:
            symbols = np.full(len(rows), self.symbols[0], dtype=object)

        data = {'pos': np.concatenate([main_c[:, plane], depth_c[:, plane]]),
                'size': np.concatenate([np.full(len(main_idx), main_size, dtype=int), depth_size]),
                'symbol': symbols, 'pen': pens, 'brush': brushes}
        return data, info

    def get_symbol_sizes(self, main_slice_idx, slice_idx, indices=None, half_size=3):
        marker_size = round(self.marker_size * ((half_size - abs(main_slice_idx - slice_idx)) / half_size))
        n_markers = self.get_n_markers(indices=indices)
        return np.full(n_markers, marker_size)

    def get_n_markers(self, slice_idx=None, indices=None):
        if indices is None:
            indices = self.current_slice_mask(slice_idx)
        if len(self.data):
            return np.count_nonzero(indices)
        else:
            return 0

    def get_colours(self, current_slice=None, indices=None):
        if indices is None:
            indices = self.current_slice_mask(current_slice)
        if indices is not None:
            return self.data.loc[indices, 'colour']
        else:
            return np.array([])

    def current_slice_mask(self, current_slice):
        if len(self.data):
            return self.coordinates[:, self.axis] == current_slice
        return np.zeros(len(self.data), dtype=bool)