import re

import cached_property
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

    @property
    def has_colours(self):
        return self.__has_colours

    @property
    def has_hemispheres(self):
        return self.__has_hemispheres

    def get_3d_markers(self, main_slice_idx, z_radius=3, base_size=None):
        """
        Collect markers from slices surrounding *main_slice_idx*.

        Marker size scales linearly with proximity: a point at distance *d*
        from the main slice gets size ``base_size * (z_radius - d) / z_radius``.

        Parameters
        ----------
        main_slice_idx : int
            Index of the currently displayed slice.
        z_radius : int
            Half-width of the depth window.  Overridden by ``self.z_radius``
            when that attribute is set.
        base_size : int or None
            Reference size for a marker at distance 0.  Pass the current UI
            spin-box value so surrounding markers scale consistently with the
            main-slice markers.  Defaults to ``self.marker_size``.

        Returns
        -------
        dict
            Keys: ``'pos'`` (N×2), ``'size'`` (N,), ``'symbol'`` (N,),
            and ``'pen'`` (N,) when :attr:`has_colours` is True.
            Returns empty arrays when *z_radius* is falsy or no points fall
            in range.
        """
        empty = {'pos': np.empty(0), 'size': np.empty(0), 'symbol': np.empty(0)}
        if self.has_colours:
            empty['pen'] = np.empty(0)

        z_radius = self.z_radius if self.z_radius is not None else z_radius
        if not z_radius:
            return empty

        base_size = base_size if base_size is not None else self.marker_size

        z = self.coordinates[:, self.axis]
        mask = (z >= main_slice_idx - z_radius) & (z < main_slice_idx + z_radius) & (z >= 0)
        if not mask.any():
            return empty

        pos = self.coordinates[mask][:, self.plane_axes]
        dist_to_main = np.abs(z[mask] - main_slice_idx)
        sizes = np.round(base_size * ((z_radius - dist_to_main) / z_radius)).astype(int)

        symbols = (self.data.loc[mask, 'symbol'].values if self.has_hemispheres
                   else np.full(mask.sum(), self.symbols[0]))

        output = {'pos': pos, 'size': sizes, 'symbol': symbols}
        if self.has_colours:
            output['pen'] = self.data.loc[mask, 'pen'].values
        return output

    def get_draw_params(self, current_slice):
        indices = self.current_slice_mask(current_slice)
        if indices is not None:
            draw_params = {
                'pen': self.data.loc[indices, 'pen'].values,
                'brush': self.data.loc[indices, 'brush'].values
            }
            return draw_params
        else:
            return {'pen': np.empty(0), 'brush': np.empty(0)}

    def get_symbols(self, current_slice):
        if self.has_hemispheres:
            indices = self.current_slice_mask(current_slice)
            if indices is not None:
                return self.data.loc[indices, 'symbol'].values
            else:
                return np.array([])
        else:
            return self.symbols[0]

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

    def get_pos(self, current_slice=None, indices=None):
        if indices is None:
            indices = self.current_slice_mask(current_slice)
        if indices is not None:
            return self.coordinates[np.ix_(indices, self.plane_axes)]
        else:
            return np.empty((0, 2))
