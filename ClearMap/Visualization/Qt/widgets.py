import re

import numpy as np
import pandas as pd

from matplotlib.colors import to_hex
from PyQt5.QtGui import QColor
import pyqtgraph as pg

from ClearMap.gui.gui_utils import pseudo_random_rgb_array


def is_valid_hex_color(s):
    """
    To check if the input is a valid hex color triplet despite the lack of # at the start
    """
    return bool(re.fullmatch(r'(?:#[0-9a-fA-F]{6}|[0-9a-fA-F]{6})', s))


class Scatter3D:
    def __init__(self, coordinates, smarties=False, colors=None, hemispheres=None, half_slice_thickness=None,
                 marker_size=5):
        self.__coordinates = None
        self.__has_hemispheres = hemispheres is not None  # FIXME: this should be renamed to has_different_symbols
        self.half_slice_thickness = half_slice_thickness
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
                    colors_dict = {col: to_hex(col) for col in np.unique(colors, axis=0)}
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

    def set_data(self, df):
        # print(self.data['colour'].values, df['colour'])
        if isinstance(df, dict):
            df = pd.DataFrame(df)
        if 'colour' in df.columns:
            if isinstance(df['colour'][0], np.ndarray):  # TODO: should be iterable
                df['colour'] = [to_hex(c) for c in df['colour']]  # OPTIMISE: see map
            elif isinstance(df['colour'][0], QColor):
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

    def get_all_data(self, main_slice_idx, half_slice_thickness=3):  # FIXME: rename
        """
        Get surrounding markers

        Parameters
        ----------
        main_slice_idx int

        half_slice_thickness int

        Returns
        -------

        """
        if not half_slice_thickness:  # WARNING: optimisation
            output = {
                'pos': np.empty(0),
                'size': np.empty(0),
            }
            if self.has_colours:
                output['pen'] = np.empty(0)
            return output
        half_slice_thickness = self.half_slice_thickness if self.half_slice_thickness is not None else half_slice_thickness
        columns = ['x', 'y', 'colour', 'size', 'symbol']
        if self.has_colours:
            columns += ['pen']
        rows = [pd.DataFrame(columns=columns)]
        for i in range(main_slice_idx - half_slice_thickness, main_slice_idx + half_slice_thickness):
            if i < 0:  # or i > self.coordinates[:, 2].max()
                continue
            else:
                current_slice = i
            current_z_data = pd.DataFrame(columns=['x', 'y', 'colour', 'size', 'symbol'])  # WARNING: this is x/y of the view, not the 3D image
            indices = self.current_slice_indices(current_slice)
            pos = self.get_pos(indices=indices)
            if not all(pos.shape):  # empty
                continue
            current_z_data[['x', 'y']] = pos
            current_z_data['colour'] = self.get_colours(indices=indices).values  # Otherwise uses index from source
            current_z_data['size'] = self.get_symbol_sizes(main_slice_idx, current_slice,
                                                           indices=indices, half_size=half_slice_thickness)
            current_z_data['symbol'] = self.get_symbols(current_slice)  # FIXME: check if we need to verify that symbools exist
            if self.has_colours:
                current_z_data['pen'] = self.data.loc[indices, 'pen'].values
                # current_z_data['brush'] = self.data.loc[indices, 'brush'].values
            rows.append(current_z_data)

        data_df = pd.concat(rows)  # FIXME: check if to_dict method in dataframe
        output = {'pos': data_df[['x', 'y']].values,  # WARNING: this is x/y of the view, not the 3D image
                  'size': data_df['size'].values,
                  'symbol': data_df['symbol'].values}
        if self.has_colours:
            output['pen'] = data_df['pen'].values
            # output['brush'] = data_df['brush'].values
        return output

    def get_draw_params(self, current_slice):
        indices = self.current_slice_indices(current_slice)
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
            indices = self.current_slice_indices(current_slice)
            if indices is not None:
                return self.data.loc[indices, 'symbol'].values
            else:
                return np.array([])
        else:
            return self.symbols[0]

    def get_symbol_sizes(self, main_slice_idx, slice_idx, indices=None, half_size=3):
        marker_size = round(10 * ((half_size - abs(main_slice_idx - slice_idx)) / half_size))
        n_markers = self.get_n_markers(indices=indices)
        return np.full(n_markers, marker_size)

    def get_n_markers(self, slice_idx=None, indices=None):
        if indices is None:
            indices = self.current_slice_indices(slice_idx)
        if len(self.data):
            return np.count_nonzero(indices)
        else:
            return 0

    def get_colours(self, current_slice=None, indices=None):
        if indices is None:
            indices = self.current_slice_indices(current_slice)
        if indices is not None:
            return self.data.loc[indices, 'colour']
        else:
            return np.array([])

    def current_slice_indices(self, current_slice):  #  FIXME: more of a mask actually
        if len(self.data):
            return self.coordinates[:, self.axis] == current_slice

    def get_pos(self, current_slice=None, indices=None):
        if indices is None:
            indices = self.current_slice_indices(current_slice)
        if indices is not None:
            axes = [0, 1, 2]
            axes.pop(self.axis)  # coordinates in the two other axes
            return self.coordinates[np.ix_(indices, axes)]
        else:
            return np.empty((0, 2))
