import numpy as np
import pandas as pd
import pyqtgraph as pg
from PyQt5.QtGui import QColor
from matplotlib.colors import to_hex

from ClearMap.gui.gui_utils import pseudo_random_rgb_array


class Scatter3D:
    def __init__(self, coordinates, smarties=False, colors=None, hemispheres=None, half_slice_thickness=None):
        self.__coordinates = None
        self.__has_hemispheres = hemispheres is not None
        self.default_symbol = '+'
        self.alternate_symbol = 'p'
        self.half_slice_thickness = half_slice_thickness
        self.axis = 2

        if smarties and colors is None:
            n_samples = coordinates.shape[0]
            colors = pseudo_random_rgb_array(n_samples)
        if colors is not None and not isinstance(colors[0], str):  # Convert to hex if not yet
            colors = [to_hex(c) for c in colors]

        self.__has_colours = colors is not None

        if hemispheres is not None:
            self.symbol_map = {
                0: self.default_symbol,
                255: self.alternate_symbol
            }
            # symbols = symbols.decode()

        # colors = if colors is None else np.array([QColor( * col.astype(int)) for col in colors]
        self.data = pd.DataFrame({
            'x': coordinates[:, 0],
            'y': coordinates[:, 1],
            'z': coordinates[:, 2],
            'hemisphere': hemispheres,
            'colour': colors
        })  # TODO: could use id instead of colour
        if colors is not None:
            unique_colors = np.unique(colors)
            self.point_map = pd.DataFrame({
                'colour': unique_colors,
                'pen': [pg.mkPen(c) for c in unique_colors],
                'brush': [pg.mkBrush(c) for c in unique_colors]
            })

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
        rows = [pd.DataFrame(columns=['x', 'y', 'colour', 'size'])]
        for i in range(main_slice_idx - half_slice_thickness, main_slice_idx + half_slice_thickness):
            current_z_data = pd.DataFrame(columns=['x', 'y', 'colour', 'size'])  # WARNING: this is x/y of the view, not the 3D image
            if i < 0:  # or i > self.coordinates[:, 2].max()
                continue
            else:
                current_slice = i
            pos = self.get_pos(current_slice)
            if not all(pos.shape):  # empty
                continue
            current_z_data[['x', 'y']] = pos
            current_z_data['colour'] = self.get_colours(current_slice).values  # Otherwise uses index from source
            current_z_data['size'] = self.get_symbol_sizes(main_slice_idx, current_slice, half_size=half_slice_thickness)
            rows.append(current_z_data)

        data_df = pd.concat(rows)
        output = {'pos': data_df[['x', 'y']].values,  # WARNING: this is x/y of the view, not the 3D image
                  'size': data_df['size'].values}
        if self.has_colours:
            output['pen'] = data_df['colour'].map(dict(self.point_map[['colour', 'pen']].values)).values
        return output

    def get_draw_params(self, current_slice):
        colours = self.get_colours(current_slice)
        draw_params = {
            'pen': colours.map(dict(self.point_map[['colour', 'pen']].values)).values,
            'brush': colours.map(dict(self.point_map[['colour', 'brush']].values)).values
        }
        return draw_params

    def get_symbols(self, current_slice):
        if self.has_hemispheres:
            indices = self.current_slice_indices(current_slice)
            if indices is not None:
                return self.data.loc[indices, 'hemisphere'].map(self.symbol_map).values
            else:
                return np.array([])
        else:
            return self.default_symbol

    def get_symbol_sizes(self, main_slice_idx, slice_idx, half_size=3):
        marker_size = round(10 * ((half_size - abs(main_slice_idx - slice_idx)) / half_size))
        n_markers = self.get_n_markers(slice_idx)
        return np.full(n_markers, marker_size)

    def get_n_markers(self, slice_idx):
        if len(self.data):
            return np.count_nonzero(self.current_slice_indices(slice_idx))
        else:
            return 0

    def get_colours(self, current_slice):
        indices = self.current_slice_indices(current_slice)
        if indices is not None:
            return self.data.loc[indices, 'colour']
        else:
            return np.array([])

    def current_slice_indices(self, current_slice):
        if len(self.data):
            return self.coordinates[:, self.axis] == current_slice

    def get_pos(self, current_slice):
        indices = self.current_slice_indices(current_slice)
        if indices is not None:
            axes = [0, 1, 2]
            axes.pop(self.axis)  # coordinates in the two other axes
            return self.coordinates[np.ix_(indices, axes)]
        else:
            return np.empty((0, 2))
