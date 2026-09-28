import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from natsort import natsorted

from ClearMap.Utils.exceptions import ClearMapValueError
from ClearMap.Utils.tag_expression import Expression


Pathlike = str | Path


############################################### TILE PATTERNS DISCOVERY ##############################################

@dataclass(frozen=True)
class ChannelPatternSpec:
    name: str                 # channel name from the UI
    data_type: str            # selected content type
    extension: str            # e.g. ".ome.tif" (normalized to str)
    pattern_relpath: str      # pattern string relative to src_folder


def _as_list(x):
    return x if isinstance(x, list) else [x]


def get_tiles_list_from_sample_folder(src_dir: Path, min_file_number: int = 10,
                                      tile_extensions: List[str] = ['.ome.tif', '.ome.npy']) -> Dict[Path, List[Path]]:
    tile_extensions = _as_list(tile_extensions)
    data_dirs: Dict[Path, List[Path]] = {}
    for f_name in sorted(src_dir.iterdir()):
        f_path = src_dir / f_name
        if f_path.is_dir():
            for tile_extension in tile_extensions:
                tiles = sorted(f_path.glob(f'*{tile_extension}'))  # , recursive=True)
                if tiles and len(tiles) > min_file_number:
                    data_dirs[f_path] = tiles
                    break  # Only get the first tile extension found
    return data_dirs


def extract_channel_number(pattern_str: str) -> Optional[int]:
    match = re.search(r'(?:_)?[cC](\d{1,2})\b', pattern_str)
    if match:
        return int(match.group(1))
    return None


def pattern_finders_from_base_dir(src_dir: Pathlike, min_file_number: int = 10,
                                  tile_extension: List[str] = ['.ome.tif', '.ome.npy']) -> List["PatternFinder"]:
    src_dir = Path(src_dir)
    data_dirs = get_tiles_list_from_sample_folder(src_dir, min_file_number=min_file_number,
                                                  tile_extensions=tile_extension)
    finders = []
    for path in data_dirs.keys():
        sub_dir = path.relative_to(src_dir)
        tmp = PatternFinder.from_mixed_file_list(src_dir / sub_dir, data_dirs[path])
        if isinstance(tmp, (tuple, list)):
            finders.extend(tmp)
        else:
            finders.append(tmp)

    channel_finders = []
    no_channel_finders = []
    for finder in finders:
        if extract_channel_number(finder.pattern.pattern_str) is not None:
            channel_finders.append(finder)
        else:
            no_channel_finders.append(finder)

    channel_finders = natsorted(channel_finders, key=lambda f: extract_channel_number(f.pattern.pattern_str))
    no_channel_finders = natsorted(no_channel_finders, key=lambda f: f.pattern.pattern_str)

    return no_channel_finders + channel_finders


class Pattern(Expression):
    """
    Extends expression to support unlabeled axes. We start with undefined axes i.e. I,J,K, etc.
    and then we replace them with the actual axis name when we know it.
    we can parse the pattern from a string containing the placeholders (by default '?').
    We can also highlight a given axis specified by index or name.
    """
    def __init__(self, pattern_str):
        super().__init__(None)
        self.pattern_str = pattern_str
        self.bg_color = "#60798B"
        self._text_color = "#1A72BB"
        self.html_style = f"background-color:{self.bg_color};text-color:{self._text_color}"
        self.place_holder_symbol = '?'
        super().__init__(self.__placeholder_pattern_to_expression_string(pattern_str))

    def string(self, values=None):
        # if values is None:
        #     return self.pattern_str
        if isinstance(values, dict) and 'C' in values.keys():
            if 'C' not in self.tag_names():
                self.set_channel_tag_name()
        return super().string(values)

    def relative_string(self, base_dir: Path) -> str:
        """Return the pattern string relative to base_dir."""
        return str(Path(self.string()).relative_to(base_dir))

    def assign_axes_from_combo(self, axis_names: list[str]) -> None:
        """axis_names is a list like ['Z','Y','X'] mapped to existing tag slots."""
        if len(axis_names) != self.n_tags():
            raise ClearMapValueError('Axis list length must match number of tags')
        if 'C' in axis_names:
            raise NotImplementedError(f'Channel splitting is not implemented yet, cannot split axis nb{axis_names.index("C")}')
        self.set_axes_names(axis_names)

    def set_axis_name(self, axis_index, new_name):
        self.tags[axis_index].name = new_name

    def set_axes_names(self, names):
        if isinstance(names, dict):
            for ax, name in names.items():
                axis_index = [i for i, tag in enumerate(self.tags) if tag.name == ax]
                self.set_axis_name(axis_index, name)
        else:
            for i, name in enumerate(names):
                self.set_axis_name(i, name)

    def set_channel_tag_name(self):
        chan_start_idx = str(self).find('C<') + 1
        for i, tag in enumerate(self.tag_names()):
            tag_idx_range = self.char_index(tag, with_markups=True)
            if tag_idx_range[0] == chan_start_idx:
                self.set_axis_name(i, 'C')
                break

    def get_channel_indices(self):
        if 'C' in self.tag_names():
            return self.char_index('C')
        if self.pattern_str.find('C?') == -1:
            return None
        channel_indices = []
        for i in range(self.pattern_str.find('C?') + 1, len(self.pattern_str)):
            if self.pattern_str[i] == '?':
                channel_indices.append(i)
            else:
                break
        return channel_indices

    def highlight_digits(self, axis=None, cluster_idx=None):
        """
        Highlight the placeholder digits of the pattern at index cluster_idx

        Parameters
        ----------
        cluster_idx: int
            The index in the string of the first digit to be highlighted

        Returns
        -------

        """
        if cluster_idx is not None:
            n_digits = 0
            for i, c in enumerate(self.pattern_str[cluster_idx:]):
                if c == self.place_holder_symbol:
                    n_digits += 1
                else:
                    out = (self.pattern_str[:cluster_idx] +
                           self.__highlighted_symbols(n_digits) +
                           self.pattern_str[cluster_idx + n_digits:])
                    return out
        elif axis is not None:
            out = list(self.string(values={ax: '?' for ax in self.tag_names()}))
            start, end = self.char_index(axis)
            out[start:end] = list(self.__highlighted_symbols(end - start))
            return ''.join(out)
        else:
            raise ClearMapValueError('Must provide either cluster_idx or axis')


    def __highlighted_symbols(self, n, symbol='?'):
        return f'<span style={self.html_style}>{symbol * n}</span>'

    def __placeholder_pattern_to_expression_string(self, pattern_str):
        """
        Convert a pattern string with placeholders
        (e.g. '/data/experiment/17-34-31_auto_Blaze_C00_xyz-Table Z????.ome.tif')
        to an expression string with tags.
        (e.g. '/data/experiment/17-34-31_auto_Blaze_C00_xyz-Table Z<I,4>.ome.tif')

        .. note::
            The placeholder symbol is self.place_holder_symbol (default '?')
            The dimensions are named I, J, K, etc.

        Parameters
        ----------
        pattern_str: str
            The pattern string with placeholders

        Returns
        -------
        str
            The expression string with tags
        """
        pattern_str = self.__fuse_zeros(pattern_str)
        self.pattern_str = pattern_str
        expression_string = ''
        current_dim = 'I'
        dim_size = 0
        for i, c in enumerate(pattern_str):
            if c == self.place_holder_symbol:
                if dim_size == 0:
                    expression_string += f'<{current_dim},'
                dim_size += 1
            else:
                if dim_size > 0:
                    expression_string += f'{dim_size}>'
                    current_dim = chr(ord(current_dim) + 1)
                    dim_size = 0
                expression_string += c
        return expression_string

    def __fuse_zeros(self, pattern):
        """
        When not all digits are used in a zero padded pattern and were not detected.

        Parameters
        ----------
        pattern: str
            The pattern to be fixed

        Returns
        -------
        str
            The pattern with all 0s attached to ? converted to ?
        """
        pattern = list(pattern)
        for i in range(len(pattern) - 1, 1, -1):  # 1 to avoid overshooting
            if pattern[i] == self.place_holder_symbol and pattern[i - 1] == '0':
                pattern[i - 1] = self.place_holder_symbol
        return ''.join(pattern)


class PatternFinder:  # TODO: from_df class_method
    def __init__(self, folder, tiff_list=None, df=None, axes_order=None):
        self.folder = Path(folder)

        if tiff_list is not None:
            self.df = self.file_list_to_df(tiff_list)
        elif df is not None:
            self.df = df
        else:
            raise ValueError('Must supply at least tiff_list or df')
        self.pattern = self.pattern_from_df(self.df)
        if axes_order is not None:
            self.pattern.axes_order = axes_order

    @classmethod
    def from_mixed_file_list(cls, folder, file_list):
        """
        Create a PatternFinder from a list of tiff paths potentially containing different channels

        Parameters
        ----------
        folder
        file_list

        Returns
        -------

        """
        df = cls.file_list_to_df(file_list)
        pattern = cls.pattern_from_df(df)
        finders = cls.split_channel(folder, df, pattern)
        if finders:
            return finders
        else:
            print(f'Could not find different channels in Pattern {pattern.pattern_str}')
            return cls(folder, df=df)

    @staticmethod
    def file_list_to_df(file_names):
        file_names = [str(f_name) for f_name in file_names]

        # Group file paths by length (different length means different patterns)
        grouped_files = {}
        for f_name in file_names:
            length = len(f_name)
            if length not in grouped_files:
                grouped_files[length] = []
            grouped_files[length].append(f_name)

        dataframes = []
        for length, files in grouped_files.items():
            data = [list(f_name) for f_name in files]
            dataframes.append(pd.DataFrame(data))

        if len(dataframes) == 1:
            return dataframes[0]
        elif len(dataframes) == 0:
            raise ClearMapValueError(f'No files found in list: {file_names}')
        else:
            raise NotImplementedError('Multiple patterns other than channel (different length)'
                                      ' in the same folder not supported yet')

    @classmethod
    def split_channel(cls, folder, df, pattern):
        channel_indices = pattern.get_channel_indices()
        if not channel_indices:
            return
        # Remove rows with identical values in channel_indices columns
        channel_df = df.iloc[:, channel_indices].drop_duplicates()
        # Concatenate remaining columns and cast to int
        channel_numbers = channel_df.apply(lambda x: int(''.join(x.astype(str))), axis=1).values

        other_axes = {ax: '?' for ax in pattern.tag_names() if ax != 'C'}
        expressions = [Pattern(pattern.string(values={'C': c, **other_axes})) for c in channel_numbers]

        pattern_finders = [cls(folder, e.glob()) for e in expressions]
        return pattern_finders

    @staticmethod
    def pattern_from_df(df):
        pattern = ''
        first_row = df.iloc[0]
        for i, col in enumerate(df):
            pattern += first_row[i] if (df[col] == first_row[i]).all() else '?'  # ? if not all letters are the same in the column
        # FIXME: skips some spaces x and ]
        return Pattern(pattern)

    def to_channel_config(self, *, data_type: str, extension: str, base_dir: Path) -> dict:
        """
        Returns a minimal config subtree for this channel.
        """
        # ensure channel tag is set if needed
        _ = self.pattern.string()  # ensures internal state
        return {
            'data_type': data_type,
            'extension': extension,
            'path': self.pattern.relative_string(base_dir),
            'resolution': [1, 1, 1],
            'orientation': (0, 0, 0),
            'comments': '',
            'slicing': {'x': None, 'y': None, 'z': None}
        }
