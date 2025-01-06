import glob
import os
from copy import deepcopy
from pathlib import Path

import pandas as pd

import tifffile
from PIL import Image
from tqdm import tqdm

from ClearMap.IO.Workspace import Workspace
from ClearMap.IO import IO as clearmap_io
from ClearMap.Utils.tag_expression import Expression


class MetadataError(Exception):
    pass


# class NotAnOmeFile(tifffile.TiffFileError):   Not available in tifffile==0.15.1
class NotAnOmeFile(Exception):
    pass


def _get_ome_dict_tifffile(img_path):  # WARNING: works only with recent versions of tifffile not 0.15.1
    if not tifffile.TiffFile(img_path).is_ome:
        raise NotAnOmeFile(f"File {img_path} is not a valid ome.tif file")
    ome_metadata = tifffile.tiffcomment(img_path)
    ome_dict = tifffile.xml2dict(ome_metadata)
    return ome_dict


def _get_ome_dict_pil(img_path):
    not_an_ome_msg = f"File {img_path} is not a valid ome.tif file"
    if not img_path.endswith('.ome.tif'):  # Weak but can't rely on tifffile
        raise NotAnOmeFile(not_an_ome_msg)
    img = Image.open(img_path)
    if 270 not in img.tag.keys():
        raise NotAnOmeFile(not_an_ome_msg)
    ome_metadata = img.tag[270][0]
    import xmltodict
    ome_dict = xmltodict.parse(ome_metadata)
    return ome_dict


def get_ome_dict(img_path):
    if hasattr(tifffile, 'tiffcomment'):
        parser = 'tifffile'
        return parser, _get_ome_dict_tifffile(img_path)
    else:
        parser = 'PIL'
        return parser, _get_ome_dict_pil(img_path)


def parse_overlaps(img_path):
    parser, ome_dict = get_ome_dict(img_path)
    # custom_props = ome_dict['OME']['CustomAttributes']['Properties']['prop']
    if parser == 'tifffile':
        custom_props = ome_dict['OME']['CustomAttributes']['PropArray']
        x_overlap = round(custom_props['xyz-Table_X_Overlap']['Value'])
        y_overlap = round(custom_props['xyz-Table_Y_Overlap']['Value'])
    elif parser == 'PIL':
        try:
            custom_props = ome_dict['OME']['ca:CustomAttributes']['PropArray']
            x_overlap = round(float(custom_props['xyz-Table_X_Overlap']['@Value']))
            y_overlap = round(float(custom_props['xyz-Table_Y_Overlap']['@Value']))
        except KeyError:
            custom_props = ome_dict['OME']['ca:CustomAttributes']['Properties']['prop']
            for attr in custom_props:
                if '@label' in attr.keys():
                    if attr['@label'] == 'xyz-Table X Overlap':
                        x_overlap = round(attr['@Value'])
                    elif attr['@label'] == 'xyz-Table X Overlap':
                        y_overlap = round(attr['@Value'])
    else:
        raise ValueError(f'parser type "{parser}" is not recognised')

    # x_overlap = [elt['Value'] for elt in custom_props if elt['label'] == 'xyz-Table X Overlap'][0]
    # y_overlap = [elt['Value'] for elt in custom_props if elt['label'] == 'xyz-Table Y Overlap'][0]
    return x_overlap, y_overlap


def parse_img_shape(img_path):
    parser, ome_dict = get_ome_dict(img_path)
    if parser == 'tifffile':
        return [ome_dict['OME']['Image']['Pixels'][f'Size{ax}'] for ax in ('X', 'Y', 'Z')]
    elif parser == 'PIL':
        return [ome_dict['OME']['Image']['Pixels'][f'@Size{ax}'] for ax in ('X', 'Y', 'Z')]


def parse_img_res(img_path):
    parser, ome_dict = get_ome_dict(img_path)
    if parser == 'tifffile':
        return [ome_dict['OME']['Image']['Pixels'][f'PhysicalSize{ax}'] for ax in ('X', 'Y', 'Z')]
    elif parser == 'PIL':
        return [ome_dict['OME']['Image']['Pixels'][f'@PhysicalSize{ax}'] for ax in ('X', 'Y', 'Z')]


def define_auto_stitching_params(img_path, stitching_cfg):
    overlaps = [stitching_cfg['overlap_x'], stitching_cfg['overlap_y']]
    if any([overlap == 'auto' for overlap in overlaps]):
        parsed_overlaps = parse_overlaps(img_path)
    projection_thickness = stitching_cfg['projection_thickness']
    for i in range(len(overlaps)):  # WARNING: skips Z below but no use
        if overlaps[i] == 'auto':
            overlaps[i] = parsed_overlaps[i]
        if projection_thickness[i] == 'auto':
            projection_thickness[i] = overlaps[i]  # TODO: see if 0.9*overlaps[i] instead
    return overlaps, projection_thickness


def define_auto_resolution(img_path, cfg_res):
    if cfg_res == 'auto':
        cfg_res = ('auto', )*3
    out_res = deepcopy(cfg_res)
    if not cfg_res.count('auto'):
        return out_res

    parsed_res = None
    try:
        parsed_res = parse_img_res(img_path)
    except NotAnOmeFile as e:
        print(str(e))
        print('Defaulting to config values')
    except KeyError as e:
        print(f"Could not find resolution for image {img_path}, defaulting to config")

    if parsed_res is None and cfg_res.count('auto'):
        raise MetadataError(f"Could not determine auto config for file {img_path}")

    for i, ax_res in enumerate(cfg_res):
        if ax_res == 'auto':
            out_res[i] = parsed_res[i]

    return out_res


def get_file_path(cfg, path_name):
    return os.path.join(get_base_dir(cfg), cfg['src_paths'][path_name])


def get_base_dir(cfg):
    return os.path.expanduser(cfg['base_directory'])


##################################################################################################################


def get_tiles_list_from_sample_folder(src_dir: Path, min_file_number=10, tile_extensions=['.ome.tif', '.ome.npy']):
    data_dirs = {}
    for f_name in sorted(src_dir.iterdir()):
        f_path = src_dir / f_name
        if f_path.is_dir():
            for tile_extension in tile_extensions:
                tiles = sorted(f_path.glob(f'*{tile_extension}'))  # , recursive=True)
                if tiles and len(tiles) > min_file_number:
                    data_dirs[f_path] = tiles
                    break  # Only get the first tile extension found
    return data_dirs


def pattern_finders_from_base_dir(src_dir, min_file_number=10, tile_extension=['.ome.tif', '.ome.npy']):
    src_dir = Path(src_dir)
    if not isinstance(tile_extension, (tuple, list)):
        tile_extension = [tile_extension]
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
    return finders


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
            raise ValueError('Must provide either cluster_idx or axis')


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
            raise ValueError(f'No files found in list: {file_names}')
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
