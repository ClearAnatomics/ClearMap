import os
from copy import deepcopy

import pandas as pd

import tifffile
from PIL import Image
from tqdm import tqdm

from ClearMap.IO.Workspace import Workspace
from ClearMap.IO import IO as clearmap_io


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
    overlaps = [stitching_cfg['rigid']['overlap_x'], stitching_cfg['rigid']['overlap_y']]
    if any([overlap == 'auto' for overlap in overlaps]):
        parsed_overlaps = parse_overlaps(img_path)
    projection_thickness = stitching_cfg['rigid']['project_thickness']
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

def get_tiles_by_ext(folder, extension='.ome.tif'):
    return [f_name for f_name in sorted(os.listdir(folder)) if f_name.endswith(extension)]


def get_tiles_list_from_sample_folder(src_dir, min_file_number=10, tile_extension='.ome.tif'):
    data_dirs = {}
    for f_name in sorted(os.listdir(src_dir)):
        f_path = os.path.join(src_dir, f_name)
        if os.path.isdir(f_path):
            ome_tiffs = get_tiles_by_ext(f_path, extension=tile_extension)
            if ome_tiffs and len(ome_tiffs) > min_file_number:
                data_dirs[f_path] = ome_tiffs
    return data_dirs


def pattern_finders_from_base_dir(src_dir, axes_order=None, min_file_number=10, tile_extension='.ome.tif'):
    data_dirs = get_tiles_list_from_sample_folder(src_dir, min_file_number=min_file_number,
                                                  tile_extension=tile_extension)
    finders = []
    for path in data_dirs.keys():
        sub_dir = path.replace(src_dir, '')
        sub_dir = sub_dir[1:] if sub_dir.startswith(os.sep) else sub_dir
        tmp = PatternFinder.from_mixed_tiff_lists(os.path.join(src_dir, sub_dir), data_dirs[path], axes_order=axes_order)
        if isinstance(tmp, (tuple, list)):
            finders.extend(tmp)
        else:
            finders.append(tmp)
    return finders


class PatternFinder:  # TODO: from_df class_method
    def __init__(self, folder, tiff_list=None, df=None, axes_order=None):
        self.folder = folder

        if tiff_list is not None:
            self.df = self.get_df_from_file_list(tiff_list)
        elif df is not None:
            self.df = df
        else:
            raise ValueError('Must supply at least tiff_list or df')
        self.pattern = Pattern(self.pattern_from_df(self.df))
        if axes_order is not None:
            self.pattern.axes_order = axes_order

    @property
    def x_values(self):
        return self.df.loc[:, self.pattern.x_rng].drop_duplicates().sort_values(self.pattern.x_rng).values

    @property
    def y_values(self):
        return self.df.loc[:, self.pattern.y_rng].drop_duplicates().sort_values(self.pattern.y_rng).values

    @property
    def z_values(self):
        return self.df.loc[:, self.pattern.z_rng].drop_duplicates().sort_values(self.pattern.z_rng).values

    @property
    def c_values(self):
        return self.df.loc[:, self.pattern.c_rng].drop_duplicates().sort_values(self.pattern.c_rng).values

    @property
    def tiff_list(self):
        return self.get_tiff_list(self.df)

    @property
    def tiff_paths(self):
        return [os.path.join(self.folder, f_name) for f_name in self.tiff_list]

    @classmethod
    def from_mixed_tiff_lists(cls, folder, tiff_list, axes_order=None):
        df = cls.get_df_from_file_list(tiff_list)
        pattern = Pattern(cls.pattern_from_df(df))
        finders = cls.split_channel(folder, df, pattern, axes_order=axes_order)
        if finders is not None:
            return finders
        else:
            print(f'Could not find different channels in Pattern {pattern}')
            return cls(folder, tiff_list, axes_order=axes_order)

    @staticmethod
    def get_df_from_file_list(file_names):
        data = []
        for f_name in file_names:
            data.append([c for c in f_name])
        return pd.DataFrame(data)

    @classmethod
    def split_channel(cls, folder, df, pattern, axes_order=None):
        return cls.split_axis(folder, df, pattern, 'C', axes_order=axes_order)

    @classmethod
    def split_axis(cls, folder, df, pattern, axis_letter, axes_order=None):
        c_idx = [i for i, chunk in enumerate(pattern.chunks) if chunk.endswith(axis_letter)]
        if c_idx:
            cluster_idx = c_idx[0]
        else:
            return
        if axes_order is not None:  # FIXME: not supplied by UI
            for k in axes_order.keys():  # Remove C from axes_order if we split by C
                if axes_order[k] > axes_order['c']:
                    axes_order[k] -= 1
            axes_order['c'] = None
        columns = pattern.digit_clusters[cluster_idx]
        axis_values = df[columns].drop_duplicates().values
        pattern_finders = []
        for axis_val in axis_values:
            sub_df = df.copy()
            for col, v in zip(columns, axis_val):
                sub_df = sub_df[sub_df[col] == v]

            tiff_list = cls.get_tiff_list(sub_df)
            pattern_finders.append(cls(folder, tiff_list, axes_order=axes_order))
        return pattern_finders

    def get_sub_tiff_list(self, x=None, y=None, z=None, c=None):
        df = self.df.copy()
        ranges = (self.pattern.x_rng, self.pattern.y_rng, self.pattern.z_rng, self.pattern.c_rng)
        for axis_val, ax_rng in zip((x, y, z, c), ranges):
            if axis_val is not None:
                for col, v in zip(ax_rng, axis_val):
                    df = df[df[col] == v]
        return self.get_tiff_list(df)

    def get_sub_tiff_paths(self, x=None, y=None, z=None, c=None):
        return [os.path.join(self.folder, f_name) for f_name in self.get_sub_tiff_list(x, y, z, c)]

    def sub_pattern_str(self, x=None, y=None, z=None, c=None):
        return str(PatternFinder(self.folder, tiff_list=self.get_sub_tiff_list(x, y, z, c)).pattern)

    @classmethod
    def get_tiff_list(cls, df):
        expr = ''.join([f'{{0[{col}]}}' for col in df.columns])
        tiff_list = df.agg(expr.format, axis=1)
        return tiff_list

    @staticmethod
    def pattern_from_df(df):
        pattern = ''
        row1 = df.iloc[0]
        for i, col in enumerate(df):
            if (df[col] == row1[i]).all():
                pattern += row1[i]
            else:
                pattern += '?'
        return PatternFinder.__fix_pattern(pattern)

    @staticmethod
    def __fix_pattern(pattern):
        """
        When not all digits are used in a zero padded pattern and were not detected.

        Parameters
        ----------
        pattern

        Returns
        -------

        """
        pattern = list(pattern)
        for i, c in enumerate(pattern[::-1]):
            # print(i, c)
            if c == '?':
                if pattern[::-1][i + 1] == '0':
                    pattern[(len(pattern) - 1) - (i + 1)] = '?'
        return ''.join(pattern)


class Pattern:
    def __init__(self, pattern_str):
        self.chunks = []
        self.digit_clusters = []
        self.pattern_elements = []  # e.g. ['<X,2>', '<Y,2>']
        self.pattern_str = pattern_str
        self.parse_pattern(pattern_str)
        self.axes_order = {'x': None, 'y': None, 'z': None, 'c': None}

    def __str__(self):
        return self.pattern_str

    @property
    def clearmap_pattern(self):
        out = ''
        for i in range(len(self.chunks)):
            out += self.chunks[i]
            if i < len(self.digit_clusters):
                ax = [k for k, v in self.axes_order.items() if v == i][0]
                out += f'<{ax.upper()},{len(self.digit_clusters[i])}>'
        return out

    @property
    def x_order(self):
        return self.axes_order['x']

    @property
    def y_order(self):
        return self.axes_order['y']

    @property
    def z_order(self):
        return self.axes_order['z']

    @property
    def c_order(self):
        return self.axes_order['c']

    @property
    def x_rng(self):
        if self.x_order is None:
            return
        else:
            return self.digit_clusters[self.x_order]

    @property
    def y_rng(self):
        if self.y_order is None:
            return
        else:
            return self.digit_clusters[self.y_order]

    @property
    def z_rng(self):
        if self.z_order is None:
            return
        else:
            return self.digit_clusters[self.z_order]

    @property
    def c_rng(self):
        if self.c_order is None:
            return
        else:
            return self.digit_clusters[self.c_order]

    def get_formatted_pattern(self):
        out = ''
        for i in range(len(self.chunks)):
            out += self.chunks[i]
            if i < len(self.pattern_elements):
                out += self.pattern_elements[i]
        return out

    def get_chars_before_cluster_idx(self, cluster_idx):
        start_idx = self.digit_clusters[cluster_idx][0]
        return self.pattern_str[:start_idx]

    def get_chars_after_cluster_idx(self, cluster_idx):
        end_idx = self.digit_clusters[cluster_idx][-1]
        return self.pattern_str[end_idx+1:]

    def highlight_digits(self, cluster_idx):
        return '{}{}{}'.format(self.get_chars_before_cluster_idx(cluster_idx),
                               self.hightlighted_q_marks(len(self.digit_clusters[cluster_idx])),
                               self.get_chars_after_cluster_idx(cluster_idx))

    def hightlighted_q_marks(self, n):
        return f'<span style="background-color:#60798B;text-color:#1A72BB">{"?" * n}</span>'

    def parse_pattern(self, pattern_str):
        current_chunk = ''
        current_digit_cluster = []
        for i, c in enumerate(pattern_str):
            if c == '?':
                current_digit_cluster.append(i)
                if current_chunk:
                    self.chunks.append(current_chunk)
                    current_chunk = ''
                    continue
            else:
                if current_digit_cluster:
                    self.digit_clusters.append(current_digit_cluster)
                    self.pattern_elements.append('')
                    current_digit_cluster = []
                current_chunk += c
        if current_chunk:
            self.chunks.append(current_chunk)


def stack_tiles_to_columns(sample_folder, axes_order, remove_tiles=False):
    """

    Parameters
    ----------
    sample_folder str:
        The folder where the data is located
    axes_order dict:
        A dictionary of the type {'x': 1, 'y': 0, 'z': 3, 'c': 2} indicating the order of each axis
    remove_tiles bool:
        Whether to remove the individual tiles at the end

    Returns
    -------

    """
    pattern_finders = pattern_finders_from_base_dir(sample_folder, axes_order=axes_order)
    images_to_remove = []
    for pat_finder in tqdm(pattern_finders):
        for y in tqdm(pat_finder.y_values):
            for x in tqdm(pat_finder.x_values, leave=False):
                img_paths = pat_finder.get_sub_tiff_paths(x=x, y=y)
                sub_pat = PatternFinder(folder=sample_folder, tiff_list=img_paths,
                                        axes_order=pat_finder.pattern.axes_order)
                ws = Workspace('CellMap', directory=sub_pat.folder)
                ws.update({'raw': sub_pat.pattern.clearmap_pattern.replace(sub_pat.folder, '')})
                # ws.info()
                new_path = pat_finder.sub_pattern_str(x=x, y=y)
                new_path = new_path.replace('_xyz-Table Z????', '')
                new_path = os.path.join(pat_finder.folder, new_path)
                clearmap_io.convert(ws.source('raw'), new_path)
                images_to_remove.extend(img_paths)

    with open('/tmp/file_to_rm.txt', 'a') as handle:
        handle.write('\n'.join(images_to_remove))
    if remove_tiles:
        for f_path in images_to_remove:
            os.remove(f_path)
