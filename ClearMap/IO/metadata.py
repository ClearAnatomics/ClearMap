import os
from copy import deepcopy


import pandas as pd

import tifffile
from PIL import Image


class MetadataError(Exception):
    pass


# class NotAnOmeFile(tifffile.TiffFileError):   Not available in tifffile==0.15.1
class NotAnOmeFile(Exception):
    pass


def _get_ome_dict_tifffile(img_path):  # WARNING: works only with recent versions of tifffile not 0.15.1
    if not tifffile.TiffFile(img_path).is_ome:
        raise NotAnOmeFile("File {} is not a valid ome.tif file".format(img_path))
    ome_metadata = tifffile.tiffcomment(img_path)
    ome_dict = tifffile.xml2dict(ome_metadata)
    return ome_dict


def _get_ome_dict_pil(img_path):
    not_an_ome_msg = "File {} is not a valid ome.tif file".format(img_path)
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
        raise ValueError('parser type "{}" is not recognised'.format(parser))

    # x_overlap = [elt['Value'] for elt in custom_props if elt['label'] == 'xyz-Table X Overlap'][0]
    # y_overlap = [elt['Value'] for elt in custom_props if elt['label'] == 'xyz-Table Y Overlap'][0]
    return x_overlap, y_overlap


def parse_img_shape(img_path):
    parser, ome_dict = get_ome_dict(img_path)
    if parser == 'tifffile':
        return [ome_dict['OME']['Image']['Pixels']['Size{}'.format(ax)] for ax in ('X', 'Y', 'Z')]
    elif parser == 'PIL':
        return [ome_dict['OME']['Image']['Pixels']['@Size{}'.format(ax)] for ax in ('X', 'Y', 'Z')]


def parse_img_res(img_path):
    parser, ome_dict = get_ome_dict(img_path)
    if parser == 'tifffile':
        return [ome_dict['OME']['Image']['Pixels']['PhysicalSize{}'.format(ax)] for ax in ('X', 'Y', 'Z')]
    elif parser == 'PIL':
        return [ome_dict['OME']['Image']['Pixels']['@PhysicalSize{}'.format(ax)] for ax in ('X', 'Y', 'Z')]


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
        print("Could not find resolution for image {}, defaulting to config".format(img_path))

    if parsed_res is None and cfg_res.count('auto'):
        raise MetadataError("Could not determine auto config for file {}".format(img_path))

    for i, ax_res in enumerate(cfg_res):
        if ax_res == 'auto':
            out_res[i] = parsed_res[i]

    return out_res


def get_file_path(cfg, path_name):
    return os.path.join(get_base_dir(cfg), cfg['src_paths'][path_name])


def get_base_dir(cfg):
    return os.path.expanduser(cfg['base_directory'])


##################################################################################################################

def get_ome_tiffs(folder):
    return [f_name for f_name in os.listdir(folder) if f_name.endswith('.ome.tif')]


def get_ome_tiff_list_from_sample_folder(src_dir):
    data_dirs = {}
    for f_name in os.listdir(src_dir):
        f_path = os.path.join(src_dir, f_name)
        if os.path.isdir(f_path):
            ome_tiffs = get_ome_tiffs(f_path)
            if len(ome_tiffs) > 10:
                data_dirs[f_path] = ome_tiffs
    return data_dirs


def pattern_finders_from_base_dir(src_dir):
    data_dirs = get_ome_tiff_list_from_sample_folder(src_dir)
    finders = []
    for path in data_dirs.keys():
        sub_dir = path.replace(src_dir, '')
        sub_dir = sub_dir[1:] if sub_dir.startswith(os.sep) else sub_dir
        finders.append(PatternFinder(sub_dir, data_dirs[path]))
    return finders


class PatternFinder:  # FIXME: needs dir
    def __init__(self, folder, tiff_list):
        self.folder = folder
        self.df = self.get_df_from_file_list(tiff_list)
        self.pattern = Pattern(self.pattern_from_df(self.df))

    def get_df_from_file_list(self, file_names):
        df = pd.DataFrame()
        for f_name in file_names:
            df = pd.concat((df, pd.DataFrame(data=[c for c in f_name]).T))
        return df

    def split_channel(self, channel_order):
        last_channel_idx = self.pattern.digit_clusters[channel_order][-1]
        channel_values = self.df.loc[:, last_channel_idx].sort_values().unique()
        pattern_finders = []
        for channel in channel_values:
            channel_df = self.df[self.df[last_channel_idx == channel]]
            pattern_finders.append(PatternFinder(channel_df))
        return pattern_finders

    def pattern_from_df(self, df):
        pattern = ''
        row1 = df.iloc[0]
        for i, col in enumerate(df):
            if (df[col] == row1[i]).all():
                pattern += row1[i]
            else:
                pattern += '?'
        return self.__fix_pattern(pattern)

    def __fix_pattern(self, pattern):
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
        return '<span style="background-color:#60798B;text-color:#1A72BB">{}</span>'.format('?'*n)

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

# pattern_finders_from_base_dir(cfg)
