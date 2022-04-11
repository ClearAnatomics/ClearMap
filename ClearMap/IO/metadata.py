import os
from copy import deepcopy

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
