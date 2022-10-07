import numpy as np

from ClearMap.Alignment.Annotation import annotation, __get_module_annotation_file, convert_label
from ClearMap.IO import IO as clearmap_io
from ClearMap.Visualization import Color as col


def color_map(alpha=True, as_int=False, int_type='uint8'):
    """
    Generates a color map from color ids to rgb

    Arguments
    ---------
    alpha : bool
        If True return a color map with alpha values.

    Returns
    -------
    color_map : array
        An array of rgb colors for each label.
    """
    cm = annotation.colors_rgb
    return col.color(cm, alpha=alpha, as_int=as_int, int_type=int_type)


def write_color_palette(filename=None):
    """
    Creates a pal or lut file for Imaris or Imagej based on label colors of atlas.

    Arguments
    ---------
    filename : str
        The name of the color palette file.

    Returns
    -------
    filename : str
        The name of the file to which the color palette was written.
    """

    cm = color_map(alpha=False, as_int=True)

    fext = clearmap_io.file_extension(filename)
    if fext == 'pal':
        col.write_PAL(filename, cm)
    elif fext == 'lut':
        col.write_LUT(filename, cm)
    else:
        raise RuntimeError(f'Color palette format: {fext} not lut or pal')

    return filename


def write_color_annotation(filename, annotation_file=None):
    """Creates a rgb image from the atlas color data.

    Arguments
    ---------
    filename : str
      The name of the color palette file.
    annotation_file : str
      File name of the atlas annotation.

    Returns
    -------
    filename : str
      The name of the file to which the color atlas was written.
    """
    # load atlas and convert to order
    annotation_file = __get_module_annotation_file(annotation_file)
    atlas = np.array(clearmap_io.read(annotation_file), dtype=int)
    atlas = convert_label(atlas, key='id', value='order', method='map')

    # apply color map
    cm = color_map(alpha=False, as_int=True)
    atlas = cm[atlas]

    return clearmap_io.write(filename, atlas)


def convert_label_to_color(label, key='id', level=None, alpha=True, as_int=False, int_type='uint8'):
    """
    Convert label using the atlas annotation colors.

    Arguments
    ---------
    label : array
        List of labels to convert.
    key : str
        The key corresponding to the label.
    level : nt or None
        Convert at this level of the hierarchy. If None use full hierarchy.
    alpha : bool
        If True, return rgba values.
    as_int : bool
        If True, return rgb value as as int between 0 and 255.
    int_type : dtype
        Type for the color array.

    Returns
    -------
    colors : array
        List of colors for each label.
    """
    return annotation.label_to_color(label, key=key, level=level, alpha=alpha, as_int=as_int, int_type=int_type)
