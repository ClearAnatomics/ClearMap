# -*- coding: utf-8 -*-
"""
TIF module provides interface to read and write tif image files.

Note
----
This module relies on the tifffile library.
"""
__author__ = 'Christoph Kirst <christoph.kirst.ck@gmail.com>, Charly Rousseau <charly.rousseau@icm-institute.org>'
__license__ = 'GPLv3 - GNU General Public License v3 (see LICENSE.txt)'
__copyright__ = 'Copyright © 2020 by Christoph Kirst'
__webpage__ = 'https://idisco.info'
__download__ = 'https://www.github.com/ChristophKirst/ClearMap2'

import warnings
from functools import cached_property

import numpy as np
from tifffile import tifffile

from ClearMap.IO.Source import Source as AbstractSource
from ClearMap.IO.Source import VirtualSource as AbstractVirtualSource
import ClearMap.IO.Slice as cmp_clicing

from ClearMap.Utils.Lazy import lazyattr


###############################################################################
# ## Source class
###############################################################################

class Source(AbstractSource):
    """Class to handle a tif file source

    Note
    ----
    Its assumed that the image data is stored in a series of the tif file.

    .. warning:: It is also assumed that the last 3 dimensions are the image dimensions in the order z,y,x.
    """
    def __init__(self, location, series=0, multi_file=False):
        try:
            self._tif = tifffile.TiffFile(location, multifile=multi_file)
        except TypeError:  # TODO: filter with message
            self._tif = tifffile.TiffFile(location, _multifile=multi_file)  # Versions >= 2020.9.30
        # FIXME: to see if read or write mode, check 1) if the file exists, 2) if there is metadata in the file
        self._series = series
        self.multi_file = multi_file
        if not self.series_mode and not self.pages_mode:
            raise ValueError(f'Unknown metadata type {self._metadata_type}')

    @property
    def name(self):
        return "Tif-Source"

    @lazyattr
    def series(self):
        return self._tif.series[self._series]

    @property
    def is_clearmap(self):
        return self._tif.is_shaped  # Likely written by clearmap if not more metadata

    @property
    def axes_order(self):
        md = self.metadata(info=['shape'])
        return md['order']

    @cached_property
    def series_mode(self):
        return (self._tif._multifile or
                self._metadata_type == 'imagej_metadata' or
                self._metadata_type == 'shaped_metadata' or
                self._metadata_type is None)

    @cached_property
    def pages_mode(self):
        return self._metadata_type == 'ome_metadata'

    @cached_property
    def shape(self):
        try:
            shape_ = shape_from_tif(self.tif_shape)
        except ValueError as err:
            print(f'ERROR transposing shape of {self._tif.filename} ({self.tif_shape}) from {self.axes_order} to ZYX order!')
            raise err
        return shape_

    @cached_property
    def tif_shape(self):
        if self.series_mode:
            shape = self._parse_series_shape()
        else:
            shape = self._parse_pages_shape()
        # Remove empty dimensions
        shape = tuple(dim for dim in shape if dim != 1)
        return shape

    def _parse_series_shape(self):
        return self.series.shape

    def _parse_pages_shape(self):
        _shape = self._tif.pages[0].shape  # The 2D shape of the image
        n_pages = len(self._tif.pages)
        if n_pages > 1:
            _shape = (n_pages,) + _shape
        return _shape

    @cached_property
    def dtype(self):
        if self.series_mode:
            return self.series.dtype
        else:
            return self._tif.pages[0].dtype

    @property
    def location(self):
        return self._tif._fh.path

    @location.setter
    def location(self, value):
        if value != self.location:
            self._tif = tifffile.TiffFile(value, multifile=False)

    @property
    def array(self, processes=None):
        array = self._tif.asarray(maxworkers=processes)
        return self.to_clearmap_order(array)

    @cached_property
    def element_strides(self):
        """The strides of the array elements.

        Returns
        -------
        strides : tuple
            Strides of the array elements.

        Note
        ----
        The strides of the elements module itemsize instead of bytes.
        """
        memmap = self.as_memmap()
        return tuple(s // memmap.itemsize for s in memmap.strides)

    def __getitem__(self, slicing, processes=None):
        ndim = self.ndim
        if ndim >= 3:
            slicing = cmp_clicing.unpack_slicing(slicing, ndim)  # matches dimensions for slicing (may assume space at the end)

            slicing_z = slicing[-1]
            if isinstance(slicing_z, (int, np.int64)):
                slicing_z = int(slicing_z)
            array = self._tif.asarray(key=slicing_z, maxworkers=processes)
            array = self.to_clearmap_order(array)

            slicing_xy = (Ellipsis,) + slicing[-3: -1]  #  Assumes that the last dimensions are space and in the order z,y,x ??
            if len(array.shape) > len(self._tif.pages[0].shape):  # FIXME: is self._tif.pages[0].shape used for series mode?
                slicing_xy = slicing_xy + (slice(None), )
            return array[slicing_xy]
        else:
            array = self._tif.asarray(maxworkers=processes)
            array = self.to_clearmap_order(array)

            return array[slicing]

    def __setitem__(self, *args):
        memmap = self.as_memmap()
        memmap.__setitem__(*args)

    def to_clearmap_order(self, array):
        try:
            transposed = tif_to_clearmap_order(array, self.axes_order)
        except ValueError as err:
            print(f'ERROR transposing array {self._tif.filename} with shape {array.shape} '
                  f'from {self.axes_order} to ClearMap (ZYX) order!')
            raise err
        return transposed

    @cached_property
    def _metadata_type(self):
        populated_metadata = [f'{t}_metadata' for t in self._tif.flags
                              if getattr(self._tif, f'{t}_metadata', None) is not None]
        if not populated_metadata:
            return None
        elif len(populated_metadata) > 1:
            warnings.warn(f'Multiple metadata types found in tif file {self.location}!; metadata: {populated_metadata}')
        return populated_metadata[0]

    def get_raw_metadata_dictionary(self):
        if not self._metadata_type:
            return {}
        md = getattr(self._tif, self._metadata_type) or {}
        if self._tif.is_ome:
            md = tifffile.xml2dict(md).get('OME', {})
        return md

    def metadata(self, info=('shape', 'resolution', 'overlap')):
        """Returns metadata from this tif file.

        Arguments
        ---------
        info : list, 'all' or None
            Optional list of keywords, if 'all' return full tif metadata, if None return default set info.

        Returns
        -------
        metadata : dict
            Dictionary with the metadata.
        """
        metadata = self.get_raw_metadata_dictionary()
        if not metadata:
            warnings.warn(f'No metadata found in tif file {self._tif.filename}!'
                          f'Assuming XYZ order and shape {self.shape}.')
            shape = self.shape
            order = ''.join([ax for i, ax in zip(shape, 'XYZ')])
            return {'shape': shape, 'order': order}

        if info is all:
            raise DeprecationWarning('The all argument is deprecated. Use "all" instead.')
        elif info == 'all':
            return metadata

        if self._metadata_type == 'ome_metadata':
            parser = OMEMetadataParser(self, metadata, info)
        elif self._metadata_type == 'imagej_metadata':
            parser = ImageJMetadataParser(self, metadata, info)
        elif self._metadata_type == 'shaped_metadata':
            parser = ClearMapMetadataParser(self, metadata, info)
        else:
            raise ValueError(f'Unknown metadata type {self._metadata_type}.'
                             f'Please subclass BaseMetadataParser to handle this metadata type.')

        parser.parse()
        return parser.info

    def as_memmap(self):
        try:
            return self.to_clearmap_order(tifffile.memmap(self.location))
        except ValueError as err:
            raise ValueError(f'The tif file {self.location} cannot be memmaped!; {err}')

    def as_virtual(self):
        return VirtualSource(source=self)

    def as_real(self):
        return self

    def as_buffer(self):
        return self.as_memmap()

    # #### Formatting ###
    def __str__(self):
        try:
            name = self.name or ''
            shape = repr((self.shape,)) if self.shape else ''
            dtype = f'[{self.dtype}]' if self.dtype else ''
            order = f'|{self.order}|' if self.order else ''
            location = f'{self.location}' if self.location else ''
            location = location if len(location) <= 100 else location[:50] + '...' + location[-50:]
        except TypeError:
            name = shape = dtype = order = location = ''

        return f'{name}{shape}{dtype}{order}{location}'


class VirtualSource(AbstractVirtualSource):
    def __init__(self, source=None, shape=None, dtype=None, order=None, location=None, name=None):
        super(VirtualSource, self).__init__(source=source, shape=shape, dtype=dtype, order=order, location=location, name=name)
        if isinstance(source, Source):
            self.multi_file = source.multi_file
            self.series = source._series

    @property
    def name(self):
        return 'Virtual-Tif-Source'

    def as_virtual(self):
        return self

    def as_real(self):
        return Source(location=self.location, series=self.series, multi_file=self.multi_file)

    def as_buffer(self):
        return self.as_real().as_buffer()


###############################################################################
# ## TIF Parsers
###############################################################################
class BaseMetadataParser:
    def __init__(self, source, metadata, info_categories):
        self.source = source
        self.metadata = metadata
        if isinstance(info_categories, str):
            info_categories = [info_categories]
        self.info = {k: None for k in info_categories}

    def parse(self):
        if 'shape' in self.info:
            self.parse_pixel_metadata()
        if 'resolution' in self.info:
            self.parse_resolution()
        if 'overlap' in self.info:
            self.parse_overlap()
        if 'description' in self.info:
            self.parse_description()
        if 'tile_configuration' in self.info:
            self.parse_tile_configuration()
        if 'date' in self.info:
            self.parse_date()

    def update_info(self, name, keys, mdict, astype):
        value = []
        for k in keys:
            v = mdict
            for kk in k.split('.'):  # Iterate over dot nested keys
                v = v.get(kk, None)
                if v is None:
                    break
            if v is not None:
                value.append(astype(v))
        if value:
            self.info[name] = tuple(value)

    def parse_order(self):
        self.info['order'] = self.pixels_metadata.get('DimensionOrder', None)

    def parse_shape(self):
        self.update_info('shape', tuple([f'Size{d}' for d in self.info['order']]),
                         self.pixels_metadata, int)
        # Remove empty dimensions
        if self.info['shape'] is not None and [d for d in self.info['shape'] if d != 1] != self.source.shape:
            order = [d for s, d in zip(self.info['shape'], self.info['order']) if s > 1]
            self.info['order'] = order
            self.info['shape'] = self.source.shape

    def parse_pixel_metadata(self):
        """
        Parse only the pixel metadata (i.e. shape and order) from the OME metadata.

        Returns
        -------

        """
        self.parse_order()
        self.parse_shape()

    def parse_resolution(self):
        self.info['resolution'] = tuple(float(self.pixels_metadata[f'PhysicalSize{dim}'])
                                        for dim in self.info['order'] if dim in 'XYZ')

    def parse_date(self):
        self.info['date'] = self.metadata.get('Image', {}).get('CreationDate', None)

    def parse_description(self):
        if self.source.pages_mode:
            desc = self.source._tif.pages[0].description
        else:
            desc = self.source.series.description
        if desc:
            self.info['description'] = desc
        else:
            self.info['description'] = self.metadata.get('Image', {}).get('Description', None)

    def parse_tile_configuration(self):
        warnings.warn(f"Tile configuration parsing is not available for {self.__class__}, skipping!")

    def parse_overlap(self):
        warnings.warn(f"Overlap parsing is not available for {self.__class__}, skipping!")

    @cached_property
    def pixels_metadata(self):
        raise NotImplementedError("Subclasses should implement this method")


class OMEMetadataParser(BaseMetadataParser):
    """
    Parser for OME metadata.
    Typically, this is tuned for the metadata of the OME-TIFF files
    generated by either the UltraMicroscopeII or the Blaze
    light-sheet microscopes from LaVision BioTec (now Miltenyi Biotec).
    """

    @cached_property
    def pixels_metadata(self):
        return self.metadata.get('Image', {}).get('Pixels', {})

    def parse_overlap(self):
        custom_md = self.metadata.get('CustomAttributes', {}).get('PropArray', {})  # UM2
        if custom_md:
            overlap_keys = [f'xyz-Table_{dim}_Overlap.Value' for dim in 'XY']
            self.update_info('overlap', overlap_keys, custom_md, float)
        else:
            custom_md = self.metadata.get('CustomAttributes', {}).get('Properties', {}).get('prop', {})
            overlap_keys = [f'xyz-Table {dim} Overlap' for dim in 'XY']
            overlaps = [float(label.get('Value')) for label in custom_md
                        if label.get('label') in overlap_keys]
            self.info['overlap'] = tuple(overlaps) if overlaps else None

    def parse_tile_configuration(self):
        tile_cfg_txt = (self.metadata.get('Image', {}).get('CustomAttributes', {})
                        .get('TileConfiguration', {}).get('TileConfiguration', ''))
        if tile_cfg_txt:
            tile_cfg_txt = [ln.strip() for ln in tile_cfg_txt[1:].split(')') if ln]
            tile_cfg = [ln.split(';;') for ln in tile_cfg_txt]
            tile_cfg = [(ln[0], ln[1][1:]) for ln in tile_cfg]
            self.info['tile_configuration'] = tile_cfg


class ImageJMetadataParser(BaseMetadataParser):

    @cached_property
    def pixels_metadata(self):
        labels = self.metadata.get('Labels')
        if labels:
            return {(line.split('=', 1)[0]).strip(): (line.split('=', 1)[1]).strip()
                    for line in labels[0].split('\n') if '=' in line}
        else:
            print(f'WARNING: Image: {self.source._tif.filename}, no labels found in imagej metadata!;'
                  f' metadata: {self.metadata}')
            return self.parse_info_field()

    def parse_info_field(self):
        parsed_info = {}
        try:
            md_info = self.metadata['Info'].split('\n')
        except KeyError:
            warnings.warn(f'No Info metadata found in tif file {self.source._tif.filename}!'
                          f'Metadata: {self.metadata}')
            parsed_info['order'] = 'XYZ'
            return parsed_info
        if md_info[0].startswith('NRRD'):  # FIXME: just get pixel metadata
            self.__parse_nrrd(md_info, parsed_info)
        else:
            raise ValueError(f'Unknown metadata type {self.source._metadata_type} and format: {md_info[0]};'
                             f' info: {md_info}')
        return parsed_info

    def __parse_nrrd(self, md_info, parsed_info):
        md_info = {ln.split(':', 1)[0]: ln.split(':', 1)[1]
                   for ln in md_info if ':' in ln}
        if 'sizes' in md_info:
            index_to_dim = {0: 'X', 1: 'Y', 2: 'Z'}
            space_directions = md_info.get('space directions', None)
            space_directions = tuple(tuple([int(dim_) for dim_ in v.strip('()').split(',')])
                                     for v in space_directions.split(' ') if v)
            parsed_info['DimensionOrder'] = [index_to_dim[np.argmax(v)]
                                             for v in space_directions]
            dim_to_index = {d: i for i, d in index_to_dim.items()}
            shape = [int(sz) for sz in md_info['sizes'].split(' ') if sz]
            for ax in parsed_info['DimensionOrder']:
                parsed_info[f'Size{ax}'] = shape[dim_to_index[ax]]
                if ax in 'XYZ':
                    parsed_info[f'PhysicalSize{ax}'] = max(space_directions[dim_to_index[ax]])
        else:
            raise ValueError(f'Unknown metadata type {self.source._metadata_type} and format: {md_info[0]};'
                             f' info: {md_info}')


class ClearMapMetadataParser(BaseMetadataParser):
    """
    Parser for ClearMap metadata.

    Typically, this is tuned for the metadata of the ClearMap generated tif
    files which might lack some of the metadata fields but where the axes
    order is bound to be 'XYZ'.
    """
    def parse_order(self):
        # parsed_info = self.metadata[0]  # FIXME: check, which is best
        super().parse_order()
        if self.info['order'] is None:
            warnings.warn('WARNING: No dimension order found in tif metadata! Assuming "XYZTC" order.')
            self.info['order'] = ''.join([d for d in 'XYZTC'
                                          if f'Size{d}' in self.pixels_metadata.keys()])
            if not self.info['order']:
                self.info['order'] = 'XYZ'


###############################################################################
# ## IO functionality
###############################################################################


def is_tif(source):
    """Checks if this is a TIF source"""
    if isinstance(source, Source):
        return True
    if isinstance(source, str):
        try:
            Source(source)
        except tifffile.TiffFileError:  # Do not catch missing file or permission errors
            return False
        return True
    return False


def read(source, slicing=None, sink=None, **args):
    """Read data from a tif file.

    Arguments
    ---------
    source : str or TIF class
        The name of the tif file or a TIF class.
    slicing : slice, Slice or None
        An optional sub-slice to consider.

    Returns
    -------
    data : array
        The image data in the tif file.
    """
    if not isinstance(source, Source):
        source = Source(source)
    if slicing is None:
        return source.array
    else:
        return source[slicing]


def write(sink, data, **args):
    """Write data to a tif file

    Arguments
    ---------
    sink : str
        The name of the tif file.

    Returns
    -------
    sink : str
        The name of the tif file.
    """
    try:
        data = array_to_tif(data)
    except ValueError as err:
        raise ValueError(f'Cannot write array to tif file {sink}!; {err}')
    # TODO: add axes order 'XYZ(C)' to metadata
    try:
        tifffile.imsave(sink, data, **args)  # noqa
    except AttributeError:
        tifffile.imwrite(sink, data, **args)
    return sink


def create(location=None, shape=None, dtype=None, mode=None, as_source=True, **kwargs):
    """Create a memory map.

    Arguments
    ---------
    location : str
      The filename of the memory mapped array.
    shape : tuple or None
      The shape of the memory map to create.
    dtype : dtype
      The data type of the memory map.
    mode : 'r', 'w', 'w+', None
      The mode to open the memory map.
    as_source : bool
      If True, return as Source class.

    Returns
    -------
    memmap : np.memmap
      The memory map.

    Note
    ----
    By default memmaps are initialized as fortran contiguous if order is None.
    """
    if shape is None:
        raise ValueError('Shape for new tif file must be given!')
    try:
        shape = shape_to_tif(shape)
    except ValueError as err:
        print(f'ERROR: could not transpose shape of {location} ({shape}) to ZYX order!; {err}')
        raise err
    mode = 'r+' if mode == 'w+' or mode is None else mode
    dtype = 'float64' if dtype is None else dtype

    memmap = tifffile.memmap(location, shape=shape, dtype=dtype, mode=mode)
    if as_source:
        return Source(location)
    else:
        return memmap


################################################################################
# ### Array axes order
################################################################################

def map_axes(source_order, dest_order, ndim=None):
    """
    Generate a mapping of axes from source order to destination order.
    The mapping is a tuple of integers that can be used with np.transpose to reorder the axes of an array.

    Parameters
    ----------
    source_order : str
        The order of the axes in the source array (e.g., 'ZYX').
    dest_order : str
        The desired order of axes in the output array (e.g., 'XYZ').

    Returns
    -------
    tuple
        A tuple of integers that can be used with np.transpose to reorder the axes of an array.
    """
    if not ndim:
        ndim = len(source_order)
    if isinstance(source_order, (tuple, list)):
        source_order = ''.join(source_order)
    if isinstance(dest_order, (tuple, list)):
        dest_order = ''.join(dest_order)
    source_order = source_order.upper()
    dest_order = dest_order.upper()

    if 'C' in source_order and 'C' not in dest_order:
        warnings.warn('Color channel is present in source order but not explicitly in destination order, '
                      'assuming it should be moved to the end of the array.')
        dest_order += 'C'
    elif 'C' in dest_order and 'C' not in source_order:
        raise ValueError('Cannot add color channel to array without color channel.')

    # if order were given with too much information, discard by least significant
    for dim in 'TCZ':
        if ((ndim < len(source_order) and dim in source_order) or
                (ndim < len(dest_order) and dim in dest_order)):
            source_order = source_order.replace(dim, '')
            dest_order = dest_order.replace(dim, '')

    if len(source_order) != len(dest_order):
        raise ValueError(f'Source and destination order must have the same number of axes.'
                         f' Source order: {source_order}, Destination order: {dest_order}')

    return tuple(source_order.index(axis) for axis in dest_order)


def transpose_array(array, source_order, dest_order):
    """
    Transpose the axes on array from source_order to dest_order.

    Parameters
    ----------
    array : np.ndarray
        The array to transpose
    source_order : str
        The order of the axes in the array (with letters representing the axes)
    dest_order : str
        The desired order of axes in the output array (with letters representing the axes)

    Returns
    -------
    The transposed array
    """
    transposition = map_axes(source_order, dest_order, ndim=array.ndim)
    return np.transpose(array, transposition)


def tif_to_clearmap_order(array, array_order='XYZ'):
    """
    Transpose the axes on array to the ClearMap order ('ZYX')

    Parameters
    ----------
    array : np.ndarray
        The array to transpose
    array_order : str
        The order of the input array (with letters representing the axes)

    Returns
    -------
    The transposed array
    """
    return transpose_array(array, array_order, 'ZYX')


def reorder_shape(shape, order, dest_order):
    """
    Rearrange the shape tuple to match the given order.

    Parameters
    ----------
    shape : tuple
        The original shape of the array
    order : str
        The order of the axes in the array (e.g., 'ZYX' or 'CZYX')
    dest_order : str
        The desired order of axes in the output array (e.g., 'XYZ' or 'XYZC')

    Returns
    -------
    tuple
        The shape rearranged to match the given order
    """
    axes_mapping = map_axes(order, dest_order, ndim=len(shape))
    rearranged_shape = tuple(shape[ax] for ax in axes_mapping)
    return rearranged_shape


def shape_from_tif(shape, order='XYZ', dest_order='ZYX'):   # FIXME: extract CLEARMAP_ORDER and TIF_ORDER to constants
    """
    Rearrange the shape tuple to match the 'XYZ' (or 'XYZC') order based on the given axes order.

    Parameters
    ----------
    shape : tuple
        The original shape of the array.
    order : str
        The order of the axes in the array (e.g., 'ZYX' or 'CZYX'). Default is 'XYZ' (Default tiff order).
    dest_order : str
        The desired order of axes in the output array (e.g., 'XYZ' or 'XYZC'). Default is 'ZYX' (ClearMap array order).

    Returns
    -------
    tuple
        The shape rearranged to match the 'XYZ' (or 'XYZC') order.
    """
    return reorder_shape(shape, order, dest_order)


def shape_to_tif(shape, order='ZYX', dest_order='XYZ'):
    """
    Rearrange the shape tuple to match the 'XYZ' (or 'XYZC') order based on the given axes order.

    Parameters
    ----------
    shape : tuple
        The original shape of the array.
    order : str
        The order of the axes in the array (e.g., 'ZYX' or 'CZYX'). Default is 'ZYX' (ClearMap array order).
    dest_order : str
        The desired order of axes in the output array (e.g., 'XYZ' or 'XYZC'). Default is 'XYZ' (Default tiff order).

    Returns
    -------
    tuple
        The shape rearranged to match the 'XYZ' (or 'XYZC') order.
    """
    return reorder_shape(shape, order, dest_order)

#
# def shape_from_tif(shape):
#     ndim = len(shape)
#     shape = shape[:max(0, ndim-3)] + shape[-3:][::-1]  # FIXME: rewrite for color and use in array_from_tiff too
#     return shape
#
#
# def array_from_tif(array):  # to ZYXC orientation
#     ndim = array.ndim
#     n_colors = min(array.shape)  # valid if rgb
#     if ndim == 4 and n_colors in (3, 4):  # Put color last for RGB images
#         col_idx = array.shape.index(n_colors)
#
#         new_axes = list(range(ndim))
#         new_axes.pop(col_idx)
#         new_axes = new_axes[::-1]
#         new_axes.append(col_idx)
#         array = np.transpose(array, new_axes)
#     else:
#         axes = list(range(ndim))
#         axes = axes[:max(0, ndim-3)] + axes[-3:][::-1]
#         array = array.transpose(axes)
#     return array

# def array_from_tif(array):  # to ZYXC orientation
#     return tif_to_clearmap_order(array)


def array_to_tif(array, source_order='ZYX'):
    return transpose_array(array, source_order, 'XYZ')


################################################################################
# ### Tests
################################################################################

def _test():
    import ClearMap.Tests.Files as tfs
    import ClearMap.IO.TIF as tif

    filename = tfs.filename('tif_2d')
    t = tif.Source(location=filename)
    print(t)

    filename = tfs.filename('tif_2d_color')
    t = tif.Source(location=filename)
    print(t)

    d = tif.read(filename)
    print(d.shape)

    v = t.as_virtual()
    print(v)

    q = v.as_real()
    print(q)
