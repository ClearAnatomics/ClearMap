# -*- coding: utf-8 -*-
"""
MHD
===

IO interface to mhd / raw files.

Note
----
Using the function :func:`write_header_from_source` a mhd header for a numpy
binary array can be created to enable loading the numpy file into imagej.
"""
__author__    = 'Christoph Kirst <christoph.kirst.ck@gmail.com>'
__license__   = 'GPLv3 - GNU General Pulic License v3 (see LICENSE.txt)'
__copyright__ = 'Copyright © 2020 by Christoph Kirst'
__webpage__   = 'http://idisco.info'
__download__  = 'http://www.github.com/ChristophKirst/ClearMap2'

import os
import re
import struct

import numpy as np

from ClearMap.IO import IO as clearmap_io


# TODO: implement sources !


# ##############################################################################
# ## Utils
# ##############################################################################

dtype_to_mtype = {
    np.dtype('int8')    : "MET_CHAR",
    np.dtype('uint8')   : "MET_UCHAR",
    np.dtype('int16')   : "MET_SHORT",
    np.dtype('uint16')  : "MET_USHORT",
    np.dtype('int32')   : "MET_INT",
    np.dtype('uint32')  : "MET_UINT",
    np.dtype('int64')   : "MET_LONG",
    np.dtype('uint64')  : "MET_ULONG",
    np.dtype('float32') : "MET_FLOAT",
    np.dtype('float64') : "MET_DOUBLE"
}   

mtype_to_dtype = {v: k for v, k in dtype_to_mtype.items()}

tag_order = [
    'Comment',
    'ObjectType',
    'TransformType',
    'NDims',
    'BinaryData',
    'ElementByteOrderMSB',
    'BinaryDataByteOrderMSB',
    'Color',
    'Position',
    'Offset',
    'Orientation',
    'AnatomicalOrientation',
    'TransformMatrix',
    'CenterOfRotation',
    'CompressedData',
    'CompressedDataSize',
    'DimSize',
    'HeaderSize',
    'Modality',
    'SequenceID',
    'ElementMin',
    'ElementMax',
    'ElementNumberOfChannels',
    'ElementSize',
    'ElementType',
    'ElementSpacing',
    'ElementDataFile'
]


def write_header(filename, mhd_header):
    """Write raw header mhd file.
  
    Arguments
    ---------
    filename : str
        Filename of header.
    mhd_header : dict
       Dictionary of meta data.
  
    Returns
    -------
    filename : str
        The filename of the mhd header.
    """
    header = ''
    for tag in tag_order:
        if tag in mhd_header.keys():
            header += '%s = %s\n' % (tag, mhd_header[tag])
    with open(filename, 'w') as f:
        f.write(header)
    return filename


def header_from_source(source, header=None):
    """Create a mhd header file for a source file.
  
    Arguments
    ---------
    source : Source specification
        Source file or class to create a mhd header file for.
    header : dict or None
       Optional additional entries for the header file.
  
    Returns
    -------
    filename : str
        The filename of the mhd header.
    """
    source = clearmap_io.as_source(source)

    if not isinstance(source, (clearmap_io.mmp.Source,)):
        raise ValueError('The source %r is not valid for mhd header creation!' % (source))

    if source.order not in ['C', 'F']:
        raise ValueError('The source %r is not contiguous!')

    if source.dtype not in dtype_to_mtype.keys():
        raise ValueError('Data type %d of source %r not valid for mhd file format!' % (source.dtype, source))
    mtype = dtype_to_mtype[source.dtype]

    # construct the header info
    mhd_header = {}

    # generic
    mhd_header['ObjectType'] = 'Image'
    mhd_header['BinaryData'] = 'True'
    mhd_header['BinaryDataByteOrderMSB'] = 'False'

    # shape
    mhd_header['NDims'] = '%d' % source.ndim

    shape = source.shape
    if source.order == 'C':
        shape = shape[::-1]
    mhd_header['DimSize'] = ' '.join(['%d' % s for s in shape])

    # data
    mhd_header['ElementType'] = mtype
    mhd_header['HeaderSize'] = '%d' % source.offset
    mhd_header['ElementDataFile'] = os.path.split(os.path.abspath(source.location))[-1]

    if isinstance(header, dict):
        mhd_header.update(header)

    return mhd_header


def write_header_from_source(source, filename=None, header=None):
    """Create a mhd header file for a source file.
  
    Arguments
    ---------
    source : Source specification
        Source file or class to create a mhd header file for.
    filename : str or None
        Filename of the mhd file. If None, the source location with extension 'mhd' is used.
    header : dict or None
        Optional additional entries for the header file.
  
    Returns
    -------
    filename : str
        The filename of the mhd header.
    """
    source = clearmap_io.as_source(source)

    mhd_header = header_from_source(source, header=header)

    if filename is None:
        filename = source.location + '.mhd'

    return write_header(filename, mhd_header)


def write_raw(filename, data):
    """Write the data into a raw format file.

    Arguments
    ---------
    filename : str
        The file name as regular expression.
    data : array
        The data to write in to a raw file.

    Returns
    -------
    filename: str
        The file name of raw file
    """
    with open(filename, 'wb') as raw_file:
        d = len(data.shape)
        if d <= 2:
            data.transpose([1, 0]).tofile(raw_file)
        elif d == 3:
            data.transpose([2, 1, 0]).tofile(raw_file)
        elif d == 4:
            data.transpose([3, 2, 1, 0]).tofile(raw_file)
        else:
            raise NotImplementedError('writeRawData: image dimension %d not supported!' % d)

    return filename


def write(filename, source, header=None, **kwargs):
    """Write  data into to raw/mhd file pair

    Arguments
    ---------
    filename : str
        The file name of the raw/mhd file.
    source : source specification
        The source to write as mhd/raw file.

    Returns
    -------
    filename : str
        The filename of the mhd file.
    """
    fext = clearmap_io.file_extension(filename)
    if fext == "raw":
        header_name = filename[:-3] + 'mhd'
        raw_name = filename
    elif fext == 'mhd':
        header_name = filename
        raw_name = filename[:-3] + 'raw'
    else:
        header_name = filename + '.mhd'
        raw_name = filename + '.raw'

    hdm_header = header_from_source(source, header=header)

    write_header(header_name, hdm_header)
    write_raw(raw_name, source)

    return header_name


# The following reader functions were borrowed from the
# Lasagna project (https://github.com/SainsburyWellcomeCentre/lasagna) with the permission of the author.
# Please note that the built-in reader does not support the compressed raw format.
# If required, please install the vtk library for compressed raw support.

def mhd_read(f_name, fall_back_mode=False):
    """
    Read an MHD file using either VTK (if available) or the slower-built in reader
    if fallBackMode is true we force use of the built-in reader
    ..warning::
        The built in reader does not support the compressed raw format, it requires the vtk library
    """
    if not fall_back_mode:
        # Attempt to load vtk
        try:
            import vtk  # Seems not exist currently for Python 3 (Jan 2017)
            from vtk.util.numpy_support import vtk_to_numpy
        except ImportError:
            print('Optional library VTK was not detected. Falling back to built in (but slower) MHD reader'
                  'If you experience issues loading MHD files, please consider installing VTK.'
                  '(try "conda install -c conda-forge vtk")')
            fall_back_mode = True

    if fall_back_mode:
        return mhd_read_fallback(f_name)
    else:
        # use VTK
        imr = vtk.vtkMetaImageReader()
        imr.SetFileName(f_name)
        imr.Update()

        im = imr.GetOutput()

        rows, cols, z = im.GetDimensions()
        sc = im.GetPointData().GetScalars()
        arr = vtk_to_numpy(sc)
        arr = arr.reshape(z, cols, rows)
        arr = arr.swapaxes(0, 2)  # FIXME: why swap
        print("Using VTK to read MHD image of size: cols: %d, rows: %d, layers: %d" % (rows, cols, z))
        return arr


def mhd_read_fallback(header_file_path):
    """
    Read the header file from the MHA file then use it to
    build a 3D numpy array from the raw file

    Arguments
    ---------
    header_file_path : str
        the path of the mhd (header) file
    """
    if not os.path.exists(header_file_path):
        print('File "{}" not found'.format(header_file_path))
        return
    else:
        hdr = mhd_read_header_file(header_file_path)
        if not hdr:
            print("No data extracted from header file")
            return

    if 'dimsize' not in hdr:
        print("Can not find dimension size information in MHD file. Not importing data")
        return

    if 'elementdatafile' not in hdr:
        print("Can not find the data file as the key 'elementdatafile' does not exist in the MHD file")
        return

    return mhd_read_raw_file(header_file_path, hdr)


def mhd_read_raw_file(header_file_path, header):
    """
    Raw .raw file associated with the MHD header file
    CAUTION: this may not adhere to MHD specs! Report bugs to author.
    """

    if 'headersize' in header and header['headersize'] > 0:
        print("\n\n **MHD reader can not currently cope with header information in .raw file. "
              "Contact the author** \n\n")
        return False

    # Set the endian type correctly
    endian = '<'  # default little endian
    if 'byteorder' in header and header['byteorder'].lower() == 'true':
        endian = '>'  # big endian

    format_type = get_format_type_from_mhd_header(header)
    if not format_type:
        print("\nCan not find data format type in MHD file. **CONTACT AUTHOR**\n")
        return False

    raw_f_path = os.path.join(os.path.dirname(header_file_path), header['elementdatafile'])
    with open(raw_f_path, 'rb') as raw_file:
        data = raw_file.read()

    dim_size = [round(d) for d in header['dimsize']]
    # from: http://stackoverflow.com/questions/26542345/
    # reading-data-from-a-16-bit-unsigned-big-endian-raw-image-file-in-python
    fmt = '{}{}{}'.format(endian, (np.prod(dim_size)), format_type)
    pix = np.asarray(struct.unpack(fmt, data))

    return pix.reshape((dim_size[2], dim_size[1], dim_size[0])).swapaxes(0, 2)  # FIXME: why swapaxes


def get_format_type_from_mhd_header(header):
    DATA_TYPES = {
        'float': 'f',
        'double': 'd',
        'long': 'l',
        'ulong': 'L',
        'char': 'c',
        'uchar': 'B',
        'short': 'h',
        'ushort': 'H',
        'int': 'i',
        'uint': 'I'
    }
    # Set the data type correctly
    if 'datatype' in header:
        datatype = header['datatype'].lower()
        try:
            format_type = DATA_TYPES[datatype]
        except KeyError:
            format_type = False
    else:
        format_type = False

    # If we couldn't find it, look in the ElenentType field
    if not format_type:
        if 'elementtype' in header:
            datatype = header['elementtype'].lower()
            if datatype == 'met_short':
                format_type = 'h'
            else:
                format_type = False
    else:
        format_type = False
    return format_type


def mhd_read_header_file(header_file_path):
    """
    Read an MHD plain text header file and return contents as a dictionary
    """
    def to_num(nb_str):
        if '.' in nb_str:
            return float(nb_str)
        else:
            return int(nb_str)

    with open(header_file_path, 'r') as fid:
        header_lines = fid.read().splitlines()

    hdr_info = {}
    for line in header_lines:
        if not line:
            continue

        m = re.match('\A(\w+)', line)
        if m is None:
            continue

        key = m.groups()[0].lower()  # This is the data key

        # Now we get the data
        m = re.match('\A\w+ *= * (.*) *', line)
        if m is None:
            print("Can not get data for key {}".format(key))
            continue

        if len(m.groups()) > 1:
            print("multiple matches found during mhd_read_header_file. skipping {}".format(key))
            continue

        # If we're here, we found reasonable data
        data = m.groups()[0]

        # If treat any non numerical character as a string and add to the dict
        if re.match('.*[^0-9 \.].*', data) is not None:
            hdr_info[key] = data
            continue

        # Otherwise we have a single number or a list of numbers in space-separated form.
        # So we return these as a list or a single number.
        numbers = [to_num(nb) for nb in data.split(' ') if len(nb) > 0]

        # If the list has just one number we return it directly
        if len(numbers) == 1:
            numbers = numbers[0]

        hdr_info[key] = numbers

    return hdr_info


###############################################################################
# ## Tests
###############################################################################
def _test():
    import numpy as np
    import ClearMap.IO.MHD as mhd

    data = np.array(255 * np.random.rand(200, 500, 20), order='C')
    data = np.array(data, dtype='uint8')

    data[:] = 0
    data[10:25, 5:10, 3:15] = 255

    f_name = 'test.npy'
    np.save(f_name, data)

    h = mhd.header_from_source(f_name, header=None)

    h_name = mhd.write_header_from_source(f_name)
