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
__author__ = 'Christoph Kirst <christoph.kirst.ck@gmail.com>'
__license__ = 'GPLv3 - GNU General Pulic License v3 (see LICENSE.txt)'
__copyright__ = 'Copyright © 2022 by Christoph Kirst'
__webpage__ = 'https://idisco.info'
__download__ = 'https://github.com/ClearAnatomics/ClearMap'

import os
import numpy as np
import zlib

import ClearMap.IO.Source as src
from ClearMap.IO import IO as clearmap_io
from ClearMap.IO.FileUtils import file_extension, is_file


###############################################################################
# Source class
###############################################################################

class Source(src.Source):
    """Mhd/raw array source."""

    def __init__(self, location, name=None):
        """Mhd source class constructor.

        Arguments
        ---------
        location : str
          The file name of the mhd source.
        """
        super().__init__(name=name)
        self._location = _header_file(location)
        self._memmap = None
        self._array = None
        self._init_data()

    def _init_data(self):
        if self._memmap is None:
            try:
                self._memmap = _memmap(self._location)
            except:
                self._memmap = None
        if self._memmap is None and self._array is None:
            try:
                self._array = _array(self._location)
            except:
                self._array = None

    @property
    def name(self):
        return "Mhd-Source"

    @property
    def location(self):
        return self._location

    @location.setter
    def location(self, value):
        if value != self.location:
            self._location = value
            self._init_data()

    @property
    def array(self):
        """The underlying data array.

        Returns
        -------
        array : array
          The underlying data array of this source.
        """
        if self._array is not None:
            return self._array
        elif self._memmap is not None:
            return np.array(self._memmap)
        else:
            return _array(self._location)

    @array.setter
    def array(self, value):
        header, header_file, raw_file = \
            _header_from_array(value, location=self._location, return_header_and_raw_file=True)
        _write_header(header_file, header)
        _write_raw(raw_file, value, compression=_compression_from_header(header))
        self._init_data()

    @property
    def shape(self):
        """The shape of the source.

        Returns
        -------
        shape : tuple
          The shape of the source.
        """
        if self._array is not None:
            return self._array.shape
        elif self._memmap is not None:
            return self._memmap.shape
        else:
            return _shape(self._location)

    @shape.setter
    def shape(self, value):
        raise NotImplementedError('Cannot set shape of mhd file')

    @property
    def dtype(self):
        """The data type of the source.

        Returns
        -------
        dtype : dtype
          The data type of the source.
        """
        if self._array is not None:
            return self._array.dtype
        elif self._memmap is not None:
            return self._memmap.dtype
        else:
            return _dtype(self._location)

    @dtype.setter
    def dtype(self, value):
        raise NotImplementedError('Cannot set dtype of mhd file')

    @property
    def order(self):
        """The order of how the data is stored in the source.

        Returns
        -------
        order : str
          Returns 'C' for C contiguous and 'F' for fortran contiguous, None otherwise.
        """
        return _order(self.location)

    @order.setter
    def order(self, value):
        raise NotImplementedError('Cannot set order of mhd file')

    @property
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
        self._init_data()
        if self._array is not None:
            source = self._array
        elif self._memmap is not None:
            source = self._memmap
        else:
            raise ValueError('Cant determine strides for source without data.')

        return tuple(s // source.itemsize for s in source.strides)

    @property
    def offset(self):
        """The offset of the memory map in the file.

        Returns
        -------
        offset : int
          Offset of the memory map in the file.
        """
        return _offset(self.location)

    # Data
    def __getitem__(self, *args):
        self._init_data()
        if self._memmap is not None:
            return self._memmap.__getitem__(*args)
        elif self._array is not None:
            return self._array.__getitem__(*args)
        else:
            return _array(self.location).__getitem__(*args)

    def __setitem__(self, *args):
        if self._memmap is None:
            self._memmap = _memmap(self.location)
        self._memmap.__setitem__(*args)

    def metadata(self, info=None):
        """Returns metadata from this mhd file.

        Arguments
        ---------
        info : list or all
          Optional list of keywords, if all return full tif metadata, if None return default set info.

        Returns
        -------
        metadata : dict
          Dictionary with the metadata.
        """
        header = _read_header(self.location)
        if isinstance(info, list):
            header = {key: header[key] for key in info}
        return header

    def as_array(self):
        return self.array

    def as_memmap(self):
        if self._memmap is None:
            self._memmap = _memmap(self.location)
        return self._memmap

    def as_virtual(self):
        return VirtualSource(source=self)

    def as_real(self):
        return self

    def as_buffer(self):
        return self.as_memmap()

    # Formatting
    def __str__(self):
        try:
            name = self.name
            name = f'{name}' if name is not None else ''
        except:
            name = ''

        try:
            shape = self.shape
            shape = f'{shape}' if shape is not None else ''
        except:
            shape = ''

        try:
            dtype = self.dtype
            dtype = f'[{dtype}]' if dtype is not None else ''
        except:
            dtype = ''

        try:
            order = self.order
            order = f'|{order}|' if order is not None else ''
        except:
            order = ''

        try:
            location = self.location
            location = f'{location}' if location is not None else ''
            if len(location) > 100:
                location = location[:50] + '...' + location[-50:]
            if len(location) > 0:
                location = f'{{{location}}}'
        except:
            location = ''

        return name + shape + dtype + order + location


class VirtualSource(src.VirtualSource):
    def __init__(self, source=None, shape=None, dtype=None, order=None, location=None, name=None):
        super(VirtualSource, self).__init__(source=source, shape=shape, dtype=dtype, order=order, location=location,
                                            name=name)

    @property
    def name(self):
        return 'Virtual-Mhd-Source'

    def as_virtual(self):
        return self

    def as_real(self):
        return Source(location=self.location)

    def as_buffer(self):
        return self.as_real().as_buffer()


###############################################################################
# IO Interface
###############################################################################

def is_mhd(source):
    """Checks if this source is an MHD source."""
    if isinstance(source, Source):
        return True
    if isinstance(source, str) and source[-3:] in ('raw', 'mhd'):
        header_file = _header_file(source)
        try:
            Source(header_file)
        except:
            return False
        return True
    return False


def read(source, slicing=None, **kwargs):
    """Read data from a mhd file.

    Arguments
    ---------
    source : str
      The name of the mhd file.
    slicing : slice, Slice or None
      An optional sub-slice to consider.

    Returns
    -------
    array : array
      The image data in the tif file as a buffer.
    """
    if not isinstance(source, Source):
        source = Source(source)
    if slicing is None:
        return source.array
    else:
        return source.__getitem__(slicing)


def write(sink, data, slicing=None, **kwargs):
    """Write specialization for mhd files."""
    # Note: data is ClearMap Source
    if isinstance(sink, Source):
        sink.__setitem__(slicing, data.array)
    elif isinstance(sink, str):
        # header file
        shape = data.shape
        dtype = data.dtype

        header_file, raw_file = _header_and_raw_file(location=sink)
        if is_file(header_file):
            header = _read_header(header_file)
            shape = _from_mhd['DimSize'](header['DimSize'])
            dtype = _from_mhd['ElementType'](header['ElementType'])
            raw_file = _raw_file_from_header(header, header_file=header_file)
        elif slicing is not None:
            raise ValueError('Cannot write sliced data into non-existent source %r!' % (sink,))
        else:  # create header file
            header, header_file, raw_file = \
                _header_from_array(array=data, location=sink, return_header_and_raw_file=True)
            _write_header(header_file, header)

        if dtype != data.dtype:
            raise ValueError('Type %r and array type %r mismatch!' % (dtype, data.dtype))
        if slicing is None and shape != data.shape:
            raise ValueError('Shape %r and array shape %r mismatch!' % (shape, data.shape))

        # raw file
        if slicing is None:
            _write_raw(raw_file, data.array, compression=_compression_from_header(header))
        else:
            memmap = _memmap(header_file)
            memmap[slicing] = data.array
    else:
        raise ValueError('Invalid sink specification %r' % sink)

    return sink


def create(location=None, shape=None, dtype=None, array=None, as_source=True, **kwargs):
    header, header_file, raw_file = \
        _header(location=location, shape=shape, dtype=dtype, array=array, return_header_and_raw_file=True)
    _write_header(header_file, header)
    if array is not None:
        _write_raw(raw_file, array, compression=_compression_from_header(header))
    if as_source:
        return Source(header_file)
    else:
        return header_file

###############################################################################
# Utils
###############################################################################


_dtype_to_mhd_type = {
    np.dtype('int8'): "MET_CHAR",
    np.dtype('uint8'): "MET_UCHAR",
    np.dtype('int16'): "MET_SHORT",
    np.dtype('uint16'): "MET_USHORT",
    np.dtype('int32'): "MET_INT",
    np.dtype('uint32'): "MET_UINT",
    np.dtype('int64'): "MET_LONG",
    np.dtype('uint64'): "MET_ULONG",
    np.dtype('float32'): "MET_FLOAT",
    np.dtype('float64'): "MET_DOUBLE"
}

_mhd_type_to_dtype = {v: k for k, v in _dtype_to_mhd_type.items()}


def _from_mhd_bool(text):
    return text == 'True'


def _to_mhd_vector(item):
    return ' '.join(str(x) for x in item)


def _from_mhd_vector(text, convert=float):
    return tuple(convert(x) for x in text.split())


def _to_mhd_shape(shape):
    return _to_mhd_vector(shape)


def _from_mhd_shape(text):
    shape = _from_mhd_vector(text, convert=int)
    return shape


def _to_mhd_type(dtype):
    if dtype not in _dtype_to_mhd_type.keys():
        raise NotImplementedError(f'Data type {dtype} not supported!')
    return _dtype_to_mhd_type[dtype]


def _from_mhd_type(text):
    if text not in _mhd_type_to_dtype.keys():
        raise NotImplementedError(f'Data type {text} not supported!')
    return _mhd_type_to_dtype[text]


def _from_mhd_data_file(text, header_file=None):
    if 'LIST' in text:
        raise NotImplementedError("File lists not supported.")
    if header_file is not None:
        raw_file = os.path.join(os.path.split(header_file)[0], text)
    else:
        raw_file = text
    return raw_file


def _to_mhd_data_file(filename):
    if filename is None:
        return None
    return os.path.split(filename)[1]


_to_mhd = {
    'DimSize': _to_mhd_shape,
    'NDims': str,
    'ElementType': _to_mhd_type,
    'HeaderSize': str,
    'ElementDataFile': _to_mhd_data_file,
    'Offset': _to_mhd_vector,
    'TransformMatrix': _to_mhd_vector,
    'CenterOfRotation': _to_mhd_vector,
    'ElementSpacing': _to_mhd_vector,
    'CompressedData': str,
    'BinaryData': str,
    'BinaryDataByteOrderMSB': str,
    'ObjectType': str,
}


_from_mhd = {
    'DimSize': _from_mhd_shape,
    'NDims': int,
    'ElementType': _from_mhd_type,
    'HeaderSize': int,
    'ElementDataFile': _from_mhd_data_file,
    'Offset': _from_mhd_vector,
    'TransformMatrix': _from_mhd_vector,
    'CenterOfRotation': _from_mhd_vector,
    'ElementSpacing': _from_mhd_vector,
    'CompressedData': _from_mhd_bool,
    'BinaryData': _from_mhd_bool,
    'BinaryDataByteOrderMSB': _from_mhd_bool,
    'ObjectType': str,
}


_mhd_key_order = [
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


def _compression_from_header(header=None):
    compression = False
    if header is not None and 'CompressedData' in header.keys():
        compression = _from_mhd['CompressedData'](header['CompressedData'])
    return compression


def _header_file(filename):
    """Returns the header file name for the mhd/raw file."""
    fext = file_extension(filename)

    # REFACTOR: use
    #  base, ext = os.path.splitext(filename)
    #  extensions = {'raw': '.mhd', 'zraw': '.mhd', 'mhd': '.mhd'}
    #  new_ext = extensions.get(ext, '.mhd')
    #  header_file = base + new_ext

    if fext == "raw":
        header_file = filename[:-3] + 'mhd'
    elif fext == 'zraw':
        header_file = filename[:-4] + 'mhd'
    elif fext == 'mhd':
        header_file = filename
    else:
        header_file = filename + '.mhd'

    return header_file


def _raw_file(filename, compression=False):  # FIXME:
    """Returns the raw file name for the mhd/raw file."""
    file_ext = file_extension(filename)
    raw = 'raw' if not compression else 'zraw'
    if file_ext == "mhd":
        file_ext = file_extension(filename[:-4])  # FIXME: .split('.')[-1] or[:-len(file_ext)]
        if file_ext is not None:  # mhd header for a different binary file format.
            raw_file = filename[:-4]
        else:
            raw_file = filename[:-3] + raw
    elif file_ext in ['raw', 'zraw']:
        raw_file = filename
    else:  # another file binary for which we want a mhd header.
        raw_file = filename

    return raw_file


def _header_file_from_header(header, file_path=None):
    """Returns the header file name from the header."""
    if header is not None and 'ElementDataFile' in header.keys():
        raw_file = header['ElementDataFile']
        raw_file = _from_mhd_data_file(raw_file, header_file=file_path)
        header_file = _header_file(raw_file)
    else:
        raise ValueError('Cannot infer header file name.')

    return header_file


def _raw_file_from_header(header, header_file=None):
    """Returns the raw file name from the header file name and header."""
    if header is not None and 'ElementDataFile' in header.keys():
        raw_file = _from_mhd_data_file(header['ElementDataFile'], header_file=header_file)
    elif header_file is not None:
        compression = _compression_from_header(header)
        raw_file = _raw_file(header_file, compression=compression)
    else:
        raise ValueError('filename or header must be given')

    return raw_file


def _header_and_raw_file(location=None, header_file=None, raw_file=None, header=None):
    """Determine header and raw file from specifications."""
    # header file
    if header_file is None:
        if location is not None:
            header_file = _header_file(location)
        elif raw_file is not None:
            header_file = _header_file(raw_file)
        elif header is not None:
            header_file = _header_file_from_header(header)
    elif header_file is not None:
        header_file = _header_file(header_file)

    # raw file
    if raw_file is None:
        if header is not None:  # try header first as this might differ from standard conventions for raw/mhd naming
            try:
                raw_file = _raw_file_from_header(header, header_file=header_file)
            except:
                pass
    if raw_file is None:
        if location is not None:
            raw_file = _raw_file(location)
        if header_file is not None:
            raw_file = _raw_file(header_file)

    return header_file, raw_file


def _header(location=None, shape=None, dtype=None, array=None, offset=None,
            header_file=None, raw_file=None, compression=None, header=None,
            return_header_and_raw_file=False, **kwargs):
    """Create header dictionary from array specifications."""
    header_file, raw_file = _header_and_raw_file(location=location, header_file=header_file,
                                                 raw_file=raw_file, header=header)

    if dtype is None:
        if array is not None:
            dtype = array.dtype
    if dtype is not None:
        if dtype not in _dtype_to_mhd_type.keys():
            raise ValueError(f'Data type {dtype} not supported for mhd files!')

    if shape is None:
        if array is not None:
            shape = array.shape

    # construct the header info
    mhd_header = {
        'ObjectType': 'Image',
        'BinaryData': 'True',
        'BinaryDataByteOrderMSB': 'False'
    }

    # shape
    if shape is not None:
        mhd_header['NDims'] = _to_mhd['NDims'](len(shape))
        mhd_header['DimSize'] = _to_mhd['DimSize'](shape)

    # data
    if dtype is not None:
        mhd_header['ElementType'] = _to_mhd['ElementType'](dtype)
    if offset is not None:
        mhd_header['HeaderSize'] = _to_mhd['HeaderSize'](offset)

    if raw_file is not None:
        mhd_header['ElementDataFile'] = _to_mhd['ElementDataFile'](raw_file)

    if compression is not None:
        mhd_header['CompressedData'] = _to_mhd['CompressedData'](compression)

    if header:
        mhd_header.update(header)

    if return_header_and_raw_file:
        return mhd_header, header_file, raw_file
    else:
        return mhd_header


def _write_header(filename, header):
    """Write mhd header file."""
    mhd_header = ''
    for key in _mhd_key_order:
        if key in header.keys():
            mhd_header += f'{key} = {header[key]}\n'

    with open(filename, 'w') as file:
        file.write(mhd_header)

    return filename


def _read_header(filename):
    """Read mhd header file."""
    header = {}
    with open(filename, 'r') as header_file:
        for line in header_file:
            line = line.strip()
            if line:
                line = [x.strip() for x in line.split('=')]
                if len(line) < 2:
                    line += [None]

                key, value = line[:2]
                header[key] = value

    return header


def _dtype(filename):
    """Determine data type from mhd file."""
    header = _read_header(filename)
    return _from_mhd['ElementType'](header['ElementType'])


def _shape(filename):
    """Determine data type from mhd file."""
    header = _read_header(filename)
    return _from_mhd['DimSize'](header['DimSize'])


def _order(filename):
    """Determine shape from mhd file."""
    # Note: this fixes the order how to store data read from disk in memory.
    return 'F'


def _offset(filename):
    """Offset of data in file."""
    header = _read_header(filename)
    return _from_mhd['HeaderSize'](header['HeaderSize'])


def _read_info(filename):
    """Returns all information to read data."""
    header_file = _header_file(filename)
    header = _read_header(header_file)

    order = 'F'  # FIXME: unused because overwritten below
    dtype = _from_mhd['ElementType'](header['ElementType'])
    shape = _from_mhd['DimSize'](header['DimSize'])

    if 'HeaderSize' in header.keys():
        offset = _from_mhd['HeaderSize'](header['HeaderSize'])
    else:
        offset = 0

    compression = _compression_from_header(header)

    raw_file = _raw_file_from_header(header, header_file=header_file)

    # transposition of axes as data is expected to be written in Fortran order on disk
    order = 'C'  # transpose will turn into 'F' order as default in ClearMap.
    shape = tuple(reversed(shape))
    transpose = tuple(reversed(range(len(shape))))

    # special case of npy files accesed via mhd header
    if file_extension(raw_file) == 'npy':
        transpose = tuple(range(len(shape)))

    return raw_file, shape, dtype, order, offset, compression, transpose


def _array(filename):  # TODO: vtk version if import succeeds
    """Read the actual data into a numpy array."""
    raw_filename, shape, dtype, order, offset, compression, transpose = _read_info(filename)

    with open(raw_filename, 'rb') as file:
        if compression:
            data = np.frombuffer(zlib.decompress(file.read()), dtype=dtype)
        else:
            data = np.asarray(np.fromfile(file, dtype=dtype, offset=offset), order=order, dtype=dtype)
        data = data.reshape(shape).transpose(transpose)

    return data


def _memmap(filename, mode=None):
    """Create memmap to the mhd data."""
    raw_filename, shape, dtype, order, offset, compression, transpose = _read_info(filename)
    if compression:
        raise ValueError('Cannot create memmap to compressed mhd/raw file.')
    mode = 'r+' if mode is None else mode
    return np.memmap(raw_filename, dtype=dtype, mode=mode, offset=offset,
                     shape=shape, order=order).transpose(transpose)


def _write_raw(filename, array, compression=False):
    """Write the data into a raw format file."""
    if not isinstance(array, np.ndarray):
        array = array.as_buffer()
    array = array.transpose(tuple(reversed(range(array.ndim))))

    with open(filename, 'wb') as raw_file:
        if compression:
            compressed = zlib.compress(array.tostring())
            raw_file.write(compressed)
        else:
            array.tofile(raw_file)  # tofile writes in C order !

    return filename


def _header_from_array(array, location=None, header=None, return_header_and_raw_file=False):
    if array.dtype not in _dtype_to_mhd_type.keys():
        raise ValueError(f'Data type {array.dtype} of array not valid for mhd file format!')

    return _header(location=location, shape=array.shape, dtype=array.dtype, offset=0,
                   header=header, return_header_and_raw_file=return_header_and_raw_file)


###############################################################################
# Utils
###############################################################################

def header_from_source(source=None, location=None, header=None, return_header_and_raw_file=False):
    """Create a mhd header dictionary for a source.

    Arguments
    ---------
    source : Source specification
        Source file or class to create a mhd header file for.
    location : str or None
        Location of the mhd file. If None, the source location with extension 'mhd' is used.
    header : dict or None
        Optional additional entries for the header file.
    return_header_and_raw_file : bool
        If True also return the header file.

    Returns
    -------
    header : dict
        The mhd header dictionary.

    Note
    ----
    This function can be used to make npy files readable in image processing software tools that
    allow to import mhd files, such ash ImageJ.
    """
    if source is None and location is not None:
        source = location

    source = clearmap_io.as_source(source)

    if location is None and source.location is not None:
        location = source.location

    if source.dtype not in _dtype_to_mhd_type.keys():
        raise ValueError(f'Data type {source.dtype} of source {source} not valid for mhd file format!')

    # handle special sources
    shape = source.shape
    if location is not None:
        ext = file_extension(location)
        if ext == 'npy':
            # npy files are not transposed when written
            shape = shape[::-1]

    return _header(location=location, shape=shape, dtype=source.dtype, offset=source.offset,
                   header=header, return_header_and_raw_file=return_header_and_raw_file)


def write_header_from_source(source, location=None, header=None):
    """Create a mhd header file for a source file.

    Arguments
    ---------
    source : Source specification
        Source file or class to create a mhd header file for.
    location : str or None
        Location of the mhd file. If None, the source location with extension 'mhd' is used.
    header : dict or None
        Optional additional entries for the header file.

    Returns
    -------
    filename : str
        The filename of the mhd header.
    """
    header, header_file, raw_file = header_from_source(source, location=location,
                                                       header=header, return_header_and_raw_file=True)
    return _write_header(header_file, header)


###############################################################################
# Test
###############################################################################

def _test():
    import numpy as np
    import ClearMap.IO.MHD as mhd
    from importlib import reload
    reload(mhd)

    data = np.array(255 * np.random.rand(20, 30, 40), order='C')
    data = np.array(data, dtype='uint8')
    data[:5,:10,:15] = 0

    mhd_shape = mhd._to_mhd_shape(data.shape)
    print(mhd_shape)
    shape = mhd._from_mhd_shape(mhd_shape)
    print(shape, data.shape)

    header, header_file, raw_file = mhd.header_from_source(data, location='test.mhd', return_header_and_raw_file=True)
    print(header)
    print(header_file, raw_file)
    mhd._write_header(header_file, header)
    mhd._write_raw(raw_file, data)

    array = mhd._array('test.mhd')
    print(array.shape, data.shape)
    np.all(array == data)

    memmap = mhd._memmap('test.mhd')
    np.all(memmap == data)

    mhd.create(location='test.mhd', array=data)

    source = mhd.Source('test.mhd')
    print(source)
    np.all(data == source.array)

    #reload(mhd)
    fname = 'test.npy'
    np.save(fname, data)

    header, header_file, raw_file = mhd.header_from_source(fname, header=None, return_header_and_raw_file=True)
    print(header)
    print(header_file, raw_file)

    mhd._write_header(header_file, header)
    source = mhd.Source(header_file)
    print(source)
    np.all(data == source.array)

    header_file = mhd.write_header_from_source('test.npy')
    source = mhd.Source(header_file)
    print(source)
    np.all(data == source.array)

    import ClearMap.IO.IO as io
    io.write('test.mhd', data)

    source = mhd.Source('test.mhd')
    print(source)
    np.all(data == source.array)

    import ClearMap.IO.FileUtils as fu
    for file in ['test.mhd', 'test.raw', 'test.npy', 'test.npy.mhd']:
        fu.delete_file(file)


    data_path = os.path.expanduser('~/Science/Projects/WholeBrainClearing/Axons/Analysis/AxonMap/data/'
                                   '2.Test_data_brain_regions/elastix_auto_to_reference/')
    data_file = os.path.join(data_path, 'result.0.mhd')
    #data_file = os.path.join(data_path, 'result.0.zraw')

    #source = mhd.Source(data_file)
    #array = source.array