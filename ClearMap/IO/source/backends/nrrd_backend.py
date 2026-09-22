# encoding: utf-8
"""
NRRD
====

IO interface to NRRD volumetric image data files.

Note
----
    The interface is based on nrrd.py for reading and writing nrrd files.
    See 'this link <https://teem.sourceforge.net/nrrd/format.html>`_ for
    specifications.
"""
__author__    = 'Christoph Kirst <christoph.kirst.ck@gmail.com>'
__license__   = 'GPLv3 - GNU General Public License v3 (see LICENSE.txt)'
__copyright__ = 'Copyright © 2020 by Christoph Kirst'
__webpage__   = 'https://idisco.info'
__download__  = 'https://www.github.com/ChristophKirst/ClearMap2'

import os
import gzip
import bz2
import datetime 

import numpy as np

import ClearMap.IO.source.Source as source_mod
from ClearMap.IO.source.Source import ReprFields as RF
from ClearMap.Utils.exceptions import NrrdError, ClearMapPermissionError


###############################################################################
### NrrdSource class
###############################################################################

class NrrdSource(source_mod.Source):
    """Nrrd array source."""
    _CACHED_PROPERTIES = ('_header', 'shape', 'dtype', 'offset')
    _REPR_FIELDS = (RF.NAME, RF.SHAPE, RF.DTYPE, RF.ORDER, RF.LOCATION)
  
    def __init__(self, location, mode=None):
        """Nrrd source class constructor.

        Arguments
        ---------
        location : str
            The file nameof the nrrd source.
        """
        super().__init__(mode=mode)
        self._location = location

    @property
    def location(self):
        return self._location

    @location.setter
    def location(self, value):
        if value != self.location:
            self._location = value

    @property
    def array(self):
        """The underlying data array.

        Returns
        -------
        array : array
              The underlying data array of this source.
        """
        return _array(self.location)

    @array.setter
    def array(self, value):
        _write_data(self.location, value)

    @property
    def shape(self):
        """The shape of the source.

        Returns
        -------
        shape : tuple
            The shape of the source.
        """
        return _shape(self.location)

    @shape.setter
    def shape(self, value):
        #TODO: fix
        raise NotImplementedError('Cannot set shape of nrrd file')

    @property
    def dtype(self):
        """The data type of the source.

        Returns
        -------
        dtype : dtype
            The data type of the source.
        """
        return _dtype(self.location)

    @dtype.setter
    def dtype(self, value):
        #TODO: fix
        raise NotImplementedError('Cannot set dtype of nrrd file')

    @property
    def order(self):
        """The order of how the data is stored in the source.

        Returns
        -------
        order : str
            Returns 'C' for C contigous and 'F' for fortran contigous, None otherwise.
        """
        return _order(self.location)

    @order.setter
    def order(self, value):
        #TODO: fix
        raise NotImplementedError('Cannot set order of nrrd file')

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
        memmap = _memmap(self.location)
        return  tuple(s // memmap.itemsize for s in memmap.strides)

    @property
    def offset(self):
        """The offset of the memory map in the file.

        Returns
        -------
        offset : int
            Offset of the memeory map in the file.
        """
        return _offset(self.location)

    ### Data
    def __getitem__(self, *args):
        memmap = _memmap(self.location)
        return memmap.__getitem__(*args)

    def __setitem__(self, *args):
        if not self.is_writable:
            raise ClearMapPermissionError(f'NrrdSource {self} was opened read-only (mode="r"). '
                                          f'Use io.edit() to open for writing.')
        memmap = _memmap(self.location)
        memmap.__setitem__(*args)

    def metadata(self, info = None):
        """Returns metadata from this nrrd file.

        Arguments
        ---------
        info : list or all
            Optional list of keywords, if all return full tif metadata, if None return default set info.

        Returns
        -------
        metadata : dict
            Dictionary with the meta data.
        """
        return _read_header(self.location)

    def as_memmap(self):
        return _memmap(self.location)

    def as_virtual(self):
        return NrrdVirtualSource(source = self)

    def as_real(self):
        return self

    def as_buffer(self):
        return self.as_memmap()


class NrrdVirtualSource(source_mod.VirtualSource):
    _real_class = NrrdSource

    def __init__(self, source=None, shape=None, dtype=None,
                 order=None, location=None, name=None, mode=None):
      super().__init__(source=source, shape=shape, dtype=dtype, order=order,
                       location=location, name=name, mode=mode)
      if isinstance(source, NrrdSource):
          self.location = source.location


###############################################################################
### IO Interface
###############################################################################

def is_nrrd(source):
    """Checks if this source is a NRRD source"""
    if isinstance(source, NrrdSource):
        return True
    if isinstance(source, str) and len(source) >= 4 and source[-4:] == 'nrrd':
        try:
            NrrdSource(source)
        except:
            return False
        return True
    return False


def read(source, slicing=None, **kwargs):
    """Read data from a nrrd file.
  
    Arguments
    ---------
    source : str
        The name of the nrrd file.
    slicing : slice, Slice or None
        An optional sub-slice to consider.
  
    Returns
    -------
    array : array
        The image data in the tif file as a buffer.
    """
    if not isinstance(source, NrrdSource):
        source = NrrdSource(source)
    if slicing is None:
        return source.array
    else:
        return source.__getitem__(slicing)


def write(sink, data, slicing=None, **kwargs):
    if isinstance(sink, NrrdSource):
        sink = sink.location
    if not isinstance(sink, str):
        raise ValueError(f'Invalid sink specification {sink!r}')

    if slicing is not None:
        memmap = _memmap(sink, mode='r+')
        memmap[slicing]= data
        return sink
    else:
        return _write(sink, data)


###############################################################################
### Reading
###############################################################################

def _convert_to_reproducible_floatingpoint(x):
    # This will help prevent loss of precision
    # IEEE754-1985 standard says that 17 decimal digits is enough in all cases.
    if type(x) == float:
        value = '{:.16f}'.format(x).rstrip('0').rstrip('.') # Remove trailing zeros, and dot if at end
    else:
        value = str(x)
    return value

_TYPEMAP_NRRD2NUMPY = {
    'signed char': 'i1',
    'int8': 'i1',
    'int8_t': 'i1',
    'uchar': 'u1',
    'unsigned char': 'u1',
    'uint8': 'u1',
    'uint8_t': 'u1',
    'short': 'i2',
    'short int': 'i2',
    'signed short': 'i2',
    'signed short int': 'i2',
    'int16': 'i2',
    'int16_t': 'i2',
    'ushort': 'u2',
    'unsigned short': 'u2',
    'unsigned short int': 'u2',
    'uint16': 'u2',
    'uint16_t': 'u2',
    'int': 'i4',
    'signed int': 'i4',
    'int32': 'i4',
    'int32_t': 'i4',
    'uint': 'u4',
    'unsigned int': 'u4',
    'uint32': 'u4',
    'uint32_t': 'u4',
    'longlong': 'i8',
    'long long': 'i8',
    'long long int': 'i8',
    'signed long long': 'i8',
    'signed long long int': 'i8',
    'int64': 'i8',
    'int64_t': 'i8',
    'ulonglong': 'u8',
    'unsigned long long': 'u8',
    'unsigned long long int': 'u8',
    'uint64': 'u8',
    'uint64_t': 'u8',
    'float': 'f4',
    'double': 'f8',
    'block': 'V'
}

_TYPEMAP_NUMPY2NRRD = {
    'i1': 'int8',
    'u1': 'uint8',
    'i2': 'int16',
    'u2': 'uint16',
    'i4': 'int32',
    'u4': 'uint32',
    'i8': 'int64',
    'u8': 'uint64',
    'f4': 'float',
    'f8': 'double',
    'V': 'block'
}

_NUMPY2NRRD_ENDIAN_MAP = {
    '<': 'little',
    'L': 'little',
    '>': 'big',
    'B': 'big'
}

def _parse_nrrdvector(inp):
    """Parse a vector from a nrrd header, return a list."""
    assert inp[0] == '(', "Vector should be enclosed by parenthesis."
    assert inp[-1] == ')', "Vector should be enclosed by parenthesis."
    return [_convert_to_reproducible_floatingpoint(x) for x in inp[1:-1].split(',')]

def _parse_optional_nrrdvector(inp):
    """Parse a vector from a nrrd header that can also be none."""
    if (inp == "none"):
        return inp
    else:
        return _parse_nrrdvector(inp)

_NRRD_FIELD_PARSERS = {
    'dimension': int,
    'type': str,
    'sizes': lambda fieldValue: [int(x) for x in fieldValue.split()],
    'endian': str,
    'encoding': str,
    'min': float,
    'max': float,
    'oldmin': float,
    'old min': float,
    'oldmax': float,
    'old max': float,
    'lineskip': int,
    'line skip': int,
    'byteskip': int,
    'byte skip': int,
    'content': str,
    'sample units': str,
    'datafile': str,
    'data file': str,
    'spacings': lambda fieldValue: [_convert_to_reproducible_floatingpoint(x) for x in fieldValue.split()],
    'thicknesses': lambda fieldValue: [_convert_to_reproducible_floatingpoint(x) for x in fieldValue.split()],
    'axis mins': lambda fieldValue: [_convert_to_reproducible_floatingpoint(x) for x in fieldValue.split()],
    'axismins': lambda fieldValue: [_convert_to_reproducible_floatingpoint(x) for x in fieldValue.split()],
    'axis maxs': lambda fieldValue: [_convert_to_reproducible_floatingpoint(x) for x in fieldValue.split()],
    'axismaxs': lambda fieldValue: [_convert_to_reproducible_floatingpoint(x) for x in fieldValue.split()],
    'centerings': lambda fieldValue: [str(x) for x in fieldValue.split()],
    'labels': lambda fieldValue: [str(x) for x in fieldValue.split()],
    'units': lambda fieldValue: [str(x) for x in fieldValue.split()],
    'kinds': lambda fieldValue: [str(x) for x in fieldValue.split()],
    'space': str,
    'space dimension': int,
    'space units': lambda fieldValue: [str(x) for x in fieldValue.split()],
    'space origin': _parse_nrrdvector,
    'space directions': lambda fieldValue:
                        [_parse_optional_nrrdvector(x) for x in fieldValue.split()],
    'measurement frame': lambda fieldValue:
                        [_parse_nrrdvector(x) for x in fieldValue.split()],
}

_NRRD_REQUIRED_FIELDS = ['dimension', 'type', 'encoding', 'sizes']

# The supported field values
_NRRD_FIELD_ORDER = [
    'type',
    'dimension',
    'space dimension',
    'space',
    'sizes',
    'space directions',
    'kinds',
    'endian',
    'encoding',
    'min',
    'max',
    'oldmin',
    'old min',
    'oldmax',
    'old max',
    'content',
    'sample units',
    'spacings',
    'thicknesses',
    'axis mins',
    'axismins',
    'axis maxs',
    'axismaxs',
    'centerings',
    'labels',
    'units',
    'space units',
    'space origin',
    'measurement frame',
    'data file']


def _dtype_from_header(fields):
    """Determine the numpy dtype of the data."""
    
    # Process the data type
    np_typestring = _TYPEMAP_NRRD2NUMPY[fields['type']]
    if np.dtype(np_typestring).itemsize > 1:
        if 'endian' not in fields:
            raise NrrdError('Nrrd header misses required field: "endian".')
        if fields['endian'] == 'big':
            np_typestring = '>' + np_typestring
        elif fields['endian'] == 'little':
            np_typestring = '<' + np_typestring

    return np.dtype(np_typestring)


def _validate_magic_line(line):
    """For NRRD files, the first four characters are always "NRRD", and
    remaining characters give information about the file format version
    """
    if not line.startswith('NRRD'):
        raise NrrdError('Missing magic "NRRD" word. Is this an NRRD file?')
    try:
        if int(line[4:]) > 5:
            raise NrrdError('NRRD file version too new for this library.')
    except:
        raise NrrdError(f'Invalid NRRD magic line: {line}')
    return len(line)


def _read_header(filename):
    """Parse the fields in the nrrd header

    nrrdfile can be any object which supports the iterator protocol and
    returns a string each time its next() method is called — file objects and
    list objects are both suitable. If csvfile is a file object, it must be
    opened with the ‘b’ flag on platforms where that makes a difference
    (e.g. Windows)

    >>> _read_header(("NRRD0005", "type: float", "dimension: 3"))
    {'type': 'float', 'dimension': 3, 'keyvaluepairs': {}}
    >>> _read_header(("NRRD0005", "my extra info:=my : colon-separated : values"))
    {'keyvaluepairs': {'my extra info': 'my : colon-separated : values'}}
    """
    
    if isinstance(filename, str):
        nrrd_file = open(filename,'rb')
    else:
        nrrd_file = filename

    # Collect number of bytes in the file header (for seeking below)
    header_size = 0
    it = iter(nrrd_file)
    header_size += _validate_magic_line(next(it).decode('ascii'))
    header = { 'keyvaluepairs': {} }
    for raw_line in it:
        header_size += len(raw_line)
        raw_line = raw_line.decode('ascii')

        # Trailing whitespace ignored per the NRRD spec
        line = raw_line.rstrip()
        # Comments start with '#', no leading whitespace allowed
        if line.startswith('#'):
            continue
        # Single blank line separates the header from the data
        if line == '':
            break
        # Handle the <key>:=<value> lines first since <value> may contain a
        # ': ' which messes up the <field>: <desc> parsing
        key_value = line.split(':=', 1)
        if len(key_value) == 2:
            key, value = key_value
            # escape \\ and \n ??
            # value.replace(r'\\\\', r'\\').replace(r'\n', '\n')
            header['keyvaluepairs'][key] = value
            continue

        # Handle the "<field>: <desc>" lines.
        field_desc = line.split(': ', 1)
        if len(field_desc) == 2:
            field, desc = field_desc
            ## preceding and suffixing white space should be ignored.
            field = field.rstrip().lstrip()
            desc = desc.rstrip().lstrip()
            if field not in _NRRD_FIELD_PARSERS:
                raise NrrdError(f'Unexpected field in nrrd header: "{field}".')
            if field in header.keys():
                raise NrrdError(f'Duplicate header field: "{field}"')
            header[field] = _NRRD_FIELD_PARSERS[field](desc)
            continue

        # Should not reach here
        raise NrrdError(f'Invalid header line: "{line}"')

    # Check whether the required fields are there
    for field in _NRRD_REQUIRED_FIELDS:
        if field not in header:
            raise NrrdError(f'Nrrd header misses required field: "{field}".')

    # line reading was buffered; correct file pointer to just behind header:
    nrrd_file.seek(header_size)

    return header


def _array(filename):
    """Read the actual data into a numpy array."""
    
    with open(filename,'rb') as f_handle:
        fields = _read_header(f_handle)

        dtype = _dtype_from_header(fields)
        shape = fields['sizes']
        order = 'F'

        # offset
        n_pixels=np.prod(shape)
        data_file_handle = f_handle
        datafile = fields.get("datafile", fields.get("data file", None))
        if datafile is not None:
            if os.path.isabs(datafile):
                datafilename = datafile
            else:
                datafilename = os.path.join(os.path.dirname(filename), datafile)
            data_file_handle = open(datafilename,'rb')

        if fields['encoding'] == 'raw':
            byte_skip = fields.get('byteskip', fields.get('byte skip', 0))
            if byte_skip == -1: # This is valid only with raw encoding
              totalbytes = dtype.itemsize * n_pixels
              data_file_handle.seek(-totalbytes, 2)
            else:
              lineskip = fields.get('lineskip', fields.get('line skip', 0))
              for _ in range(lineskip):
                data_file_handle.readline()
              data_file_handle.read(byte_skip)
            data = np.fromfile(data_file_handle, dtype)

        elif fields['encoding'] == 'gzip' or\
             fields['encoding'] == 'gz':
            gzipfile = gzip.GzipFile(fileobj=data_file_handle)
            # Again, unfortunately, np.fromfile does not support
            # reading from a gzip stream, so we'll do it like this.
            data = np.fromstring(gzipfile.read(), dtype)

        elif fields['encoding'] == 'bzip2' or\
             fields['encoding'] == 'bz2':
            bz2_file = bz2.BZ2File(fileobj=data_file_handle)
            # Again, unfortunately, np.fromfile does not support
            # reading from a gzip stream, so we'll do it like this.
            data = np.fromstring(bz2_file.read(), dtype)

        else:
            raise NrrdError(f'Unsupported encoding: "{fields["encoding"]}"')

        if n_pixels != data.size:
           raise NrrdError(f'ERROR: {n_pixels}-{data.size}={n_pixels - data.size}')

        data = np.reshape(data, shape, order=order)
        return data


def _dtype(filename):
    """Determine data type from nrrd file."""
    # read header
    with open(filename,'rb') as filehandle:
        fields = _read_header(filehandle)
    
    # Determine the data type from the fields
    dtype = fields['sizes'](fields)
    return dtype


def _shape(filename):
    """Determine shape from nrrd file."""
    #read header
    with open(filename,'rb') as filehandle:
        fields = _read_header(filehandle)
    
    return tuple(fields['sizes'])


def _order(filename):
    """Determine shape from nrrd file."""
    return 'F'


def _offset(filename):
    """Offset of data in file."""
  
    with open(filename,'rb') as f_handle:
        fields = _read_header(f_handle)

        # offset
        data_file_handle = f_handle
        datafile = fields.get("datafile", fields.get("data file", None))
        if datafile is not None:
            if os.path.isabs(datafile):
                data_filename = datafile
            else:
                data_filename = os.path.join(os.path.dirname(filename), datafile)
            data_file_handle = open(data_filename,'rb')

        if fields['encoding'] == 'raw':
            byte_skip = fields.get('byteskip', fields.get('byte skip', 0))
            if byte_skip == -1: # This is valid only with raw encoding
                numPixels=np.prod(fields['sizes'])
                dtype = _dtype_from_header(fields)
                totalbytes = dtype.itemsize * numPixels
                data_file_handle.seek(-totalbytes, 2)
            else:
                lineskip = fields.get('lineskip', fields.get('line skip', 0))
                for _ in range(lineskip):
                    data_file_handle.readline()
                data_file_handle.read(byte_skip)

        return data_file_handle.tell()


def _memmap(filename, mode = None):
    """Create memmap to the nrrd data."""
    with open(filename,'rb') as f_handle:
        fields = _read_header(f_handle)

        if fields['encoding'] != 'raw':
            raise NrrdError(f'Cannot memmap to compressed file {fields["encoding"]!r}!')

        dtype = _dtype_from_header(fields)
        shape = tuple(fields['sizes'])
        order = 'F'

        #d atafile
        data_filename = filename
        data_file_handle = f_handle
        datafile = fields.get("datafile", fields.get("data file", None))
        if datafile is not None:
            if os.path.isabs(datafile):
                data_filename = datafile
            else:
                data_filename = os.path.join(os.path.dirname(filename), datafile)
            data_file_handle = open(data_filename,'rb')

        # offset
        byte_skip = fields.get('byteskip', fields.get('byte skip', 0))
        if byte_skip == -1: # This is valid only with raw encoding
            n_pixels=np.prod(shape)
            total_bytes = dtype.itemsize * n_pixels
            data_file_handle.seek(-total_bytes, 2)
        else:
            line_skip = fields.get('lineskip', fields.get('line skip', 0))
            for _ in range(line_skip):
                data_file_handle.readline()
            data_file_handle.read(byte_skip)
        offset = data_file_handle.tell()

        if mode is None:
            mode = 'r+'

        return np.memmap(data_filename, dtype=dtype, mode=mode, offset=offset, shape=shape, order=order)

###############################################################################
### Writing
###############################################################################

def _format_nrrd_list(fieldValue) :
    return ' '.join([_convert_to_reproducible_floatingpoint(x) for x in fieldValue])

def _format_nrrdvector(v) :
    return '(' + ','.join([_convert_to_reproducible_floatingpoint(x) for x in v]) + ')'

def _format_optional_nrrdvector(v):
    if v == 'none':
        return 'none'
    else :
        return _format_nrrdvector(v)

_NRRD_FIELD_FORMATTERS = {
    'dimension': str,
    'type': str,
    'sizes': _format_nrrd_list,
    'endian': str,
    'encoding': str,
    'min': str,
    'max': str,
    'oldmin': str,
    'old min': str,
    'oldmax': str,
    'old max': str,
    'lineskip': str,
    'line skip': str,
    'byteskip': str,
    'byte skip': str,
    'content': str,
    'sample units': str,
    'datafile': str,
    'data file': str,
    'spacings': _format_nrrd_list,
    'thicknesses': _format_nrrd_list,
    'axis mins': _format_nrrd_list,
    'axismins': _format_nrrd_list,
    'axis maxs': _format_nrrd_list,
    'axismaxs': _format_nrrd_list,
    'centerings': _format_nrrd_list,
    'labels': _format_nrrd_list,
    'units': _format_nrrd_list,
    'kinds': _format_nrrd_list,
    'space': str,
    'space dimension': str,
    'space units': _format_nrrd_list,
    'space origin': _format_nrrdvector,
    'space directions': lambda fieldValue: ' '.join([_format_optional_nrrdvector(x) for x in fieldValue]),
    'measurement frame': lambda fieldValue: ' '.join([_format_optional_nrrdvector(x) for x in fieldValue]),
}


def _write_data(data, filehandle, options):
    # Now write data directly
    # rawdata = data.transpose([2,0,1]).tostring(order = 'C')
    # FIXME: this is only working for numpy arrays!

    compressors = {
        'gzip': gzip,
        'bz2': bz2
    }
    encoding = options['encoding']

    if encoding == 'raw':
        data.tofile(filehandle)
    elif encoding in compressors.keys():
        with compressors[encoding].open(filename=filehandle, mode='wb') as compressor_file:
            compressor_file.write(data.tostring(order='F'))
    else:
        raise NrrdError(f'Unsupported encoding: {encoding}')


def _write(filename, data, options={}, separate_header=False):
    """Write data to nrrd file.
    
    Arguments:
        filename (str): file name as regular expression
        data (array): image data
        options (dict): options dictionary
        separateHeader (bool): write a separate header file
    
    Returns:
        str: nrrd output file name
  
    To sample date use `options['spacings'] = [s1, s2, s3]` for
    3d data with sampling deltas `s1`, `s2`, and `s3` in each dimension.
    """
    dtype = data.dtype

    options['type'] = _TYPEMAP_NUMPY2NRRD[dtype.str[1:]]
    if dtype.itemsize > 1:
        options['endian'] = _NUMPY2NRRD_ENDIAN_MAP[dtype.str[:1]]
    
    # if 'space' is specified 'space dimension' can not. See http://teem.sourceforge.net/nrrd/format.html#space
    if 'space' in options.keys() and 'space dimension' in options.keys():
        del options['space dimension']
    options['dimension'] = data.ndim
    dsize = list(data.shape)
    options['sizes'] = dsize

    # The default encoding is 'raw'
    if 'encoding' not in options:
        options['encoding'] = 'raw'

    # A bit of magic in handling options here.
    # If *.nhdr filename provided, this overrides `separate_header=False`
    # If *.nrrd filename provided AND separate_header=True, separate files   written.
    # For all other cases, header & data written to same file.
    if filename[-5:] == '.nhdr':
        separate_header = True
        if 'data file' not in options:
            datafilename = filename[:-4] + str('raw')
            if options['encoding'] == 'gzip':
                datafilename += '.gz'
            options['data file'] = datafilename
        else:
            datafilename = options['data file']
    elif filename[-5:] == '.nrrd' and separate_header:
        separate_header = True
        datafilename = filename
        filename = filename[:-4] + str('nhdr')
    else:
        # Write header & data as one file
        datafilename = filename
        separate_header = False

    with open(filename,'wb') as filehandle:
        filehandle.write(b'NRRD0005\n')
        filehandle.write(b'# This NRRD file was generated by pynrrd\n')
        filehandle.write(b'# on ' +
                         datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S').encode('ascii') +
                         b'(GMT).\n')
        filehandle.write(b'# Complete NRRD file format specification at:\n')
        filehandle.write(b'# http://teem.sourceforge.net/nrrd/format.html\n')

        # Write the fields in order, this ignores fields not in _NRRD_FIELD_ORDER
        for field in _NRRD_FIELD_ORDER:
            if field in options:
                outline = (field + ': ' +
                           _NRRD_FIELD_FORMATTERS[field](options[field]) +
                           '\n').encode('ascii')
                filehandle.write(outline)
        d = options.get('keyvaluepairs', {})
        for (k,v) in sorted(d.items(), key=lambda t: t[0]):
            outline = (str(k) + ':=' + str(v) + '\n').encode('ascii')
            filehandle.write(outline)

        # Write the closing extra newline
        filehandle.write(b'\n')

        # If a single file desired, write data
        if not separate_header:
            _write_data(data, filehandle, options)

    # If separate header desired, write data to different file
    if separate_header:
        with open(datafilename, 'wb') as datafilehandle:
            _write(data, datafilehandle, options)
    
    return filename


###############################################################################
### Tests
###############################################################################
        
def _test():
    import os
    import numpy as np

    data = np.random.rand(20,50,10)
    data[5:15, 20:45, 2:9] = 0

    filename = 'test.nrrd'

    NRRD.write(filename, data)

    check = NRRD.read(filename)
    
    print(np.all(data==check))
    
    os.remove(filename)
