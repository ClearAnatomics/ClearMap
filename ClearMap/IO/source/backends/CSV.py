# -*- coding: utf-8 -*-
"""
CSV
===

Interface to read and write csv files.

Note
----
The module utilizes the csv file writer/reader from numpy.
"""
__author__    = 'Christoph Kirst <christoph.kirst.ck@gmail.com>'
__license__   = 'GPLv3 - GNU General Public License v3 (see LICENSE.txt)'
__copyright__ = 'Copyright © 2020 by Christoph Kirst'
__webpage__   = 'https://idisco.info'
__download__  = 'https://www.github.com/ChristophKirst/ClearMap2'


import numpy as np

import ClearMap.IO.source.Source as source_mod
# noinspection PyUnusedImports
from ClearMap.IO.source.backend_defaults import create, open_ro
import ClearMap.IO.source.Slice as slc

###############################################################################
### CSVSource class
###############################################################################

class CSVSource(source_mod.Source):
    """CSV array source."""
    _CACHED_PROPERTIES = ('_frame',)


    def __init__(self, location, mode=None):
        """CSV source class constructor.

        Arguments
        ---------
        location : str
            The filename of the csv source.
        """
        super().__init__(location, mode)

    @cached_property
    def _frame(self):
        return _array(self.location)

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
        _write(self.location, value)

    @property
    def shape(self):
        """The shape of the source.

        Returns
        -------
        shape : tuple
            The shape of the source.
        """
        return self.array.shape

    @shape.setter
    def shape(self, value):
        raise NotImplementedError('Cannot set shape of csv file')

    @property
    def dtype(self):
        """The data type of the source.

        Returns
        -------
        dtype : dtype
            The data type of the source.
        """
        return self.array.dtype

    @dtype.setter
    def dtype(self, value):
        raise NotImplementedError('Cannot set dtype of csv file')

    @property
    def order(self):
        """The order of how the data is stored in the source.

        Returns
        -------
        order : str
            Returns 'C' for C contigous and 'F' for fortran contigous, None otherwise.
        """
        return self.array.order

    @order.setter
    def order(self, value):
        raise NotImplementedError('Cannot set order of csv file')

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
        array = self.array
        return tuple(s // array.itemsize for s in array.strides)

    @property
    def offset(self):
        """The offset of the memory map in the file.

        Returns
        -------
        offset : int
            Offset of the memeory map in the file.
        """
        return 0

    ### Data
    def __getitem__(self, *args):
        array = _array(self.location)
        return array.__getitem__(*args)

    def __setitem__(self, *args):
        array = _array(self.location)
        array.__setitem__(*args)
        _write(self.location, array)

    def as_memmap(self):
         raise NotImplementedError('Memmap creation not implemented yet!')

    def as_virtual(self):
        return CSVVirtualSource(source=self)

    def as_buffer(self):
        return self.array


class CSVVirtualSource(source_mod.VirtualSource):
    _real_class = CSVSource

    def __init__(self, source=None, shape=None, dtype=None, order=None,
                 location=None, name=None, mode=None):
        super().__init__(source=source, shape=shape, dtype=dtype, order=order, location=location, name=name, mode=mode)
        if isinstance(source, CSVSource):
            self.location = source.location


###############################################################################
### IO Interface
###############################################################################

def is_csv(source):
    """Checks if this source is a CSV source"""
    if isinstance(source, CSVSource):
        return True
    if isinstance(source, str) and len(source) >= 3 and source[-3:] == 'csv':
        return True
    return False


def read(source, slicing = None, as_source = None, **kwargs):
    """Read data from a csv file.

    Arguments
    ---------
    source : str
        The name of the CSV file.
    slicing : slice, Slice or None
        An optional sub-slice to consider.
    as_source : bool
        If True, return results as a source.

    Returns
    -------
    array : array
        The data in the csv file as a buffer or source.
    """
    if not isinstance(source, CSVSource):
        source = CSVSource(source)
    if slicing is None:
        if as_source:
            return source
        else:
            return source.array
    else:
        if as_source:
            return slc.Slice(source, slicing=slicing)
        else:
            return source.__getitem__(slicing)


def write(sink, data, slicing = None, **kwargs):
    """Write data to a csv file.

    Arguments
    ---------
    sink : str
        The name of the CSV file.
    data : array
        The data to write into the CSV file.
    slicing : slice, Slice or None
        An optional sub-slice to consider.

    Returns
    -------
    sink : array or source
        The sink csv file.
    """
    if not isinstance(sink, CSVSource):
        sink = CSVSource(sink)

    if slicing is not None:
        array = sink.array
        array[slicing]= data
    else:
        array = data

    return _write(sink, array)


###############################################################################
### Helpers
###############################################################################

def _write(filename, points, **args):
      """Write point data to csv file"""
      np.savetxt(filename, points, delimiter=',', newline='\n', fmt='%.5e')
      return filename


def _array(location, delimiter =',', **args):
    """Read data from csv file.
    
    Arguments
    ---------
    location : str
        Location of the csv array data.
    delimiter : char
        The delimiter between subsequent array entries.
    
    Returns
    -------
    array : array
        The data as a numpy array.
    """
    points = np.loadtxt(location, delimiter=delimiter)
    return points


###############################################################################
### Tests
###############################################################################

def test():    
    """Test CSV module"""
    import os
    import numpy as np
    from ClearMap.IO.source.backends.CSV import CSVSource
    
    location = 'test.csv'
    points = np.random.rand(5,3)

    s = CSVSource(location)
    print(s)    
    s.array = points
    print(s)    
    
    os.remove(location)
