# -*- coding: utf-8 -*-
"""
Source
======

This module provides the base class for data sources and sinks.
"""
__author__    = 'Christoph Kirst <christoph.kirst.ck@gmail.com>'
__license__   = 'GPLv3 - GNU General Pulic License v3 (see LICENSE.txt)'
__copyright__ = 'Copyright © 2020 by Christoph Kirst'
__webpage__   = 'http://idisco.info'
__download__  = 'http://www.github.com/ChristophKirst/ClearMap2'


import numpy as np

import ClearMap.IO.FileUtils as fu

from ClearMap.Utils.Formatting import ensure

###############################################################################
### Source base class
###############################################################################

class Source(object):
  """Base abstract source class."""
  
  #__slots__ = ();
  
  def __init__(self, name = None):
    """Initialization."""
    if name is not None:
      self._name = name;

  @property
  def name(self):
    """The name of this source.
    
    Returns
    -------
    name : str
      Name of this source.
    """
    if hasattr(self, '_name'):
      return self._name;
    else:
      return type(self).__name__;
  
  @name.setter
  def name(self, value):
    self._name = ensure(value, str);
  
  
  @property 
  def shape(self):
    """The shape of the source.
    
    Returns
    -------
    shape : tuple
      The shape of the source.
    """
    return None;
  
  @shape.setter
  def shape(self, value):
    raise ValueError('Cannot set shape for this source.');

  
  @property 
  def dtype(self):
    """The data type of the source.
    
    Returns
    -------
    dtype : dtype
      The data type of the source.
    """
    return None;
  
  @dtype.setter
  def dtype(self, value):
    raise ValueError('Cannot set dtype for this source.');
  
  
  @property 
  def order(self):
    """The contiguous order of the underlying data array.
    
    Returns
    -------
    order : str
      Returns 'C' for C contigous and 'F' for fortran contigous, None otherwise.
    """
    return None;
  
  @order.setter
  def order(self, value):
    raise ValueError('Cannot set order for this source.');
 
 
#  @property 
#  def memory(self):
#    """The memory type of the source.
#    
#    Returns
#    -------
#    memory : str
#      Returns 'shared' for a shared memory buffer, 'memmap' for a memory map to a file, None otherwise.
#    """
#    return None;
#  
#  @memory.setter
#  def memory(self, value):
#    raise ValueError('Cannot set memory type for this source.');
 
  
  @property
  def location(self):
    """The location where the data of the source is stored.
    
    Returns
    -------
    location : str or None
      Returns the location of the data source or None if this source lives in memory only.
    """
    return None;
  
  @location.setter
  def location(self, value):
    raise ValueError('Cannot set location for this source.');
  
  
  ### Derived properties
  @property
  def ndim(self):
    """The number of dimensions of the source.
    
    Returns
    -------
    ndim : int
      The number of dimension of the source.
    """
    return len(self.shape);  
  
  
  @property
  def size(self):
    """The size of the source.
    
    Returns
    -------
    size : int
      The number of data items in the source.
    """
    return np.prod(self.shape);  
  
  
#  @property
#  def shared(self):
#    """Returns True if the source is in shared memory.
#    
#    Returns
#    -------
#    shared : bool
#      True if the source is in shared memory.
#    """
#    return self.memory == 'shared';
#  
#  @shared.setter
#  def shared(self, value):
#    if value is True:
#      self.memory == 'shared';
#    elif self.memory == 'shared':
#      self.memory = None; 
#  
#      
#  @property
#  def memmap(self):
#    """Returns True if the source is a memory map.
#    
#    Returns
#    -------
#    memmap : bool
#      True if the source is memory mapped.
#    """
#    return self.memory == 'memmap';
#    
#  @memmap.setter
#  def memmap(self, value):
#    if value is True:
#      self.memory == 'memmap';
#    elif self.memory == 'memmap':
#      self.memory = None;
  
  
  ### Functionality
  def exists(self):
    if self.location is not None:
      return fu.is_file(self.location);
    else:
      return False;
  
  
  ### Source conversions  
  def as_virtual(self):
    """Return virtual source without array data to pass in parallel processing.
    
    Returns
    -------
    source : Source class
      The source class without array data.
    """
    #return VirtualSource(source = self.source);
    raise NotImplementedError('virtual source not implemented for this source!')
  
  def as_real(self):
    return self;
  
  def as_buffer(self):
    raise NotImplementedError('buffer not implemented for this source!')
    
  def as_memory(self):
    return np.array(self.as_buffer());
  
  ### Data
  def __getitem__(self, *args):
    raise KeyError('No getitem routine for this source!')

  def __setitem__(self, *args):
    raise KeyError('No setitem routine for this source!')
  
  def read(self, *args, **kwargs):
    raise KeyError('No read routine for this source!');
  
  def write(self, *args, **kwargs):
    raise KeyError('No write routine for this source!')
  
  ### Formatting
  def __str__(self):
    try:
      name = self.name;
      name = '%s' % name if name is not None else '';
    except:
      #print('name')
      name ='';
    
    try:
      shape = self.shape
      shape ='%r' % ((shape,)) if shape is not None else '';
    except:
      #print('shape')
      shape = '';

    try:
      dtype = self.dtype;
      dtype = '[%s]' % dtype if dtype is not None else '';
    except:
      #print('dtype')
      dtype = '';
            
    try:
      order = self.order;
      order = '|%s|' % order if order is not None else '';
    except:
      #print('order')
      order = '';
    
#    try:
#      memory = self.memory;
#      memory = '<%s>' % memory if memory is not None else '';
#    except:
#      memory = '';  
    
    try:
      location = self.location;
      location = '%s' % location if location is not None else '';
      if len(location) > 100:
        location = location[:50] + '...' + location[-50:]
      if len(location) > 0:
        location = '{%s}' % location;
    except:
      #print('location')
      location = '';    
    
#    try:
#      array = self.array.__str__();
#      if len(array) > 100:
#        e = array[100:].find('\n');
#        if e != -1:
#          array = array[:100 + e] + '...';
#      if len(array) > 0:
#        array = '\n' + array;
#    except:
#      array = '';
    
    return name + shape + dtype + order + location; # + array
  
  def __repr__(self):
    return self.__str__();


###############################################################################
### Abstract and VirtualSource base class
###############################################################################

#TODO: memory -> device argument
class AbstractSource(Source):
  """Abstract source to handle data sources without data in memory.
  
  Note
  ----
  This class handles essential info about a source and to how access its data.
  """
  
  #__slots__ = ('_shape', '_dtype', '_order', '_location')
  
  def __init__(self, source = None, shape = None, dtype = None, order = None, location = None, name = None):
    """Source class construtor.
    
    Arguments
    ---------
    shape : tuple of int or None
      Shape of the source, if None try to determine from source.
    dtype : dtype or None
      The data type of the source, if None try to detemrine from source.
    order : 'C' or 'F' or None
      The order of the source, c or fortran contiguous.
    memory : str or None
      The memory type of the source, 'memmap' uses memory mapped array and 'shared'returns a shared memory.
    location : str or None
      The location of the source.
    """
    super(AbstractSource, self).__init__(name = name);
    
    if source is not None:      
      if shape is None and hasattr(source, 'shape'):
        shape = source.shape;
      if dtype is None and hasattr(source, 'dtype'):
        dtype = source.dtype;
      if order is None and hasattr(source, 'order'):
        order = source.order;
      #if memory is None and hasattr(source, 'memory'):
      #  memory = memory.order; 
      if location is None and hasattr(source, 'location'):
        location = source.location;
        
    self._shape    = ensure(shape,    tuple);
    self._dtype    = ensure(dtype,    np.dtype);
    self._order    = ensure(order,    str);
    #self._memory   = ensure(memory,   str);
    self._location = ensure(location, str);
  
  
  @property 
  def shape(self):
    """The shape of the source.
    
    Returns
    -------
    shape : tuple
      The shape of the source.
    """
    return self._shape;
  
  @shape.setter
  def shape(self, value):
    self._shape = ensure(value, tuple);
  
  
  @property 
  def dtype(self):
    """The data type of the source.
    
    Returns
    -------
    dtype : dtype
      The data type of the source.
    """
    return self._dtype;
  
  @dtype.setter
  def dtype(self, value):
    self._dtype = ensure(value, np.dtype);
  
  
  @property 
  def order(self):
    """The continguous order of the data array of the source.
    
    Returns
    -------
    order : str
      Returns 'C' for C and 'F' for fortran contiguous arrays, None otherwise.
    """
    return self._order;
  
  @order.setter
  def order(self, value):
    if value not in [None, 'C', 'F']:
        raise ValueError("Order %r not in [None, 'C' or 'F']!" % value);
    self._order = ensure(value, str);
 
    
  @property
  def location(self):
    """The location of the source's data.
    
    Returns
    -------
    location : str or None
      Returns the location of the data source or None if there is none.
    """
    return self._location;
  
  @location.setter
  def location(self, value):
    self._location = ensure(value, str);
  
  
  def as_virtual(self):
    return self;
  
  def as_real(self):
    raise RuntimeError('The abstract source cannot be converted to a real source!')
  
  def as_buffer(self):
    raise RuntimeError('The abstract source cannot be converted to a buffer!')


class VirtualSource(AbstractSource):
  """Virtual source to handle data sources without data in memory.
  
  Note
  ----
  This class is fast to serialize and useful as a source pointer in paralle processing.  
  """
  def __init__(self, source = None, shape = None, dtype = None, order = None, location = None, name = None):
    AbstractSource.__init__(self, source=source, shape=shape, dtype=dtype, order=order, location=location, name=name)
  
  def __getitem__(self, *args):
    return self.as_real().__getitem__(*args);
    
  def __setitem__(self, *args):
    self.as_real().__setitem__(*args);

  def read(self, *args, **kwargs):
    return self.as_real().read(*args, **kwargs);
  
  def write(self, *args, **kwargs):
    self.as_real().write(*args, **kwargs);

  #def __getattr__(self, name):
  #  return getattr(self.as_real(), name);


###############################################################################
### Tests
###############################################################################

def _test():
  import ClearMap.IO.Source as src
  #reload(src)
  
  s = src.VirtualSource(shape = (50,50), dtype = float, location = '/home/test.npy', order = 'F')
  print(s)
  
  s.size
  s.ndim
  
