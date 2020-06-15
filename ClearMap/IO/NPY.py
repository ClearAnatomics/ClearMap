# -*- coding: utf-8 -*-
"""
NPY
===

IO interface to numpy arrays.
"""
__author__    = 'Christoph Kirst <christoph.kirst.ck@gmail.com>'
__license__   = 'GPLv3 - GNU General Pulic License v3 (see LICENSE.txt)'
__copyright__ = 'Copyright © 2020 by Christoph Kirst'
__webpage__   = 'http://idisco.info'
__download__  = 'http://www.github.com/ChristophKirst/ClearMap2'


import numpy as np

import ClearMap.IO.Source as src
#import ClearMap.IO.FileUtils as fu

###############################################################################
### Source classe
###############################################################################

class Source(src.Source):
  """Numpy array source."""
  
  def __init__(self, array = None, shape = None, dtype = None, order = None, name = None):
    """Numpy source class construtor.
    
    Arguments
    ---------
    array : array
      The underlying data array of this source.
    """
    super(Source, self).__init__(name=name);
    self._array = _array(shape=shape, dtype=dtype, order=order, array=array);

    
  def __getattr__(self, name):
    #numpy attributes
    if name != '_array' and hasattr(self, '_array') and hasattr(self._array, name):
      return getattr(self._array, name);
    else:
      raise AttributeError('Not such attribute %r!' % name);
  
  #def __del__(self):
  #  self._array = None;
    
  
  @property
  def name(self):
    return "Numpy-Source";  
  
  
  @property
  def array(self):
    """The underlying data array.
    
    Returns
    -------
    array : array
      The underlying data array of this source.
    """
    return self._array;
  
  @array.setter
  def array(self, value):
    self._array = _array(value);

  
  @property 
  def shape(self):
    """The shape of the source.
    
    Returns
    -------
    shape : tuple
      The shape of the source.
    """
    return self._array.shape;
  
  @shape.setter
  def shape(self, value):
    self._array.shape = value;
  
  
  @property 
  def dtype(self):
    """The data type of the source.
    
    Returns
    -------
    dtype : dtype
      The data type of the source.
    """
    return self._array.dtype;
  
  @dtype.setter
  def dtype(self, value):
    self._array = np.asarray(self._array, dtype=value)
  
    
  @property 
  def order(self):
    """The order of how the data is stored in the source.
    
    Returns
    -------
    order : str
      Returns 'C' for C contigous and 'F' for fortran contigous, None otherwise.
    """
    return order(self.array);
  
  @order.setter
  def order(self, value):
    self._array = np.asarray(self._array, order = value)
    
  
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
    return tuple(s // self._array.itemsize for s in self._array.strides)
  
  
  @property
  def offset(self):
    """The offset of the memory map in the file.
    
    Returns
    -------
    offset : int
      Offset of the memeory map in the file.
    """
    if self._array.base is not None:
      return np.byte_bounds(self._array)[0] - np.byte_bounds(self._array.base)[0];
    else:
      return 0;
  
  ### Parallel processing
  def as_virtual(self):
    #TODO: convert to shared memory array ?
    return self;
    
  def as_real(self):
    return self;
  
  def as_buffer(self):
    return self._array;
  
  ### Data
  def __getitem__(self, *args):
    return self.array.__getitem__(*args);

  def __setitem__(self, *args):
    self.array.__setitem__(*args);



#class Array(np.ndarray):
#  """Array wrapper around numpy ndarray."""
#  
#  def __new__(cls, array):
#    obj = np.asarray(array).view(cls)
#    #obj.order = order(obj);
#    return obj
#
#  def __array_finalize__(self, obj):
#    if obj is None: return
#    #self.order = getattr(obj, 'order', None)
#
#  def name(self):
#    return "Source-Numpy";
#    
#  def array(self):
#    return self.view(np.ndarray);
#  
#  @property
#  def order(self):
#    return order(self);
#    
#  @property
#  def array_strides(self):
#    return tuple(np.array(self.strides, dtype = int) / self.itemsize);
#    
#  def __str__(self):    
#    if self.shape is not None:
#      shape = '%r' % ((self.shape,));
#    else:
#      shape = '';
#
#    if self.dtype is not None:
#      dtype = '[%s]' % self.dtype;
#    else:
#      dtype = '';
#            
#    if order(self) is not None:
#      _order = '|%s|' % order(self);
#    else:
#      _order = '';
#    
#    array = super(Array, self).__str__();
#    if len(array) > 100:
#      e = array[100:].find('\n');
#      if e != -1:
#        array = array[:100 + e] + '...';
#    if len(array) > 0:
#      array = '\n' + array;
#    else:
#      array = '';
#  
#    return 'Array' + shape + dtype + _order + array
#  
#  def __repr__(self):
#    return self.__str__();


###############################################################################
### Functionality
###############################################################################

def order(array):
  """Returns the contigous order of an array.
  
  Arguments
  ---------
  array : ndarray
  
  Returns
  -------
  order : 'C', 'F', None
  """
  if isinstance(array, src.Source):
    return array.order;
  elif isinstance(array, np.ndarray):
    if array.flags['C_CONTIGUOUS']:
      return 'C'
    elif  array.flags['F_CONTIGUOUS']:
      return 'F'
    else:
      return None;
  else:
    return None;
    

###############################################################################
### IO Interface
###############################################################################

def is_numpy(source):
  if isinstance(source, (Source, np.ndarray, list, tuple)):
    return True;
  #elif isinstance(source, str): # and fu.file_extension(source) == 'npy':
  #  return True;
  else:
    return False;


def read(source, slicing = None, as_source = None, as_array = None, processes = None, **kwargs):
  if isinstance(source, (list, tuple)):
    source = np.array(source);   
  if isinstance(source, Source):
    if slicing is not None:
      source = source.__getitem__(slicing);
    if as_array:
      return source.array
    else:
      return source;
  elif isinstance(source, np.ndarray):
    if slicing is not None:
      source = source.__getitem__(slicing);
    if as_source:
      return Source(array = source);
    else:
      return source;
#  elif isinstance(source, str): # and fu.file_extension(source) == 'npy':
#    source = np.load(source);   
#    if slicing is not None:
#      source = source.__getitem__(slicing);
#    if as_source:
#      return Source(array = source);
#    else:
#      return source;
  else:
    raise ValueError('The source is not a valid numpy source!')
    

#TODO: add processes kewword for parallel writing
def write(sink, data, slicing = None, **kwargs):
  if slicing is None:
    slicing = ();
  if sink is None:
    return data.__getitem__(slicing);
  if isinstance(sink, (src.Source, np.ndarray)):
    sink.__setitem__(slicing, data);
    return sink;
#  elif isinstance(sink, str): #and fu.file_extension(sink) == 'npy'
#    if slicing != ():
#      if not fu.is_file(sink):
#        raise ValueError('Cannot write slice to a not existing file %s!' % sink);
#      memmap = np.lib.format.open_memmap(sink);
#      memmap.__setitem__(slicing, data);
#    else:
#      np.save(sink, data);
#    return sink;
  else:
    raise ValueError('The sink is not a valid numpy sink!')



def create(shape = None, dtype = None, order = None, array = None, as_source = True, **kwargs):
  """Create a numpy array.
  
  Arguments
  ---------
  shape : tuple or None
    The shape of the memory map to create.
  dtype : dtype 
    The data type of the memory map.
  order : 'C', 'F', or None
    The contiguous order of the memmap.
  array : array, Source or None
    Optional source with data to fill the numpy array with.
  as_source : bool
    If True, return as Source class.
    
  Returns
  -------
  array : np.array
    The numpy array.
    
  Note
  ----
  By default numpy arrays are initialized as fortran contiguous if order is None.
  """
  array = _array(shape=shape, dtype=dtype, order=order, array=array)
  if as_source:
    return Source(array=array);
  else:
    return array;

###############################################################################
### Helpers
###############################################################################

def _order(array):
  return order(array);

def _array(shape = None, dtype = None, order = None, array = None):
  """Create a numpy array.
  
  Arguments
  ---------
  shape : tuple or None
    The shape of the memory map to create.
  dtype : dtype 
    The data type of the memory map.
  order : 'C', 'F', or None
    The contiguous order of the memmap.
  array : array, Source or None
    Optional source with data to fill the memory map.
    
  Returns
  -------
  array : np.ndarray
    The array.
  """ 
  if isinstance(array, (list, tuple)):
    array = np.asarray(array, order=order, dtype=dtype);
  
  if isinstance(array, np.ndarray):
    shape = shape if shape is not None else array.shape;
    dtype = dtype if dtype is not None else array.dtype;
    order = order if order is not None else _order(array);

    if shape != array.shape:
      raise ValueError('Shape %r and array shape %r mismatch!' % (shape, array.shape));
    
    if dtype != array.dtype or order != _order(array):
      array = np.asarray(array, order=order, dtype=dtype);
  
  else:
    if shape is None:
      raise ValueError('Cannot create array without shape!');
    array = np.zeros(shape, dtype=dtype, order=order);
  
  return array;



###############################################################################
### Tests
###############################################################################

def _test():
  import numpy as np
  import ClearMap.IO.NPY as npy;
  #reload(npy);
  
  s = npy.Source(array=np.zeros((5,7)));
  print(s);
  
  import ClearMap.IO.Slice as slc
  t = slc.Slice(source= s, slicing= (1,));
  print(t);
  
  v = t.as_virtual()
  print(v);
  
  x = np.ones(250*1000*1000)
  xs = npy.Source(array=x);
  
  print(xs)

  del x
  del xs
