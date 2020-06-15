# -*- coding: utf-8 -*-
"""
TIF module provides interface to read and write tif image files.

Note
----
This modules relies onf the tifffile library.
"""
__author__    = 'Christoph Kirst <christoph.kirst.ck@gmail.com>'
__license__   = 'GPLv3 - GNU General Pulic License v3 (see LICENSE.txt)'
__copyright__ = 'Copyright © 2020 by Christoph Kirst'
__webpage__   = 'http://idisco.info'
__download__  = 'http://www.github.com/ChristophKirst/ClearMap2'


import tifffile as tif

import ClearMap.IO.Source as src
import ClearMap.IO.Slice as slc

from ClearMap.Utils.Lazy import lazyattr


###############################################################################
### Source class
###############################################################################

class Source(src.Source):
  """Class to handle a tif file source
  
  Note
  ----
  Its assumed that the image data is stored in a serregionies of the tif file.
  """
  def __init__(self, location, series = 0, multi_file = False):
    self._tif = tif.TiffFile(location, multifile = multi_file);
    self._series = series;
    self.multi_file = multi_file;
    
  @property
  def name(self):
    return "Tif-Source";
      
  @lazyattr
  def series(self):
    return self._tif.series[self._series];
    
  @property
  def shape(self):
    return shape_from_tif(self.tif_shape);
    
  @property
  def tif_shape(self):
    if self._tif._multifile:
      return self._tif.series[self._series].shape;
    else:
      s =  self._tif.pages[0].shape;
      l = len(self._tif.pages);
      if l > 1:
        s = (l,) + s;
    return s;
    
  @property
  def dtype(self):
    return self._tif.pages[0].dtype;
  
  @property
  def location(self):
    return self._tif._fh.path;
    
  @location.setter
  def location(self, value):
    if value != self.location:
      self._tif = tif.TiffFile(value, multifile = False);
  
  @property
  def array(self, processes = None):
     array = self._tif.asarray(maxworkers=processes);
     return array_from_tif(array);
   
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
    memmap = self.as_memmap();
    return tuple(s // memmap.itemsize for s in memmap.strides);
  
  def __getitem__(self, slicing, processes = None):
    ndim = self.ndim;
    if ndim >= 3:
      slicing = slc.unpack_slicing(slicing, ndim);

      slicing_z  = slicing[-1];
      array = self._tif.asarray(key = slicing_z, maxworkers=processes);
      array = array_from_tif(array);    
      
      slicing_xy = (Ellipsis,) + slicing[-3:-1];
      if len(array.shape) > len(self._tif.pages[0].shape):
        slicing_xy = slicing_xy + (slice(None),);
      return array[slicing_xy];
    
    else:
      array = self._tif.asarray(maxworkers=processes);
      array = array_from_tif(array)
      return array.__getitem__(slicing);
      
  def __setitem__(self, *args):
    memmap = self.as_memmap();
    memmap.__setitem__(*args);
   
  
  def metadata(self, info = None):
    """Returns metadata from this tif file.
  
    Arguments
    ---------
    source : str or Source
      The filename or data source.
    info : list or all
      Optional list of keywords, if all return full tif metadata, if None return default set info.
    
    Returns
    -------
    metadata : dict
      Dictionary with the meta data.
    """
    md = None;
    for t in self._tif.flags:
      if hasattr(self._tif, t + '_metadata'):
        md = getattr(self._tif, t + '_metadata');
        if md is not None:
          break;

    if md is None:
      md = {};
      
    if info is all:
      return md;
    elif info is None:
      info = ['shape', 'resolution', 'overlap'];
    elif isinstance(info, str):
      info = [info];
    info = {k : None for k in info};
        
    def update_info(info, name, keys, mdict, astype, include_keys=False):
      value = [];      
      for k in keys:
        try:
          v = mdict;
          for kk in k.split('.'):
            v = v.get(kk, None);
            if v is None:
              break;
          if include_keys and v is not None:
            info[k] = v;
          value.append(astype(v));
        except Exception:
          pass
      if len(value) > 0:
        info[name] = tuple(value);
    
    #get info
    mdp = md.get('Image', {}).get('Pixels', {});
    keys = info.keys();    
    
    if 'shape' in keys:
      #info['shape'] = self.shape;
      order = mdp.get('DimensionOrder', None);
      if order is None:
        order = ''.join([d for d in 'XYZTC' if 'Size' + d in mdp.keys()]);
      info['order'] = order;
      skeys = ['Size' + d for d in order];
      update_info(info, 'shape', skeys, mdp, int);
    
    if 'description' in keys:
      info['description'] = self._tif.pages[0].description;
      
    if 'resolution' in keys:
      rkeys = ['PhysicalSizeX', 'PhysicalSizeY', 'PhysicalSizeZ'];
      update_info(info, 'resolution', rkeys, mdp, float);
    
    if 'overlap' in keys:
      mdc = md.get('CustomAttributes', {}).get('PropArray',{});
      okeys = ['xyz-Table_X_Overlap.Value', 'xyz-Table_Y_Overlap.Value'];
      update_info(info, 'overlap', okeys, mdc, float); 
    
    return info;

  
  def as_memmap(self):
    try :
      return array_from_tif(tif.memmap(self.location));
    except:
      raise ValueError('The tif file %s cannot be memmaped!' % self.location);
  
  
  def as_virtual(self):
     return VirtualSource(source = self);
     
  def as_real(self):
    return self;
  
  def as_buffer(self):
    return self.as_memmap();
  
  
  ### Formatting
  def __str__(self):
    try:
      name = self.name;
      name = '%s' % name if name is not None else '';
    except:
      name ='';
    
    try:
      shape = self.shape
      shape ='%r' % ((shape,)) if shape is not None else '';
    except:
      shape = '';

    try:
      dtype = self.dtype;
      dtype = '[%s]' % dtype if dtype is not None else '';
    except:
      dtype = '';
            
    try:
      order = self.order;
      order = '|%s|' % order if order is not None else '';
    except:
      order = '';
    
    try:
      location = self.location;
      location = '%s' % location if location is not None else '';
      if len(location) > 100:
        location = location[:50] + '...' + location[-50:]
      if len(location) > 0:
        location = '{%s}' % location;
    except:
      location = '';    
    
    return name + shape + dtype + order + location


class VirtualSource(src.VirtualSource):
  def __init__(self, source = None, shape = None, dtype = None, order = None, location = None, name = None):
    super(VirtualSource, self).__init__(source=source, shape=shape, dtype=dtype, order=order, location=location, name=name);
    if isinstance(source, Source):
      self.multi_file = source.multi_file;
      self.series = source._series;
  
  @property 
  def name(self):
    return 'Virtual-Tif-Source';
  
  def as_virtual(self):
    return self;
  
  def as_real(self):
    return Source(location=self.location, series=self.series, multi_file=self.multi_file);
  
  def as_buffer(self):
    return self.as_real().as_buffer();


###############################################################################
### IO functionality
###############################################################################


def is_tif(source):
  """Checks if this source a TIF source"""
  if isinstance(source, Source):
    return True;
  if isinstance(source, str):
    try:
      Source(source);
    except:
      return False;
    return True;
  return False;
 

def read(source, slicing = None, sink = None, **args):
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
    source = Source(source);
  if slicing is None:
    return source.array
  else:
    return source.__getitem__(slicing);


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
  tif.imsave(sink, array_to_tif(data))
  return sink;


def create(location = None, shape = None, dtype = None, mode = None, as_source = True, **kwargs):
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
    raise ValueError('Shape for new tif file must be given!');
  shape = shape_to_tif(shape);
  mode = 'r+' if mode == 'w+' or mode is None else mode;
  dtype = 'float64' if dtype is None else dtype;
  
  memmap = tif.memmap(filename=location, shape=shape, dtype=dtype, mode=mode);
  if as_source:
    return Source(location);
  else:
    return memmap;



################################################################################
#### Array axes order
################################################################################
 
def shape_from_tif(shape):
  ndim = len(shape);
  shape = shape[:max(0,ndim-3)] + shape[-3:][::-1];
  return shape;

  
def shape_to_tif(shape):
  return shape_from_tif(shape);


def array_from_tif(array):
  ndim = array.ndim;
  axes = [d for d in range(ndim)];
  axes = axes[:max(0,ndim-3)] + axes[-3:][::-1];
  array = array.transpose(axes);
  return array;


def array_to_tif(array):
  return array_from_tif(array);

################################################################################
#### Meta data
################################################################################
#
#
#
#def change_OME_meta_data_string(description, info  = None):
#  """Changes the meta data in an ome image descriptor
#  
#  Arguments:
#    description (str): xml ome image description
#    info (dict): dictionary of entries to try to change
#  
#  Returns:
#    str: modified xml image descriptor
#  """
#  if not isinstance(info, dict):
#    return description;
#    
#  try:
#    xml = etree.fromstring(description);
#  except:
#    raise RuntimeError('could not parse ome xml description!');
#  
#  keys = info.keys();
#  
#  if 'overlap' in keys:
#    try:
#      #get the overlap
#      overlap = info['overlap'];
#      ex = [x for x in xml.iter('{*}xyz-Table_X_Overlap')][0];
#      ey = [x for x in xml.iter('{*}xyz-Table_Y_Overlap')][0];
#      ex.attrib['Value'] = str(overlap[0]);
#      ey.attrib['Value'] = str(overlap[1]);
#    except:
#      raise RuntimeWarning('could not change overlap in ome image description');
#    
#  #add other meta data keys here
#  
#  return etree.tostring(xml, pretty_print = False);


################################################################################
#### Tests
################################################################################
  
def _test():
  import ClearMap.Tests.Files as tfs
  import ClearMap.IO.TIF as tif
  reload(tif)
  
  filename = tfs.filename('tif_2d');
  t = tif.Source(location = filename);
  print(t)
  
  filename = tfs.filename('tif_2d_color');
  t = tif.Source(location = filename);
  print(t)
  
  d = tif.read(filename);
  print(d.shape)
  
  v = t.as_virtual()
  print(v)
  
  q = v.as_real()
  print(q)
  
  
  #filename = '/home/ckirst/Science/Projects/WholeBrainClearing/Vasculature/Experiment/17-19-19_IgG_UltraII[02 x 06]_C00_UltraII Filter0000.ome.tif';
  #t = tif.Source(location = filename);
  #print(t)  
  
