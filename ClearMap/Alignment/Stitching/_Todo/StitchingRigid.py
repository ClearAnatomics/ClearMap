# -*- coding: utf-8 -*-
"""
Stitching base module for aligning and stitching data sets.

The module features a plane wise alignment routine for data in which individual image stacks show oscillatory movements in itself.
"""
__author__    = 'Christoph Kirst <christoph.kirst.ck@gmail.com>'
__license__   = 'GPLv3 - GNU General Pulic License v3 (see LICENSE)'
__copyright__ = 'Copyright © 2020 by Christoph Kirst'
__webpage__   = 'http://idisco.info'
__download__  = 'http://www.github.com/ChristophKirst/ClearMap2'

import copy
import warnings

import itertools as itt
import functools as ft
import inspect as insp

import numpy as np
import multiprocessing as mp
import concurrent.futures
 
import graph_tool as gt
import graph_tool.topology as gtt

import matplotlib.pyplot as plt

import ClearMap.IO.IO as io
import ClearMap.IO.Source as src
import ClearMap.IO.Slice as slc
import ClearMap.IO.FileList as fl

import ClearMap.ParallelProcessing.ParallelTraceback as ptb

import ClearMap.Visualization.Plot3d as p3d
import ClearMap.Visualization.Color as col

import ClearMap.Utils.TagExpression as te
import ClearMap.Utils.Timer as tmr

from ClearMap.Utils.Formatting import ensure


########################################################################################
### Regions
########################################################################################


class Region(object):
  """Class to handle rectangular regions storing positional information."""
  
  #__slots__ = ('_position', '_shape');  
  
  def __init__(self, position = None, shape = None, lower = None, upper = None):
    """Region construtor.
    
    Arguments
    ---------
    lower, upper: tuples of int
      The corners of the rectangular region.
    """
    if lower is not None:
      position = ensure(lower, tuple); 
    if upper is not None:
      shape = tuple(u - l for u,l in zip(upper,position));
   
    self._position = ensure(position, tuple);
    self._shape    = ensure(shape, tuple);
  
  
  @property
  def position(self):
    """The position of the region.
    
    Returns
    -------
    position : tuple of ints
      The position of the region.
    """
    return self._position;
    
  @position.setter
  def position(self, position):
    self._position = ensure(position, tuple);
  
  
  @property
  def shape(self):
    """The shape of the region.
    
    Returns
    -------
    shape : tuple of int
      The shape of the layout.
    """
    return self._shape;
    
  @shape.setter
  def shape(self, shape):
    self._shape = ensure(shape, tuple);
  
  
  @property 
  def lower(self):
    """The lower corner of the source's placement.
    
    Returns
    -------
    lower : array of int
      The coordinates of the lower corner of the source's placment.
    """
    return self.position;
    
  @lower.setter
  def lower(self, lower):
    self._position = ensure(lower, tuple);
  
  
  @property 
  def upper(self):
    """The upper corner of the source's placement.
    
    Returns
    -------
    upper : array of int
      The coordinates of the upper corner of the source's placment.
    """
    return tuple(p + s for p,s in zip(self.position, self.shape));
    
  @upper.setter
  def upper(self, upper):
    position = self.position;
    self._shape = tuple(u - p for u,p in zip(upper, position));

  
  @property 
  def extent(self):
    """The difference between upper and lower corner.
    
    Returns
    -------
    extent : tuple of int
      The difference between upper and lower corner.
    
    Note
    ----
    The exten can differ from the shape in case the source is bended or wobbly.
    """
    return tuple(u - l for l,u in zip(self.lower, self.upper));
  
  
  @property
  def origin(self):
    """The origin of this region, i.e. its rectified position.
    
    Returns
    -------
    origin : tuple of ints
      The rectified position, i.e. coordiantes below zero are set to zero.
    """
    return tuple(p if p >= 0 else 0 for p in self.position);
    
  
  @property
  def ndim(self):
    """The dimension of the source.
    """
    return len(self._position);
  
  
  def position_to_local(self, position):
    """Converts a position to the a local position wrt to the sources origin.
    
    Arguments
    ---------
    position : tuple of ints
      The non-local position.
    
    Returns
    -------
    loacl_position : tuple of ints
      The local position within this region.
    """
    return tuple(p - q for p,q in zip(position, self.position));
    
  
  def position_from_local(self, local_position):
    """Converts a position given in local coordiantes to the non-local position.
    
    Arguments
    ---------
    local_position : tuple of ints
      The local position within the region.
    
    Returns
    -------
    position : tuple of ints
      The non-local position.
    """
    return tuple(p + q for p,q in zip(local_position, self.position));

  
  def coordinate_to_local(self, coordinate, axis):
    """Converts a coordinate along an axis to the a local coordinate.
    
    Arguments
    ---------
    coordinate : int
      The non-local coordinate along the axis.
    axis : int
      The axis of the coordiante.
    
    Returns
    -------
    loacl_coordainte : tuple of ints, or int
      The local coordinate within this region.
    """
    return coordinate - self.position[axis];
    
  
  def coordinate_from_local(self, loacl_coordinate, axis):
    """Converts a local coordainte along an axis to the non-local coordinate.
    
    Arguments
    ---------
    local_coordinate : int
      The local coordinate along the axis.
    axis : int
      The axis of the coordiante.
    
    Returns
    -------
    coordainte : int
      The non-local coordainte.
    """
    return loacl_coordinate + self.position[axis];
  
  
  def local_slicing(self, source = None, position = None):
    """Returns the slice of this region in a source or from a position.
    
    Arguments
    ---------
    source : Source class
      Source at which this region should be sliced.
    position : tuple of ints
      Position of the lower corner of the sink.
    
    Returns
    -------
    slice : list of slice objects
      The slice specifications for this region in a source.
      
    Note
    ----
    This routine assumes that the region is within the source. 
    No boundary checks are performed.
    """
    if source is not None:
      position = source.position; 
    if position is None:
      position = (0,) * self.ndim;
      #raise ValueError('Either the source or position argument mst be given!')
    return tuple(slice(l - p, u - p) for l,u,p in zip(self.lower, self.upper, position));
  
  
  def copy(self):
    return copy.copy(self);
  
  
  def __str__(self):
    try:
      lower = self.lower;
    except:
      lower = None;
    
    try:
      upper = self.upper;
    except:
      upper = None;
    
    s = "Region[%r, %r]" % (lower, upper);
    return s;
  
  def __repr__(self):
    return self.__str__(); 



class Overlap(Region):
  """Class to handle overlapping regions of aligned sources."""
  
  def __init__(self, position = None, shape = None, lower = None, upper = None, sources = None):
    """Region construtor.
    
    Arguments
    ---------
    lower, upper: tuples of int
      The corners of the rectangular region.
    sources : list of Source classes
      The sources that contribute to this region.
    """
    super(Overlap, self).__init__(position = position, shape = shape, lower = lower, upper = upper);
    if sources == None:
      sources = ();
    self._sources = ensure(sources, tuple);
  
  
  @property 
  def sources(self):
    """The sources that contribute to this overlap region.
    
    Returns
    -------
    sources : list of Source classes
      The sources that contribute to this region.
    """
    return self._sources;
    
  @sources.setter
  def sources(self, sources):
    self._sources = ensure(sources, tuple);
  
    
  def source_slicings(self):
    """Returns the slices to obtain the data of this overlap region from the contributing sources
    
    Returns
    -------
    slices : list of list of slice objects
      The slice specifications for the sources.
    """
    return [self.local_slicing(position = s.position) for s in self.sources];
  
  
  def source_arrays(self):
    """Returns the data arrays of the sources in this region 
    
    Returns
    -------
    data : list of arrays
      The data of all the sources that overlap in this region.
    """
    return [s[sl] for s,sl in zip(self.sources, self.source_slicings())];
  
  
  def plot(self):
    """Plots the overlap region"""
    return p3d.plot(self.source_arrays());
  
  def __copy__(self):
    new = type(self)();
    new.__dict__.update(self.__dict__);
    new._sources = list(self._sources);
    return new;
  
  def __str__(self):
    try:
      lower = self.lower;
    except:
      lower = None;
    
    try:
      upper = self.upper;
    except:
      upper = None;
      
    try:
      n_sources = len(self.sources);
    except:
      n_sources = None;
    
    s = "Overlap[%r, %r]<%r>" % (lower, upper, n_sources);
    return s;



########################################################################################
### Sources
########################################################################################

class SourceRegion(Region):
  """Class to signal a Source with a region.
  
  Note
  ----
  This class serves to identify all sources with positional information.
  """
  #def __init__(self, *args, **kwargs):
  #  super(SourceRegion, self).__init__(*args, **kwargs);
  pass

class Source(SourceRegion, src.AbstractSource):
  """Class to handle basic data sources in a layout for stitching."""
  
  #__slots__ = ('_position', '_shape', '_dtype', '_order', '_location')
  
  counter = 0;
  """Counter for the sources used to create unique ids.""" 
  
  def __init__(self, source = None, position = None, tile_position = None):
    """LayoutSource class construtor.
    
    Arguments
    ---------
    source: string, array or Source class
      The data source.
    position : tuple of int or None
      The position of the source's 'lower' corner in the layout.
    tile_position : array or None
      Optional position of this source in a tiling grid. 
    """
    if source is not None:
      source = io.as_source(source);
      sid = None;                           
    if isinstance(source, Source):
      position = source.position if position is None else position;
      tile_position = source.tile_position if tile_position is None else tile_position;
      sid = source.id;
      source = source.source;

    shape = source.shape; 
    if shape is None:
      raise ValueError('Cannot initilaize source without a shape!');    
    
    if position is None:
      position = (0,) * len(shape);
    
    SourceRegion.__init__(self, position = position, shape = shape);

    src.AbstractSource.__init__(self, source = source);   
    self._source = source;
    
    self._tile_position = ensure(tile_position, tuple);
    
    if sid is None:                              
      self._id = Source.counter;
      Source.counter += 1;
    else:
      self._id = sid;
  
  
  @property
  def name(self):  
    return 'Stitchable-' + self.source.name;
  
  
  @property
  def id(self):
    """The source id.
    
    Returns
    -------
    id : int
      The id of the source.
    
    Note
    ----
    Parallel processing changes pointers, this id can be used to identify the same source.
    """
    return self._id;
    
  @id.setter
  def id(self, id):
    self._id = id;
    
  def __eq__(self, other):
    if isinstance(other, Source):
      return self.id == other.id;
    else:
      return False;
  
    
  @property 
  def source(self):
    """The underlying IO source."""
    return self._source;
  
  
  @property
  def tile_position(self):
    """Optional position on a grid.
    
    Returns
    -------
    tile_position : tuple of int
      The tile position of this source on a grid.
    """
    return self._tile_position;
 
  @tile_position.setter
  def tile_position(self, tile_position):
    self._tile_position = ensure(tile_position, tuple);
  
  
  def slice_along_axis(self, coordinate = None, local_coordinate = None, axis = 2):
    """Slice this source at a coordinate along an axis.
    
    Arguments
    ---------
    coordinate : int
      The coordinate at which to take the slice.
    local_coordinate : int
      The local coordinate in the underlying source along the slice axis.
    axis : int
      The axis to take the slice in.
      
    Returns
    -------
    source : SlicedSource class
      The sliced source.

    Note
    ----
    Either coordiante or local_coordinate must be given.
    """
    
    if local_coordinate is None:
      if coordinate is None:
        raise ValueError('Either a coordiante or a base_coordainte needs to be given!');
      local_coordinate = self.coordinate_to_local(coordinate, axis);
    
    if not 0 <= local_coordinate < self.shape[axis]:
      raise ValueError('The source cannot be sliced at the base coordinate %d!' % local_coordinate);

    slicing = [slice(None)] * self.ndim;
    slicing[axis] = local_coordinate;
    slicing = tuple(slicing);
    
    return Slice(source=self, slicing=slicing)
  
  
  def as_virtual(self):
    new = self.copy();
    new._source = self._source.as_virtual();
    return new;
    
  def as_real(self):
    new = self.copy();               
    new._source = self._source.as_real();
    return new;    
  
  def __getitem__(self, *args):
    return self._source.__getitem__(*args);
    
  #def __setitem__(self, *args):
  #  self._source.__setitem__(*args);
  
  def __str__(self):
    return _source_string(self);
  
  # other source attributes
  #def __getattr__(self, name):
  #  print 'getattr', self.__class__, name
  #  print self._source.__class__
  #  if not hasattr(self._source, name):
  #    raise AttributeError('The source does not have the atrribute %s!' % name);
  #  return getattr(self._source, name);



class Slice(slc.Slice, SourceRegion):
  """Class to handle a slice of a stitchable source.""" 
  
  def __init__(self, source, position = None, tile_position = None, slicing = None):
    """Source class construtor.
    
    Arguments
    ---------
    source: string, array or Source class
      The image source.
    slicing : slice specification
      The slice specification to obtain this slice.
    """
    if not isinstance(source, SourceRegion):
      source = Source(source = source, position = position, tile_position = tile_position);
    
    SourceRegion.__init__(self, position = None, shape = None);
    slc.Slice.__init__(self, source = source, slicing = slicing);
  
  
  @property 
  def position(self):
    if self._position is None:
      start = slc.sliced_start(self.slicing, self.source.shape);
      reduction = slc.sliced_reduction(self.slicing, self.source.ndim);
      return tuple(self.source.position[r] + start[r] for r in reduction);
    else:
      return self._position;
  
  @position.setter
  def position(self, value):
    self._position = ensure(value, tuple);
    #raise RuntimeError('Cannot set position of this sliced source!')
  
  
  @property
  def position_unsliced(self):
    """Returns the position of this slice in the underlying space taking into account the source position.
        
    Returns
    -------
    position : array of ints or int
      The position of the slice in the higher dimensional space of the source.
    """

    if self._position is None:
      start  = slc.sliced_start(self.slicing, self.source.shape);
      return tuple(p + s for p,s in zip(self.source.position, start));
    else:
      shape = self.source.shape;
      start  = slc.sliced_start(self.slicing, shape);
      reduction = slc.sliced_reduction(self.slicing, len(shape));
      source_position = self.source.position
      position = ();
      rd = 0;
      for d in range(self.source.ndim):
        if d in reduction:
          position += (self._position[rd],);
          rd += 1;
        else:
          position += (source_position[d] + start[d],);
      return position;
  
  
  @property
  def id(self):
    return self._source.id;
  
  @property
  def tile_position(self):
    return self._source.tile_position;
  
  
  def __str__(self):
    return _source_string(self);
  
  # source attributes
  #def __getattr__(self, name):
  #  if not hasattr(self.source, name):
  #    raise AttributeError('The source does not have the atrribute %s!' % name);
  #  return getattr(self.source, name);
  
    
  def as_virtual(self):
    new = self.copy();
    new._source = self._source.as_virtual();
    return new;
    
  def as_real(self):
    new = self.copy();
    new._source = self._source.as_real();
    return new;    


def _source_string(self):
  """Generates a string  desribing a region source."""
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
    location = self.source.location;
    location = '%s' % location if location is not None else '';
    if len(location) > 25:
      #location = location[:25] + '...' + location[-25:]
      location = '...' + location[-25:];
    if len(location) > 0:
      location = '{%s}' % location;
  except:
    location = '';    
  
  try:
    position = "P%r" % (list(self.position),);
  except:
    position = '';
  
  try: 
    tile_position = "T%r" % (tuple(self.tile_position),);
  except:
    tile_position = '';
  
  try:
    ids = '(#%d)' % self.id;
  except:
    ids = '';
  
  return name + ids + tile_position + position + shape + dtype + order + location


########################################################################################
### Layouts
########################################################################################



class AlignmentBase(object):
  """Base class to handle alignments between two adjacent sources."""
  #note: could make this a source like object with data and plot routines
  
  def __init__(self, pre = None, post = None):
    """Alignment construtor.
    
    Arguments
    ---------
    pre, post: Source classes
      The pointers to the source classes of the adjacent images.
    shift : tuple of int
      Additional shift between the pre source and post source positions to better align them.
    displacement : tuple of int 
      The optimal displacement between the positions of the pre and post source.
    quality : number or None
      The quality of the alignment.
    """
    if not isinstance(pre, SourceRegion) or not isinstance(post, SourceRegion):
      raise ValueError('Pre and post specifications need to be Source classes!')
    self._pre = pre;
    self._post = post;
  
  @property 
  def sources(self):
    """The sources that contribute to this alignment.
    
    Returns
    -------
    sources : list of Source classes
      The sources that contribute to this alignment.
    """
    return [self._pre, self._post];
  
  @sources.setter
  def sources(self, sources):
    self._pre, self._post = sources;
  
  
  @property 
  def pre(self):
    """The first source that contributes to this alignment.
    
    Returns
    -------
    source : Source classes
      The first source that contribute to this alignment.
    """
    return self._pre;
  
  @pre.setter
  def pre(self, pre):
    self._pre = pre;
  
  
  @property 
  def post(self):
    """The second source that contributes to this alignment.
    
    Returns
    -------
    source : Source classes
      The second source that contribute to this alignment.
    """
    return self._post;
  
  @post.setter
  def post(self, post):
    self._post = post;
  
  
  @property
  def ndim(self):
    """Dimension of the alignment sources.
    
    Returns
    -------
    ndim : int
      The dimension of the sources.
    """
    return self.pre.ndim;
  
  
  def plot(self, *args, **kwargs):
    """Plots this alignment"""               
    plot_sources(self.sources, *args, **kwargs);
  
  def overlay(self, **kwargs):
    return overlay_sources(a.pre, a.post, **kwargs);
  
  
def plot_overlay(self, **kwargs):
  ovl = self.overlay(**kwargs);
  p3d.plot([ovl[...,0], ovl[...,1]]);
  
  
  
  
  def copy(self):
    return copy.copy(self);
  
  def __str__(self):  
    pre_id = self.pre.tile_position;
    if pre_id is None:
      pre_id = self.pre.id;
    
    post_id = self.post.tile_position;
    if post_id is None:
      post_id = self.post.id;
    
    return "Alignment(%r->%r)" % (pre_id, post_id);
  
  def __repr__(self):
    return self.__str__();



class Alignment(AlignmentBase):
  """Class to handle rigid alignments between two adjacent sources."""
  #note: could make this a source like object with data and plot routines
  
  def __init__(self, pre = None, post = None, shift = None, displacement = None, quality = None):
    """Alignment construtor.
    
    Arguments
    ---------
    pre, post: Source classes
      The pointers to the source classes of the adjacent images.
    shift : tuple of int
      Additional shift between the pre source and post source positions to better align them.
    displacement : tuple of int 
      The optimal displacement between the positions of the pre and post source.
    quality : number or None
      The quality of the alignment.
    """
    AlignmentBase.__init__(self, pre=pre, post=post);
    
    if displacement is None:
      if shift is None:
        displacement = tuple(p - q for p,q in zip(post.position, pre.position));
      else:
        displacement = tuple(p - q + s for p,q,s in zip(post.position, pre.position, shift));
    self._displacement = ensure(displacement, tuple);
    
    if quality is None:
      self._quality = -np.inf;
    else:
      self._quality = float(quality);
  
  
  @property  
  def displacement(self):
    """The displacement between the two sources given their positions and the alignment shift.
    
    Returns
    -------
    displacmeent : array of int
      The displacement between the sources.
    """
    #return tuple(p - q + s for p,q,s in zip(self.post.position, self.pre.position, self.shift));
    return self._displacement;
  
  @displacement.setter
  def displacement(self, displacement):
    #self.shift = tuple(d - (p - q) for d,p,q in zip(displacement, self.post.position, self.pre.position))
    self._displacement = ensure(displacement, tuple);
  
  
  @property 
  def shift(self):
    """The additional shift between the source positions that better aligns them.
    
    Returns
    -------
    shift : array of int
      The additional shift between the source positions that better aligns them.
    """
    #return self._shift;
    return tuple(p + d - q for p,q,d in zip(self.pre.position, self.post.position, self.displacement));
  
  @shift.setter
  def shift(self, shift):
    #self._shift = ensure(shift, tuple);
    self._displacement = tuple(q + s - p for p,q,s in zip(self.pre.position, self.post.position, shift))
  
  
  @property 
  def quality(self):
    """The quality of this alignment.
    
    Returns
    -------
    quality : float
      The quality of this alignment.
    """
    return self._quality;
  
  @quality.setter
  def quality(self, quality):
    self._quality = float(quality);
  
  
  def plot(self, *args, **kwargs):
    """Plots this alignment"""
    post = self.post.copy();
    post.position = tuple(p + d for p,d in zip(self.pre.position, self.displacement));                         
    return plot_sources([self.pre, post], *args, **kwargs);
  
  
  
  
  def plot_mip(self, *args, **kwargs):
    """Plots this alignment"""
    plot_along_axis_mip(self.pre, self.post, *args, **kwargs)
  
  
  def __str__(self):
    quality = self.quality;
    if quality is not None:
      quality = '%.2e' % quality; 
   
    pre_id = self.pre.tile_position;
    if pre_id is None:
      pre_id = self.pre.id;
    
    post_id = self.post.tile_position;
    if post_id is None:
      post_id = self.post.id;
    
    return "Alignment(%r->%r)D%rS%r[%s]" % (pre_id, post_id, self.displacement, self.shift, quality);





class Layout(SourceRegion, src.AbstractSource):
  """Base class to handle the layout of multiple sources."""
  
  def __init__(self, sources, alignments = None, position = None, shape = None, dtype = None, order = None, location = None):
    """Layout constructor.
    
    Arguments
    ---------
    sources : list of filenames or Source classes
      List of the sources of the individual sources contributing to the full image.
    alignments : list of Alignment classes
      The alignment structure of the sources.
    shape : tuple of int or None
      The fixed shape of this Layout, if None the minimal size to fit all sources will be used.
    dtype: dtype or None
      The data type to use for this layout, if None use the dtype of the first source.
    order : 'C", 'F' or None
      Contiguous order of the layout array.
    position : tuple of int or None
      The fixed position of this layout, if None the lower corner to fit all sources will be used.
    """
    SourceRegion.__init__(self, position = position, shape = shape);
    src.AbstractSource.__init__(self, source = None, shape = shape, dtype = dtype, order = order, location = location);
    
    self._sources = [s if isinstance(s, SourceRegion) else Source(source = s) for s in sources]
    self._alignments = [] if alignments is None else alignments;
    
  
  @property
  def sources(self):
    """The sources in the layout.
    
    Returns
    -------
    sources : list 
      List of the sources in this layout.
    """
    return self._sources;
   
  @sources.setter
  def sources(self, sources):
    self._sources = sources;
  
  @property
  def n_sources(self):
    """Number of sources in the layout.
    
    Returns
    -------
    n_sources : int
      Number of sources in this layout.
    """
    return len(self.sources);
  
  
  @property
  def alignments(self):
    """The alignments in the layout.
    
    Returns
    -------
    alignments : list 
      List of the alignments in this layout.
    """
    return self._alignments;
   
  @alignments.setter
  def alignments(self, alignments):
    self._alignments = alignments;
  
  
  @property
  def n_alignments(self):
    """Number of alignments in this layout.
    
    Returns
    -------
    n_alignments : int
      Number of alignments in this layout.
    """
    return len(self.alignments);
  
  
  @property
  def ndim(self):
    """Dimension of the alignment sources.
    
    Returns
    -------
    ndim : int
      The dimension of the sources.
    """
    return self.sources[0].ndim;
  
  
  
  ### Geometry
   
  @property    
  def position(self):
    """Returns the lower position of the layout.
    
    Returns
    -------
    position : tuple of ints
      The position of the lower corner of this layout.
    """
    if self._position is not None:
      return self._position;
    else:
      return self.lower; 
    
  @position.setter
  def position(self, position):
    self._position = ensure(position, tuple);
  
  
  @property    
  def lower(self):
    """Calculates the lower position of the entire layout.
    
    Returns
    -------
    lower : tuple of ints
      The lower position of the full layout.
    """
    return tuple(np.min([s.lower for s in self.sources], axis = 0));
  
  
  @property   
  def upper(self):
    """Calculates the upper position of the entire layout.
    
    Returns
    -------
    upper : tuple of ints
      The upper position of the full layout.
    """
    return tuple(np.max([s.upper for s in self.sources], axis = 0));
  
  
  @property
  def shape(self):
    """Shape of the layout.
    
    Returns
    -------
    shape : tuple of int
      The shape of the layout when stitching together all the sources.
    """
    if self._shape is not None:
      return self._shape;
    else:
      return tuple(u - o for u,o in zip(self.upper, self.origin));
  
  @shape.setter
  def shape(self, shape):
    self._shape = ensure(shape, tuple);
  
  
  def lower_to_origin(self):
    """Moves the sources so that the lower corner is at the origin."""
    lower = self.lower;
    source_positions = [tuple(p - l for p,l in zip(positions, lower)) for positions in self.source_positions()];
    self.set_source_positions(source_positions);
  
  
  def source_positions(self, sources = None):
    """Returns the positions of the sources.
    
    Returns
    -------
    positions : list of tuples of ints
      The source positions.
    """
    if sources is None:
      sources = self.sources;
    return [s.position for s in sources];

  
  def set_source_positions(self, positions = None, sources = None, update_alignments = False):
    """Sets the positions of the sources.
  
    Aruments
    --------
    positions : list of tuple of ints or None
      The new positions of the sources, if None infer a consistent solution from the alignments.
    sources : list of Source classes
      If only a subset of positions is given, this list represents the sources of those positions.
    update_alignments : bool
      If True, also update the alignments shifts to match the new positions.
    """
    if sources is None:
      sources = self.sources;

    if positions is None:
      positions = positions_from_tree(alignments = self.alignments, sources = sources);
    
    old_positions = self.source_positions(sources = sources);
    
    for s,p in zip(sources, positions):
      s.position = p;
    
    if update_alignments:
      if sources == self.sources:
        for a in self.alignments:
          a.shift = (0,) * self.ndim;
      else:
        new_positions = self.source_positions(sources = sources);
        delta = [tuple(n-o for n,o in zip(npos, opos)) for npos,opos in zip(new_positions, old_positions)];
        sources_to_index = { s: i for i,s in enumerate(sources)};
        for a in self.alignments:
          pre = a.pre in sources;
          post = a.post in sources;
          
          if pre and post:
            a.shift = (0,) * self.ndim;
          elif pre:
            i = sources_to_index[a.pre];
            a.shift = tuple(s + d for s,d in zip(a.shift, delta[i]));
          elif post:
            i = sources_to_index[a.post];
            a.shift = tuple(s - d for s,d in zip(a.shift, delta[i]));
  
  
  def sink_slicing(self):
    """Returns the slice of this layout's data in an underlying sink.
    
    Returns
    -------
    slice : list of slice objects
      The slice to use if this layout is placed in an underling sink.
      
    Note
    ----
    Positions below zero are cut off as well as above the shape.
    """
    return tuple(slice(o, o + s) for o,s in zip(self.origin, self.shape));
  
  
  ### IO
  
  @property
  def dtype(self):
    """Data type of the sources in the layout.
    
    Returns
    -------
    dtype : dtype
      Data type of the sources in this layout.
    """
    if self._dtype is not None:
      return self._dtype;
    else:
      return self.sources[0].dtype;
  
  @dtype.setter
  def dtype(self, dtype):
    self._dtype = dtype;
  
  
  @property 
  def order(self):
    """The contiguous order of the source.
    
    Returns
    -------
    order : order
      The contiguous order of the source.
    """
    if self._order is not None:
      return self._order;
    else:
      return self.sources[0].order;
  
  
  @order.setter
  def order(self, order):
    self._order = ensure(order, str);
  
  
  @property 
  def location(self):
    """The location of the layout when written.
    
    Returns
    -------
    location : str
      The location of the layout when written.
    """
    return self._location;
  
  @location.setter
  def location(self, location):
    self._location = ensure(location, str);
  
  
  ### Functionality
  
  def source_index(self, source):
    """The id of a source in the list of sources.
    
    Arguments
    ---------
    source : Source class
    
    Returns
    -------
    id : int or None
      Position of the source in the sources list, None if not found.
    """
    for i,s in enumerate(self.sources):
      if s == source:
        return i;  
    return None;
  
  
  def update_alignments_from_sources(self):
    """Updates the alignments from the source positions."""
    zero = (0,) * self.ndim;
    for a in self.alignments:
      a.shift = zero;
  
  
  def update_sources_from_alignments(self):
    """Updates the source positions from the alignments."""
    positions = positions_from_tree(sources = self.sources, alignments = self.alignments, fixed_source=self.sources[0]);
    self.set_source_positions(positions, update_alignments = True);
  
  
  def remove_source(self, source):
    """Removes a source from this Layout.
    
    Arguments
    ---------
    source : int, Source class or list of ints or Source classes
      The list of source classes to remove from this layout.
      
    Note
    ----
    The alignments are cleaned in a consistent way when this routine is used.
    """
    if not isinstance(source, (tuple, list)):
      source = [source];
    
    dels = [];
    for i,si in enumerate(source):
      if isinstance(si, int):
        si = self.sources[si];
        source[i] = si;
      sid = self.source_index(si);
      if sid is not None:
        dels.append(sid);
    
    dela = [];
    for i,a in enumerate(self.alignments):
      if a.pre in source or a.post in source:
        dela.append(i);
    
    self.sources = [s for i,s in enumerate(self.sources) if i not in dels];
    self.alignments = [a for i,a in enumerate(self.alignments) if i not in dela];
  
  
  def change_sources(self, sources):
    """changes the sources of this layout.
    
    Arguments
    ---------
    sources : list of sources
      The new sources.
      
    Note
    ----
    This allows to stitch other color channels with the same alignment.
    """
    if len(sources) != len(self._sources):
      raise ValueError('The number of sources %d and layout sources %d do not match!' % (len(sources), len(self._sources)));
    
    old_sources = self._sources;
    old_to_new = {o : n for o,n in zip(old_sources, sources)};
    
    alignments = [a.copy() for a in self._alignments];    
    for a in alignments:
      a.pre = old_to_new[a.pre];
      a.post = old_to_new[a.post];
    
    for s,o in zip(sources, old_sources):
      s.position = o.position;
      s.tile_position = o.tile_position;
     
    self._alignments = alignments;
    self._sources = sources;
  
  
  def change_source_location(self, expression, substitutions):
    """Change the sources to point to a new location.
      
    Arguments
    ---------
    expression : str
       Tag expression of source names with additional substitution tags.
    substitutions : dict
      A substitution dictionary of the form {tag_name : value}, 
      specifying how to replace the substitution tags.
      
    Note
    ----
    This function is useful to stitch other color channels of imagining data
    using the same alignments.
    """
    if not isinstance(expression, te.Expression):
      expression = te.Expression(expression);
    
    #get locations
    locations = [s.location for s in self._sources];
    for l in locations:
      if l is None:
        raise RuntimeError('The layout contains sources without locations!');
    
    #change location expressions
    locations = [expression.string(expression.values(l).update(substitutions)) for l in locations];

    sources = [s.copy() for s in self._sources];
    for s,l in zip(sources, locations):
      s.source.location = l;
    
    self.change_sources(sources);
  
  
  def sort_sources_by_position(self):
    """Sorts the sources of this layout by their current position."""
    pl = self.source_positions();
    p  = np.zeros(len(pl), dtype = object);
    for i in range(len(p)):
      p[i] = pl[i];
    sort_id = np.argsort(p);      
    self._sources = [self._sources[i] for i in sort_id];
  
  
  def connected_components(self, min_quality = None, with_sources = False):
    """Determines the connected components of the layout.
    
    Arguments
    ---------
    min_quality : float, tuple of floats or None
      The minimal quality needed to include an alignment in the calculation.
    with_sources : bool
       If True, also return the sources in each component.     
    
    Returns
    -------
    components : list of list of Alignment classes
      The connected components of the alignments.
    component_sources : list of list of Source classes
      The sources in each compoenent.      
    """
    return connected_components(alignments = self.alignments, sources = self.sources, min_quality = min_quality, with_sources = with_sources);
  
    
  def connected(self):
    """Returns True if the alignments form a single connected component.

    Returns
    -------
    connected : bool
      True if the alignments form a single connected component.
    """
    return connected(alignments = self.alignments, sources = self.sources);
  
  
  def embedding(self):
    """Splits the set of co-axial sources into a minimal set of non-overlaping regions.
    
    Returns
    -------
    shape : tuple of int
      The shape that encapsulates all the regions.
    position : tuple of int
      The lowest corner of all the regions.
    regions : list of Region classes.
      The regions of different overlaps of the individual sources.       
    
    Note
    ----
    The result can be used to stitch the images.
    """
    return embedding(sources = self._sources, shape = self._shape, position = self._position);
  
  
  def layout_from_region(self, region = None, position = None, shape = None, lower = None, upper = None):
    """Returns a layout with only the sources needed to construct the specified region
    
    Returns
    -------
    layout : Layout class
      The reduced layout needed to cover the region.    
    """
    if not isinstance(region, Region):
      if lower is None or upper is None:
        if position is None or shape is None:
          raise ValueError('Either a region or position and shape or lower and upper coordinates have to be given!');
        else:
          region = Region(position = position, shape = shape);
      else:
        region = Region(lower = lower, upper = upper);
    lower = region.lower;
    upper = region.upper;
    
    new = self.copy();
    rem = [s for s in self.sources if np.any([u < l for u,l in zip(s.upper, lower)]) \
                                      or np.any([l >= u for l,u in zip(s.lower, upper)])];
    new.remove_sources(rem);
    return new;
  
  
  def slice_along_axis(self, coordinate, axis = 2):
    """Returns a layout corresponding to a slice along a single axis in this layout.
    
    Arguments
    ---------
    coordinate : int
      The coordinate at which to take the slice.
    axis : int
      The axis to take the slice in.
    
    Returns
    -------
    layout : SlicedLayout class
      The sliced layout.
    """
    return slice_layout_along_axis(self, coordinate = coordinate, axis = axis);
  
  
  ### Alignment, placement ans stitching
  
  def align(self, max_shifts = 10, clip = None, background = None, processes = None, verbose = False):
    """Align the sources."""
    align_layout(self, max_shifts = max_shifts, clip = clip, background = background, processes = processes, verbose = verbose);
  
  
  def place(self, method = 'optimization', lower_to_origin = False, processes = None, verbose = False):
    """Optimizes positions of the sources in this layout."""
    place_layout(self, method = method, lower_to_origin = lower_to_origin, verbose = verbose);
  
  
  def stitch(self, sink = None, method = 'interpolation', processes = None, verbose = False):
    """Stitches the sources according to this layout.
   
    Arguments
    ---------
    sink : sink specification or None
      The sink to write the result to.
    method : str
      The method to use for the stitching: 'interpolation', 'max', 'min', 'mean'
  
    Returns
    -------
    stitched : array or sink
      The stitched array or sink.
    """
    return stitch_layout(self, method = method, sink = sink, verbose = verbose);
  
  
  def align_axis(self, depth = 10, max_shifts = 10, axis = None, axis_range = None, clip = None, background = None, processes = None, verbose = False):
    """Aligns sources in a layout along a single axis only.
    
    Arguments
    ---------
    layout: Layout class
      The layout in which to align the 3d sources in z-direction.
    depth : int or list of ints
      The approximate overlaps of the sources in the tiling dimensions to use 
      for mip projection when aligning the axis.
    max_shifts : tuple of ints
      The minmal and maximal shifts along all axes consider.
    axis : int
      The axis to aling the sources along.
    axis_range : tuple of int or None
      If not None, use only a sub set of the axis range to speed up processing.
    clip : number or None
      If not None, clip the soruces at this value when calculating the alignment.
    background : number or None
      If not None, if the values in the overlap region are less than this number make alignment return -inf quality as there is no signal to use for alignment. 
    processes : int or 'serial' 
      Number of processor to use for parallel processing, if 'serial' process in serial.
    verbose : bool
      Print progress information.
    """
    if axis is None:
      axis = self.axis;
    align_layout_axis(self, depth = depth, max_shifts = max_shifts, axis = axis, axis_range = axis_range, clip = clip, background = background, processes = processes, verbose = verbose);


  def place_axis(self, axis = None, method = 'optimization', min_quality = None, lower_to_origin = False, verbose = False):
    """Places the sources in a layout along a single axis only.
    
    Arguments
    ---------
    axis: int
      The axis which to place the sources along.
    method : 'optimization' or 'tree'
      The method to use to place the sources.
    min_quality : float
      The minimal quality of the alignment.
    lower_to_origin : bool
      If True the lower corner of the aligned images is set to zero.  
    """
    if axis is None:
      axis = self.axis;
    place_layout_axis(self, axis = axis, method = method, min_quality = min_quality, lower_to_origin = lower_to_origin, verbose = verbose);

  
  
  ### Data access 
  
  @property
  def array(self):
    """Returns the stitched data
    
    Arguments
    ---------
    sink : sink specification or None
      The sink to write the result to. If None return as array.
    
    Returns
    -------
    data :  array or sink
      The stitched array or sink.
    """
    return self.stitch(sink = None);
  
  
  def __getitem__(self, slicing):
    #TODO: speed up stitching for subslices etc -> for fast previews 
    return self.array.__getitem__(slicing);
  
  def __setitem__(self, slicing):
    raise ValueError('Cannot set data in a Layout.');

  
  def array_along_axis(self, sink = None, coordinate = None, axis = 2):
    """Returns the stitched data
    
    Arguments
    ---------
    sink : sink specification or None
      The sink to write the result to. If None return as array.
    coordinate : int
      The coordinate at which to take the slice.
    axis : int
      The axis to take the slice in.
    
    Returns
    -------
    data :  array or sink
      The stitched array or sink.
    """
    l = self.slice_along_axis(coordinate = coordinate, axis = axis);
    return l.stitch(sink = sink);
  
  
  def overlay(self, colors = None, percentile = 98, normalize = True, coordinate = None, axis = 2):
    """Overlays the sources to check their placement.
      
    Arguments
    ---------
    colors : list of tuple of floats or color names
      The optional RGB colors to use.
    percentile : int
      Use this percentile as upper cutoff in the resulting image to enhance contrast.
    normalize : bool
      If True normalize image to floats between 0 and 1.
    coordinate : int or None
      Optional coordinate at which to take a slice.
    axis : int
      Optional axis to take the slice in.
    
    Returns
    -------
    image : array
      A color image. 
    """
    if coordinate is None:
      layout = self;
    else:
      layout = self.slice_along_axis(coordinate = coordinate, axis = axis);
    return overlay_layout(layout, colors = colors, percentile = percentile, normalize = normalize);
  
  
  def plot(self, colors = None, percentile = 98, normalize = True, color_ids = None, coordinate = None, axis = 2):
    """Plots overlayed sources to check their placement.
      
    Arguments
    ---------
    colors : list of tuple of floats or color names
      The optional RGB colors to use.
    percentile : int
      Use this percentile as upper cutoff in the resulting image to enhance contrast.
    normalize : bool
      If True normalize image to floats between 0 and 1.
    color_ids : list of ints
      Use specific color ids for the sources contributing to the layout.
    coordinate : int or None
      Optional coordinate at which to take a slice.
    axis : int
      Optional axis to take the slice in.
    
    Returns
    -------
    image : array
      A color image. 
    """
    if coordinate is None:
      layout = self;
    else:
      layout = self.slice_along_axis(coordinate = coordinate, axis = axis);
    return plot_layout(layout, colors = colors, percentile = percentile, normalize = normalize)
  
  
  def plot_regions(self, cmap = plt.cm.rainbow, annotate = True, axes = [0,1]):
    """Overlays and plots regions to check the alignment of this layout.
    
    Arguments
    ---------
    cmap : colormap
      The color map to use to color the regions.
    annotate : bool
      Use annotaton or not.
    axes : tuple of ints
      Axes to use if sources are larger than 2d.
    """
    position, shape, regions = self.embedding();
    plot_regions(regions, sources = self.sources, cmap = cmap, annotate = annotate, axes = axes);
  
  
  def plot_alignments(self, cmap = plt.cm.rainbow, annotate = True, axes = [0,1]):
    """Overlays and plots regions to check the alignment of this layout.
    
    Arguments
    ---------
    cmap : colormap
      The color map to use to color the regions.
    annotate : bool
      Use annotaton or not.
    axes : tuple of ints
      Axes to use if sources are larger than 2d.
    """
    plot_alignments(self.alignments, sources = self.sources, cmap = cmap, annotate = annotate, axes = axes);
  
  
  def load(self, filename):
    """Loads the layout specifications from a file.
    """
    layout = load_layout(filename, self);
    self.__init__(layout);
  
  
  def save(self, filename):
    """Saves the layout to a file.
    """
    save_layout(filename, self);
  
  ### Internals
  
  def __copy__(self):
    cls = self.__class__
    new = cls.__new__(cls)
    new.__dict__.update(self.__dict__);
    
    #copy sorces and alignments
    sources, alignments = copy_sources_and_alignments(self._sources, self._alignments);
    new._sources = sources;
    new._alignments = alignments;
    
    return new;
  
  
  def sources_as_virtual(self):
    sources = self.sources;
    alignments = self.alignments;
    new_sources = [s.as_virtual() for s in sources];
    new_alignments = [a.copy() for a in alignments];
    source_to_new = {s : n for s,n in zip(sources, new_sources)};
    for a,an in zip(alignments, new_alignments):
      an.pre = source_to_new[a.pre];
      an.post = source_to_new[a.post];
    self._sources = new_sources;
    self._alignments = new_alignments;
  
  def sources_as_real(self):
    sources = self.sources;
    alignments = self.alignments;
    new_sources = [s.as_real() for s in sources];
    new_alignments = [a.copy() for a in alignments];
    source_to_new = {s : n for s,n in zip(sources, new_sources)};
    for a,an in zip(alignments, new_alignments):
      an.pre = source_to_new[a.pre];
      an.post = source_to_new[a.post];
    self._sources = new_sources;
    self._alignments = new_alignments;
  
  def as_virtual(self):
    new = self.copy();
    new.sources_as_virtual();
    return new;      
  
  def as_real(self):
    new = self.copy();
    new.sources_as_real();
    return new;
  
  
  def __str__(self):
    name = self.name;
    s = _source_string(self);
    layout = "<<%ds, %da>>" % (self.n_sources, self.n_alignments);
    return name + layout + s[len(name):];
  
  
  def __repr__(self):
    return self.__str__(); 



class TiledLayout(Layout):
  """TiledLayout handles stacks aligned on a tiling grid."""
  
  def __init__(self, sources = None, expression = None, tile_axes = None, tile_shape = None, tile_positions = None, positions = None, overlaps = None, alignments = None,  position = None, shape = None, dtype = None, order = None, location = None):
    """TiledLayout constructor.
    
    Arguments
    ---------
    sources : list of file names or Source classes
      List of the sources of the individual tiles / images.
    expression : str or None
      If sources is None, use this expression to a list of files to generate the sources.
    tile_axes : str or None
      If expression is given, use this ordering of the tag names in expression to consturct the tiling grid.
    tile_shape : tuple of ints or None
      Optional shape of the grid.
    tile_positions : list of tuple of ints or None
      Optional list of grid positions of the sources.
    positions : list of tuple of ints or None
      Optional list of positions of the individual sources.
    overlaps : tuple of ints or None
      Optional overlaps of the sources in each grid dimension.
    alignments : list of Alignment classes
      Optional alignment structure of the sources.
    shape : tuple of int or None
      The fixed shape of this Layout, if None the minimal size to fit all sources will be used.
    position : tuple of int or None
      The fixed position of this layout, if None the lower corner to fit all sources will be used.
    dtype: dtype or None
      The data type to use for this layout, if None use the dtype of the first source.
    """
    
    if expression is None and sources is None:
      raise ValueError('Either exprssion or sources must be given!');
    
    if expression is not None:
      sources, alignments, tile_positions = _initialize_tiles_from_expression(expression, tile_axes=tile_axes, tile_shape=tile_shape, tile_positions=tile_positions, overlaps=overlaps, positions=positions, alignments=alignments);
    else:
      sources, alignments, tile_positions = _initialize_tiles_from_sources(sources, tile_shape=tile_shape, tile_positions=tile_positions, positions=positions, overlaps=overlaps, alignments=alignments)
    
    #init the underlying Layout
    super(TiledLayout, self).__init__(sources = sources, alignments = alignments, position = position, 
                                      shape = shape, dtype = dtype, order = order, location = location);
    
    #set tile_positions
    for s,g in zip(self.sources, tile_positions):
      s.tile_position = g;
    
    #udate sources
    if positions is None:
      self.update_sources_from_alignments();
  
  
  @property
  def tile_dim(self):
    """Returns the dimension of the grid.
      
    Returns
    -------
    g_dim : int
      The grid dimension.
    """
    return len(self.tile_positions[0]);
  
  
  @property
  def tile_positions(self):
    """Returns the list of the grid positions of the sources.
      
    Returns
    -------
    tile_positions : list of tuple of ints
      The grid positions of the sources.
    """
    return [s.tile_position for s in self.sources];
  
  
  def source_to_tile_position(self, source):
    """Maps the source to a position on the grid in this layout.
    
    Arguments
    ---------
    source : Source class or int
      The souce or id of the source to map to a grid position.
    
    Returns
    -------
    positions : tuple of ints or None
      The grid position of the source, None if not found.
    """
    if isinstance(source, int):
      return self.sources[source].tile_position;
    elif source in self.sources:
      return source.tile_position;
    else:
      return None;
  
  
  def source_from_tile_position(self, tile_position):
    """Maps the grid position to a source in this layout.
    
    Arguments
    ---------
    tile_position : tuple of ints
      The position of the source in the grid.
    
    Returns
    -------
    source : Source class or None
      The source at the required grid position, None if not found.
    """
    for s in self.sources:
      if s.tile_position == tile_position:
        return s;
    return None;
  
  
  def center_tile_position(self):
    """Returns the most center tile position in this layout.
    
    Returns
    -------
    tile_center : tuple of ints 
      The tile position of the most central tile.
    """
    return _center_tile(self.tile_positions);
  
  
  def center_tile_source(self):
    """Returns the most central source in this tile layout.
    
    Returns
    -------
    center : tuple of ints 
      The most central tile.
    """
    tile_center = self.center_tile_position();
    return self.source_from_tile_position(tile_center);
    
  
  def sort_sources_by_tile_position(self):
    """Sorts the sources of this layout by grid position."""
    gpl = self.tile_positions;
    gp  = np.zeros(len(gpl), dtype = object);
    for i in range(len(gp)):
      gp[i] = tuple(gpl[i]);
    sort_id = np.argsort(gp);      
    self._sources = [self._sources[i] for i in sort_id];
    
  
  def adjust_overlaps(self, overlaps = None):
    """Adjusts the positions of the sources given a new estimate of the overlaps.
    
    Arguments
    ---------
    overlaps : tuple of ints or None
      Overlaps of the sources in each grid dimension.
    """    
    tile_dim = self.tile_dim;
    
    if overlaps is None:
      overlaps = 0;
    if not hasattr(overlaps, '__len__'):
      overlaps = [overlaps];
    overlaps = np.array(overlaps, dtype=int);
    overlaps = np.pad(overlaps,(0,max(0, tile_dim-len(overlaps))), 'wrap');
    
    for a in self.alignments:
      pre = a.pre.tile_position;
      post = a.post.tile_position;
      shift = list(a.shift);
      for d in range(tile_dim):
        delta = pre[d] - post[d];
        if delta == 1:
          shift[d] = pre.shape[d] - overlaps[d];
        elif delta == -1:
          shift[d] = -post.shape[d] + overlaps[d];
      a.shift = tuple(shift);
      
    for s in self.sources:
      s.position = (0,) * self.ndim;
    
    self.update_sources_from_alignments();
  
  

  def alignment_from_tile_positions(self, tile_position1, tile_position2):
    """Maps the grid position to a source in this layout.
    
    Arguments
    ---------
    tile_position1,2 : tuple of ints
      The position of the sources in the grid.
    
    Returns
    -------
    alignment : Alignment class or None
      The alignment at the required tile position, None if not found.
    """
    for a in self.alignments:
      if a.pre.tile_position == tile_position1 and a.post.tile_position == tile_position2 or \
         a.pre.tile_position == tile_position2 and a.post.tile_position == tile_position1:
          return a;
    
    return None;


  
  def align_on_tiling(self, overlaps = (50,50), max_shifts = (-50,50), clip = None, background = None, processes = None, verbose = False):
    """Align pairwise images using overlaps along and grid information.
    
    Arguments
    ---------
    layout : TiledLayout 
      The grid layout of the sources.
    overlaps : int, tuple of ints or list of tuple of ints
      The overlaps along the grid axes.
    max_shifts : tuple or list of tuple of ints
      The maximal shifts along the axes directions.
    clip : number or None
      If not None, clip the soruces at this value when calculating the alignment.
    background : number or None
      If not None, if the values in the overlap region are less than this number make alignment return -inf quality as there is no signal to use for alignment.
    processes : int or 'serial' 
      Number of processor to use for parallel processing, if 'serial' process in serial.
    verbose : bool 
      If True, print progress information.
    
    Returns
    -------
    layout : Layout class
      The updated layout.
    """
    return align_layout_on_tiling(self, overlaps = overlaps, max_shifts = max_shifts, clip = clip, background = background, processes = processes, verbose = verbose);
  
  



def _center_tile(tile_positions):
  """helper to calucalte the most central tile in a list of tile positions."""
  tdim = len(tile_positions[0]);
  tpos = np.array(tile_positions, dtype=int);
  for d in range(tdim):
    m = np.sort(tpos[:,d])[(len(tpos)-1)//2];
    tpos = np.array([t for t in tpos if t[d] == m], dtype=int);
  center = tpos[0];
  return tuple(center);


def _initialize_tiles_from_sources(sources, tile_shape = None, tile_positions = None,  positions = None, overlaps = None, alignments = None):
  """Helper to set up the TiledLayout info."""
  
  #tiling
  if tile_positions is None: # infer tiling from sources or tile_shape
    if tile_shape is None: # infer tiling from structure of sources
      if isinstance(sources, list): # nested list structure
        #grid shape
        src = sources;
        tile_shape = ();
        while isinstance(src, list):
          tile_shape += (len(src),);
          src = src[0];
      
        #convert to flat list
        src = sources;
        while isinstance(src[0], list):
          sl = [];
          for s in src:
            sl.extend(s);
          src = sl;
        sources = src;
      
      elif isinstance(sources, np.ndarray): # numpy array
        tile_shape = sources.shape;
        sources = list(sources.flat);
    #create grid positions
    tile_positions = list(itt.product(*[range(s) for s in tile_shape]));
  else:
    tile_shape = np.max(tile_positions, axis = 0) + 1;
  
  #sources
  sources = list(sources);
  if len(tile_positions) != len(sources):
    ValueError('Number of sources = %d does not match grid positions = %d !' % (len(sources), len(tile_positions)));
  
  #remove None type sources and ensure source classes
  src = []; pos = [];      
  for s,p in zip(sources, tile_positions):
    if s is not None:
      if isinstance(s, Source):
        src.append(s)
      else:
        src.append(Source(source = s));
      pos.append(p);
  sources = src;
  tile_positions = pos;
  
  #tile data
  tile_dim = len(tile_positions[0]);   
  source_dim = sources[0].ndim;
  tile_position_to_source = { p : s for p,s in zip(tile_positions, sources)};
  
  # alignments
  if alignments is None:
    alignments = [];
    
    if positions is None:     
      if overlaps is None:
        overlaps = 0;
      if not isinstance(overlaps, (list, tuple)):
        overlaps = [overlaps];
      overlaps = np.array(overlaps, dtype=int);
      overlaps = np.pad(overlaps,(0,max(0, tile_dim-len(overlaps))), 'wrap');

      for d in range(tile_dim):      
        for pre,pre_tpos in zip(sources, tile_positions):     
          if pre_tpos[d] < tile_shape[d] - 1:
            shift = tuple(0 if i!=d else pre.shape[d] - overlaps[d] for i in range(source_dim));
            post_tpos = tuple(p if i!=d else p+1 for i,p in enumerate(pre_tpos));
            post = tile_position_to_source.get(post_tpos, None);
            if post is not None:
              alignments.append(Alignment(pre = pre, post = post, shift = shift))
              
      for s in sources: # ensure source positions are properly updated
         s.position = (0,) * s.ndim;    
  
    else: #positions are given
      for d in range(tile_dim):      
        for pre,pre_tpos in zip(sources, tile_positions):     
          if pre_tpos[d] < tile_shape[d] - 1:
            post_tpos = tuple(p if i!=d else p+1 for i,p in enumerate(pre_tpos));
            post = tile_position_to_source.get(post_tpos, None);
            if post is not None:
              alignments.append(Alignment(pre = pre, post = post, shift = (0,) * pre.ndim));
              
      for s,p in zip(sources, positions): 
         s.position = p;
         
    return sources, alignments, tile_positions
  

def _initialize_tiles_from_expression(expression, tile_axes = None, tile_shape = None, tile_positions = None, overlaps = None, positions = None, alignments = None):
  """Helper to nitialize TiledLayout from an expression of file names."""
  if not isinstance(expression, te.Expression):
    expression = te.Expression(expression);
  
  tag_names = expression.tag_names();
  if tile_axes is None:
    tile_axes = tag_names;
  for n in tile_axes:
    if not n in tag_names:
      raise ValueError('The expression does not have the named pattern %s' % n);
  for n in tag_names:
    if not n in tile_axes:
      raise ValueError('The expression has the named pattern %s that is not in tile_axes=%r' % (n, tile_axes));
  #print tile_axes, tag_names
  
  
  #construct tiling
  files = fl._file_list(expression);
  tile_values = [expression.values(f) for f in files];
  tile_values = [tuple(tv[n] for n in tile_axes) for tv in tile_values];
    
  if tile_positions is not None:
    tile_positions = [t for t in tile_positions if t in tile_values];
    sources = [];
    for p in tile_positions:                    
      for s,t in zip(files, tile_values):
        if t == p:
          sources.append(s);
          break;
  else:                      
    tile_positions = tile_values;
    sources = files;
  
  tile_shape = np.max(tile_positions, axis = 0) + 1; # assume that numbering starts with 0!
    
  return _initialize_tiles_from_sources(sources, tile_positions=tile_positions, tile_shape=tile_shape, positions=positions, overlaps=overlaps, alignments=alignments)


########################################################################################
### Basic functions
########################################################################################

def check_alignments_and_sources(alignments, sources, verbose = False):
  """Checks consistency of alignments and sources."""
  for i,a in enumerate(alignments):
    if a.pre not in sources:
      if verbose:
        print("Alignment %d, pre source %d not in sources %r!" % (i, a.pre.id, [s.id for s in sources]));                                         
        #print a.pre, sources 
      return False; 
        
    if a.post not in sources:
      if verbose:
        print("Alignment %d, post source %d not in sources %r!" % (i, a.post.id, [s.id for s in sources]));
        #print a.post, sources  
      return False;
  return True;


def copy_sources_and_alignments(sources, alignments):
  """Copys the sources and alignments but not the underlying array data.
  
  Arguments
  ---------
  sources : list of Source classes or None
    List of sources.
  alignments : list of Alignment classes
    The pairwise alignments.
  
  Returns
  -------
  sources : list of Source classes or None
    Copied list of sources.
  alignments : list of Alignment classes
    Copied list of pairwise alignments.
  """
  new_sources = [s.copy() for s in sources];
  new_alignments = [a.copy() for a in alignments];
  source_to_new = {s : n for s,n in zip(sources, new_sources)};
  for a,an in zip(alignments, new_alignments):
    an.pre = source_to_new[a.pre];
    an.post = source_to_new[a.post];
  return new_sources, new_alignments;


def sources_from_alignments(alignments):
  """Returns a unique list of sources from the given alignments.
  
  Arguments
  ---------
  alignments : list of Alignment classes
    The pairwise alignments.
  
  Returns
  -------
  sources : list of Source classes
    A unique list of the sources.
  """
  sources = [];
  for a in alignments:
    if a.pre not in sources:
      sources.append(a.pre);
    if a.post not in sources:
      sources.append(a.post);
  return sources;


def source_index(sources, source):
  """The index of a source in the list of sources.
  
  Arguments
  ---------
  sources : list of source classes
    The list of sources to search in.
  source : Source class
    The source to search for.
  
  Returns
  -------
  index : int or None
    Position of the source in the sources list, None if not found.
  """  
  for i,s in enumerate(sources):
    if s == source:
      return i;  
  return None;


def connected_components(alignments, sources = None, min_quality = None, with_sources = False):
  """Returns the connected components of the alignments
  
  Arguments
  ---------
  alignments : list of Alignment classes
    The pairwise alignments.
  sources : list of Source classes or None
    Optional list of all sources.
  min_quality : float, tuple of floats or None
    The mininal quality for alignments to be included in the calculation.
  with_sources : bool
    If True, also return the sources in each component, as single sources might not appear in the alignment components.
  
  Returns
  -------
  components : list of list of Alignment classes
    The connected components of the alignments.
  """
  alignments = filter_alignments(alignments, min_quality = min_quality);
  
  if sources is None:
    sources = sources_from_alignments(alignments);
  n_sources = len(sources);
  source_to_index = { s : i for i,s in enumerate(sources)};
  #print sources, alignments
    
  #determine connected compoenents
  g = gt.Graph(directed = False);    
  g.add_vertex(n_sources);
  for a in alignments:
    g.add_edge(source_to_index[a.pre], source_to_index[a.post]);
  connected_components, hist = gtt.label_components(g);
  connected_components = np.array(connected_components.a);                            
  n_components = len(hist);
  #print connected_components, hist, len(hist), np.max(hist)   
  
  # create components
  components = [];
  for i in range(n_components):
    ids = np.where(connected_components == i)[0];
    comp = [];
    for a in alignments:
      if source_to_index[a.pre] in ids:
        comp.append(a);
    components.append(comp);
                     
  if with_sources:
    component_sources = [[sources[i] for i in np.where(connected_components == c)[0]] for c in range(n_components)];
    return components, component_sources                              
  else:
    return components;


def connected(alignments, sources = None):
  """Returns True if the alignments form a single connected component.
  
  Arguments
  ---------
  alignments : list of Alignment classes
    The pairwise alignments.
  sources : list of Source classes or None
    Optional list of all sources.
  
  Returns
  -------
  connected : bool
    True if the alignments form a single connected component.
  """
  return len(connected_components(alignments = alignments, sources = sources)) == 1;


def save_layout(filename, layout):
  """Saves a layout class to a file.
  
  Arguments
  ---------
  filename : str
    The file to save the layout too.
  layout : Layout class
    The layout to save.
  
  Returns
  -------
  file_name : str
    The file name in which the layout was saved.
  """  
  s = np.array([layout.as_virtual()], dtype=object);
  #prevent np to add .npy to a .layout file
  fid = open(filename, "wb");
  np.save(fid, s); 
  fid.close();
  
  return filename;


def load_layout(filename):
  """Loads a layout class from a file
  
  Arguments
  ---------
  filename : str
    The file to load the layout from.
  
  Returns
  -------
  layout : Layout class
    The loaded layout.
  """  
  s = np.load(filename);
  layout = s[0];
  return layout;


def slice_layout_along_axis(layout, coordinate, axis = 2):
  """Slice a layout at a coordinate along an axis.
  
  Arguments
  ---------
  layout : Layout class
    The layout to take the slice through.
  coordinate : int 
    The coordinate of the slice along the slice axis in the original layout.
  axis : int
    The axis used to slice the layout.
    
  Note
  ----
  The sources of the layout will be sliced accordingly.
  The sources position along the axis is taken into account and sources not in 
  the slice are droped. Thus, ensure the position along the slice axis is 
  aligned, e.g. by using :func:`aling_layout_along_axis`.
  """ 
  sources = [s for s in layout.sources if 0 <= coordinate - s.position[axis] < s.shape[axis]];
  
  new_layout = layout.copy();
  new_layout._sources = [s.slice_along_axis(coordinate = coordinate, axis = axis) for s in sources]
  
  source_to_sliced = {s : sl for s,sl in zip(sources, new_layout._sources)};

  alignments = [a.copy() for a in layout.alignments if a.pre in sources and a.post in sources];
  for a in  alignments:
    a.pre = source_to_sliced[a.pre]; 
    a.post = source_to_sliced[a.post];
    displacement = a.displacement;
    a.displacement = displacement[:axis] + displacement[axis+1:];
  new_layout._alignments = alignments;
  
  if layout._shape is None:
    new_layout._shape = None;
  else:
    shape = layout._shape;
    new_layout._shape = shape[:axis] + shape[axis+1:];
  
  if layout._position is None:
    new_layout._position = None;
  else:
    position = layout.position;
    new_layout._position = position[:axis] + position[axis+1:];
  
  #new_layout.coordinate = coordinate;
  #new_layout.axis = axis;
  #new_layout.slicing = tuple(slice(None) if d != axis else coordinate for d in range(layout.ndim))
  
  return new_layout;





########################################################################################
### Alignment
########################################################################################

def align_2_sources(src1, src2, max_shifts = 10, clip = None, background = None, normalize = False, verbose = False, debug = False):
  """Align 2 sources using root mean square difference measure.
  
  Arguments
  ---------
  src1, src2 : array like sources
    Sources to align.
  max_shifts : int, tuple or list of tuples of ints
    The minimum and maximum shifts along the different axes to consider for alignment.
  clip : number or None
    If not None, clip the soruces at this value when calculating the alignment.
  background : number or None
    If not None, if the values in the overlap region are less than this number make alignment return -inf quality as there is no signal to use for alignment.
  normalize : bool
    Use normalized cross correlation, instead of co-variance, i.e. subtract mean and divide by std.
  verbose : bool 
    If True print progress information.
        
  Returns
  -------
  shift : array 
    The additional shift between the first and second source for optimal pairwise alignment.
  quality : float
    Quality measure.
  """
  #if not isinstance(src1, SourceRegion):
  #  src1 = Source(source=src1);
  #if not isinstance(src2, SourceRegion):
  #  src2 = Source(source=src2);
  
  if src1.ndim != src2.ndim:
    raise ValueError('Sources expected to have same dimension, found %d and %d dimensional images!' % (src1.ndim, src2.ndim));
  ndim = src1.ndim;  
    
  # format max shifts
  max_shifts = np.array(_format_max_shifts(max_shifts, ndim), dtype=int);
  shift_min = max_shifts[:,0];
  shift_max = max_shifts[:,1];
  
  if debug:
    print('Alignment: positions = %r,%r, shapes= %r,%r shifts=%r,%r' % (src1.position, src2.position, src1.shape, src2.shape, shift_min, shift_max));
    
  slice1, slice2, pad1, pad2, slice_no_pad1, slice_no_pad2, shift_min, shift_max, fft_roi = _slicing_and_padding_for_alignment(src1, src2, shift_min, shift_max);                                                                                                                        
                                                                                                                              
  # extract relevant data    
  i1 = src1[slice1];
  i2 = src2[slice2];
  
  if debug:
    print('Alignment: data shapes = %r, %r' % (i1.shape, i2.shape));
    print('Alignment: overlaping slices = %r, %r' % (slice1, slice2));                                         
                                                      
  # clip images for better alignment performance  
  if clip is not None:
    #if sources are memmaps copy to array
    i1 = np.array(i1);
    i2 = np.array(i2);
                 
    #clip
    if isinstance(clip, (tuple, list)):
      if clip[0] is not None:
        i1[i1 < clip[0]] = clip[0];
        i2[i2 < clip[0]] = clip[0];
      if clip[1] is not None:
        i1[i1 > clip[1]] = clip[1];
        i2[i2 > clip[1]] = clip[1];
    else:
      i1[i1 > clip] = clip;
      i2[i2 > clip] = clip;
  
  #check if one of the images is background
  if background is not None:
    if isinstance(background, tuple):
      if isinstance(background[1], int):
        aa1 = aa2 = background[1];
      else:
        aa1 = np.prod(i1.shape) * background[1];
        aa2 = np.prod(i2.shape) * background[1];         
      bb = np.sum(i1 >= background[0]);              
      bad = bb < aa1;
      if verbose and bad:
        print('Alignment: Not enough pixels %d<%d above background %d in source %r' % (bb, aa1, background[0], src1.tile_position));
      if not bad:
        bb = np.sum(i2 >= background[0]);            
        bad = bb < aa2;
        if verbose and bad:
          print('Alignment: Not enough pixels %d<%d above background %d in source %r' % (bb, aa2, background[0], src2.tile_position));    
    else: #background is int                                                       
      bad = np.all(i1 < background) or np.all(i2 < background)        
      if verbose and bad:
        print('Alignment: No good signal between %r and %r' % (src1.tile_position, src2.tile_position));
    if bad:                                                              
      shift = np.zeros(i1.ndim, dtype=int);      
      quality = -np.inf;
      return shift, quality;
  
  # ensure doulbe images for fft
  i1 = np.asarray(i1, dtype = float);
  i2 = np.asarray(i2, dtype = float);
  
  #normalize the image
  if normalize:
    i1 -= np.mean(i1);
    i2 -= np.mean(i2);
    i1 *= 1.0/np.sqrt(np.sum(i1*i1));
    i2 *= 1.0/np.sqrt(np.sum(i2*i2));
  
  #pad to same size + zeros for overlap in fft
  i1 = np.pad(i1, pad1, 'constant');
  i2 = np.pad(i2, pad2, 'constant');
  
  #weights
  w1 = np.zeros(i1.shape);
  w1[slice_no_pad1] = 1;               
  w1fft = np.fft.fftn(w1);
  
  if np.any([s1.start != s2.start or s1.stop != s2.stop for s1,s2 in zip(slice_no_pad1, slice_no_pad2)]):
    w2 = np.zeros(i2.shape);
    w2[slice_no_pad2] = 1;
    w2fft = np.fft.fftn(w2);
  else:
    w2 = w1;
    w2fft = w1fft;
  
  #debug
  if debug:
    if debug is True:
      debug = 1;
    plt.figure(95+debug-1); plt.clf();
    plt.subplot(2,2,1); plt.imshow(i1.T); plt.title('i1')
    plt.subplot(2,2,2); plt.imshow(w1.T); plt.title('w1')
    plt.subplot(2,2,3); plt.imshow(i2.T); plt.title('i2')
    plt.subplot(2,2,4); plt.imshow(w2.T); plt.title('w2')
    
    plt.figure(99+debug-1); plt.clf();
    plt.subplot(1,2,1);
    plt.hist(i1.flatten(), bins = 256);
    plt.subplot(1,2,2);
    plt.hist(i2.flatten(), bins = 256);
 
    print('Alignment: shapes = %r, %r' % (i1.shape, i2.shape));
  
  # fft
  i1fft = np.fft.fftn(i1); 
  i2fft = np.fft.fftn(i2);
  s1fft = np.fft.fftn(i1 * i1);  
  s2fft = np.fft.fftn(i2 * i2);
  wssd = w1fft * np.conj(s2fft) + s1fft * np.conj(w2fft) - 2 * i1fft * np.conj(i2fft);

  #if verbose:
  #  print 'FFT done!';

  wssd = np.fft.ifftn(wssd);
  nrm  = np.fft.ifftn(w1fft * np.conj(w2fft));
  
  #if verbose:
  #  print 'iFFT done!';  
  
  # debug
  if debug:
    plt.figure(96+debug-1); plt.clf();
    plt.subplot(2,2,1); plt.imshow(np.abs(wssd.T)); plt.title('wssd')
    plt.subplot(2,2,2); plt.imshow(np.abs(nrm.T)); plt.title('nrm')
    plt.subplot(2,2,3); plt.imshow(np.abs(wssd.T) / np.abs(nrm.T));
  
  # range of interest
  wssd = wssd[fft_roi];
  nrm  = nrm[fft_roi];
  
  # normalize
  eps =  2.2204e-16;
  nrm[nrm <= eps] = eps;
  cc = np.abs(wssd / nrm);
  
  # debug
  if debug:
    plt.figure(97+debug-1); plt.clf();
    plt.subplot(2,2,1); plt.imshow(np.abs(wssd.T)); plt.title('wssd');
    plt.subplot(2,2,2); plt.imshow(np.abs(nrm.T)); plt.title('nrm');
    plt.subplot(2,2,3); plt.imshow(np.abs(wssd.T) / np.abs(nrm.T));
  
  # find optimal shift
  shift = np.argmin(cc);
  shift = np.unravel_index(shift, cc.shape);
  quality = -(cc[tuple(shift)]);
  #print shift, quality
  
  #debug
  if debug:
    plt.plot([shift[0]], [shift[1]], '*', c = 'r')
  
  # correct for cutting
  shift = tuple(s + m for s,m in zip(shift, shift_min));
  
  if verbose:
    print('Alignment: done! shift = %r, quality = %.2e' % (shift, quality));
  
  return shift, quality;



def overlap(region1, region2):
  """Overlap between two regions."""
  ovl = np.max([region1.lower, region2.lower], axis = 0);
  ovu = np.min([region1.upper, region2.upper], axis = 0);
  
  if np.any(ovu - ovl - 1 < 0):
    return None;
  else:
    return Overlap(lower = ovl, upper = ovu, sources = [region1, region2]);


def _overlap_with_shifts(src1, src2, max_shifts):
  """Calculates the maximal overlap between two sources given maximal shifts."""
  # format max shifts
  max_shifts = np.array(_format_max_shifts(max_shifts, src1.ndim), dtype=int);
  sh_min = max_shifts[:,0];
  sh_max = max_shifts[:,1];
  
  # calculate overlap regions from positions, shapes and max_shifts
  p1 = np.array(src1.position);
  p2 = np.array(src2.position);
  s1 = src1.shape; 
  s2 = src2.shape;
  #print p1, p2, s1, s2, sh_min, sh_max
    
  s1a = Region(position = tuple(p1), shape = s1);
  s2a = Region(position = tuple(p2 + sh_min), shape = tuple(s2 + sh_max - sh_min));  
  _, _, regions = embedding([s1a, s2a]);
  overlap_a = None
  for r in regions:
    if len(r.sources) == 2:
      overlap_a = r;
      break;
  if overlap_a is None:
    raise ValueError('The two sources will never overlap, increase max_shifts or change source positions!');
  
  s1b = Region(position = tuple(p1 - sh_max), shape = tuple(s1 + sh_max - sh_min));
  s2b = Region(position = tuple(p2), shape = s2);
  _, _, regions = embedding([s1b, s2b]);
  overlap_b = None
  for r in regions:
    if len(r.sources) == 2:
      overlap_b = r;
      break;
  
  return overlap_a, overlap_b


def _slicing_and_padding_for_alignment(src1, src2, shift_min, shift_max):
  # calculate overlap regions and paddings for alignment from positions, shapes and max_shifts
  ndim = src1.ndim;
  p1 = np.array(src1.position);
  p2 = np.array(src2.position);
  s1 = src1.shape; 
  s2 = src2.shape;
  #print p1, p2, s1, s2, sh_min, sh_max
    
  s1a = Region(position = tuple(p1), shape = s1);
  s2a = Region(position = tuple(p2 + shift_min), shape = tuple(s2 + shift_max - shift_min));  
  _, _, regions = embedding([s1a, s2a]);
  overlap_a = None
  for r in regions:
    if len(r.sources) == 2:
      overlap_a = r;
      break;
  if overlap_a is None:
    raise ValueError('The two sources will never overlap, increase max_shifts or change source positions!');
  
  s1b = Region(position = tuple(p1 - shift_max), shape = tuple(s1 + shift_max - shift_min));
  s2b = Region(position = tuple(p2), shape = s2);
  _, _, regions = embedding([s1b, s2b]);
  overlap_b = None
  for r in regions:
    if len(r.sources) == 2:
      overlap_b = r;
      break;
  
  for i,s in enumerate(overlap_a.sources):
    if s == s1a:
      sid = i;
      break;
  slice1 = overlap_a.source_slicings()[sid];
  
  for i,s in enumerate(overlap_b.sources):
    if s == s2b:
      sid = i;
      break;
  slice2 = overlap_b.source_slicings()[sid];  
  
  #correct max shifts to maximal useful if overlap is smaller
  p1l = p1 + [s.start for s in slice1];
  p1u = p1 + [s.stop for s in slice1];
  p2l = p2 + [s.start for s in slice2];
  p2u = p2 + [s.stop for s in slice2];            
  for d in range(ndim):
    shift_min[d] = max(shift_min[d], p1l[d] - p2u[d]);
    shift_max[d] = min(shift_max[d], p1u[d] - p2l[d]);
    
  #padding to make equal size and account for shifts / correlations due to 'wrap around' of fft
  pad1 = []; pad2 = [];
  for d in range(ndim):
    l_min = min(p1l[d], p2l[d] + shift_min[d]);
    l_max = min(p1l[d], p2l[d] + shift_max[d]);
    u_min = max(p1u[d], p2u[d] + shift_min[d]);
    u_max = max(p1u[d], p2u[d] + shift_max[d]);
    s_min = u_min - l_min;
    s_max = u_max - l_max;
    if s_min >= s_max:
      pad1.append((p1l[d] - l_min, u_min - p1u[d]));
      pad2.append((p2l[d] + shift_min[d] - l_min, u_min - (p2u[d]+shift_min[d])))
    else:
      pad1.append((p1l[d] - l_max, u_max - p1u[d]));
      pad2.append((p2l[d] + shift_max[d] - l_max, u_max - (p2u[d]+shift_max[d])))
  
  #non padded slices
  s12 = tuple(u - l + p[0] + p[1] for u,l,p in zip(p1u, p1l, pad1));
  slice_no_pad1 = tuple(slice(p[0], s-p[1]) for p,s in zip(pad1, s12));
  slice_no_pad2 = tuple(slice(p[0], s-p[1]) for p,s in zip(pad2, s12));
  
  # range of interest
  fft_roi = ();
  for d in range(ndim):
    if pad2[d][0] == 0:  # right pad
      fft_roi += (slice(None, (shift_max[d] - shift_min[d])),)
    else: # left pad
      fft_roi += (slice(-(shift_max[d] - shift_min[d]), None),); 
                     
  return slice1, slice2, pad1, pad2, slice_no_pad1, slice_no_pad2, shift_min, shift_max, fft_roi



def align_2_sources_along_axis(src1, src2, axis = 0, overlap = 10, max_shifts = 10, clip = None, background = None, verbose = False):
  """Align 2 images along a specified axis.
  
  Arguments
  ---------
  src1, src2 : 2d arrays
    The images to align.
  axis : int
    The alignment axis.
  overlap : tuple of int
    The minimum and maximum overlap along the alignment axis
  max_shifts : int, tuple of int or list of tuple of ints
    The minimum and maximum shifts along the different axes. Only the values for the axes orthogonal to the alignment axis are used.
  clip : number or None
    If not None, clip the sources at this value when calculating the alignment.
  background : number or None
    If not None, if the values in the overlap region are less than this number make alignment return -inf quality as there is no signal to use for alignment.
  verbose : bool 
    If True, print progress information.
        
  Returns
  -------
  shift : array 
    The shift of the second image wrt to the first for optimal pairwise alignment.
  quality : float
    A quality measure of the alignment.
    
  Note
  ----
  This routine simply translates overlap specifications in one axis direction into max_shifts for use with align_2_sources.
  """
  if not isinstance(src1, SourceRegion):
    src1 = Source(src1);
  if not isinstance(src2, SourceRegion):
    src2 = Source(src2);
  
  if src1.ndim != src2.ndim:
    raise ValueError('Images expected to have the same dimensions, found %d and %d dimensional images!' % (src1.ndim, src2.ndim));
  ndim = src1.ndim;
  s1 = src1.shape;
  s2 = src2.shape;
  
  max_shifts = _format_max_shifts(max_shifts, ndim)
  
  # overlaps
  if isinstance(overlap,int):
    overlap=(1,overlap);
  p1 = src1.position; p2 = src2.position;
  shifts = [];
  for d in range(ndim):
    if d == axis:
      max_ovl = min(s1[axis], s2[axis]);
      max_ovl = min(max_ovl, max(overlap));
      min_ovl = min(max_ovl, min(overlap));
      min_ovl = max(1, min_ovl);
      shifts.append((s1[axis]-max_ovl + p1[axis]-p2[axis], s1[axis]-min_ovl + p1[axis]-p2[axis]));
    else:
      shifts.append(max_shifts[d]);
  
  return align_2_sources(src1, src2, max_shifts = shifts, clip = clip, background = background, verbose = verbose)


def max_intensity_projection(data, axis = 0, function = np.max):
  """Returns the max intensity projection along a specified axis.
  
  Arguments
  ---------
  data : array
    The data array.
  axis : int
    The axis along which to perform the maximum projection.
  
  Returns
  -------
  mip : array
    The maximum intensity projection of the data along axis.
  """
  return function(data, axis = axis);


def align_2_sources_along_axis_mip(src1, src2, axis = 2, depth = 10, max_shifts = 10, clip = None, background = None, verbose = False, with_mip = False):
  """Align 2 images orthogonal to a spcified axis using max projection
  
  Arguments
  ---------
  src1, src2: 2d arrays
    The sources to align.
  axis: int
    Axis for max intensity projection (mip).
  depth : int
    The depth to use for the maximum intensity projection along the mip axis.
  max_shifts: tuple of int
    The minimum and maximum shifts along the axes.
  clip : number or None
    If not None, clip the soruces at this value when calculating the alignment.
  background : number or None
    If not None, if the values in the overlap region are less than this number make alignment return -inf quality as there is no signal to use for alignment.
  verbose : bool 
    If True, print progress information.
  with_mip: bool
    If True, also return the maximum projections used to aling the two sources.
        
  Returns
  -------
  shift : array 
    The shift of the second image wrt to the first for optimal pairwise alignment orthogonal to mip axis
  quality : float
    The quality measure of the alignment.
  mips : tuple of arrays
    Optional maximum intensity projections.
  """  
  if not isinstance(src1, SourceRegion):
    src1 = Source(src1);
  if not isinstance(src2, SourceRegion):
    src2 = Source(src2);               
  
  if src1.ndim != src2.ndim:
    raise ValueError('Images expected to have the same dimensions, found %d and %d dimensional images!' % (src1.ndim, src2.ndim));
  ndim = src1.ndim;
  
  max_shifts = _format_max_shifts(max_shifts, ndim)
  max_shifts = max_shifts[:axis] + max_shifts[axis+1:];
  
  s1 = src1.shape;
  s2 = src2.shape;
  
  # max intensity projections 
  sub1 = [slice(None)] * ndim;
  sub1[axis] = slice(max(0, s1[axis] - depth), None);
  sub1 = tuple(sub1);
  
  sub2 = [slice(None)] * ndim;
  sub2[axis] = slice(None, min(depth, s2[axis]));
  sub2 = tuple(sub2);          
              
  # calculate max projection along axis  
  mip1 = max_intensity_projection(src1[sub1], axis = axis);
  mip2 = max_intensity_projection(src2[sub2], axis = axis);                               
  
  #add position information
  p1 = src1.position[:axis] + src1.position[axis+1:];
  p2 = src2.position[:axis] + src2.position[axis+1:];
  #print axis, p1, p2
  
  mip1 = Source(mip1, position = p1, tile_position = src1.tile_position);
  mip2 = Source(mip2, position = p2, tile_position = src2.tile_position);
  #print mip1, mip2              
  
  mip_shift, quality = align_2_sources(mip1, mip2, max_shifts = max_shifts, clip = clip, background = background, verbose = verbose);
  
  shift = np.zeros(ndim, dtype=int); 
  k = 0;
  for d in range(ndim):
    if d != axis:
      shift[d] = mip_shift[k];
      k += 1;
  shift = tuple(shift);
  
  res = (shift, quality);
  if with_mip:
    res += ((mip1, mip2),);
  return res;


def align_layout(layout, max_shifts = 10, clip = None, background = None, processes = None, verbose = False):
  """Aligns the sources in a layout.
  
  Arguments
  ---------
  layout : Layout class
    The layout of the sources.
  max_shifts : int, tuple of ints or list of list of tuple of ints
    The maximal shifts of the images with respect to each other along each dimension.
  clip : number or None
    If not None, clip the soruces at this value when calculating the alignment.
  background : number or None
    If not None, if the values in the overlap region are less than this number make alignment return -inf quality as there is no signal to use for alignment.
  processes : int or 'serial'
    Number of processor to use for parallel processing, if 'serial' process in serial.
  verbose : bool 
    If True, print progress information.
    
  Returns
  -------
  layout : Layout class
    The layout with the new alignments.
  """
  if not isinstance(layout, Layout):
    raise ValueError("A Layout class is expected as input!");
  #source_to_index = {s : i for i,s in enumerate(layout.sources)};
  
  n_alignments = layout.n_alignments;
  
  if verbose:
    timer = tmr.Timer();
    print('Alignment: Aligning %r' % layout);
    
  if not isinstance(processes, int) and processes != 'serial':
    processes = mp.cpu_count();
  
  _align = ft.partial(_align_layout, n_alignments=n_alignments, max_shifts=max_shifts, 
                      clip=clip, background=background, verbose=verbose)
  
  if processes == 'serial':
    results = [_align(a.pre, a.post, i) for i,a in enumerate(layout.alignments)];
  else:
    layout.sources_as_virtual();
    alignments = layout.alignments;
    with concurrent.futures.ProcesssPoolExecutor(processes) as executor:
      results = executor.map(_align, [a.pre for a in alignments], [a.post for a in alignments], range(n_alignments));
    results = list(results);
    #layout.sources_as_real()
  
  for a,r in zip(layout.alignments, results):
    shift, quality = r;
    a.shift = shift;
    a.quality = quality;
    
  if verbose:
    timer.print_elapsed_time('Alignment: Aligning %r' % layout);
  
  return layout;


@ptb.parallel_traceback
def _align_layout(pre, post, aid, n_alignments, max_shifts, clip, background, verbose):
  if verbose:
    #id1 = source_to_index[pre]; id2 = source_to_index[post];
    print('Alignment: aligning source %d with %d, alignment pair %d/%d' % (pre.id, post.id, aid, n_alignments));  
  
  return align_2_sources(pre, post, max_shifts=max_shifts, 
                         clip=clip, background=background, verbose=verbose);



def align_layout_on_tiling(layout, overlaps = 10, max_shifts = 10, clip = None, background = None, processes = None, verbose = False):
  """Calculates shifts of the sources on a gird layout that aligns them.
  
  Arguments
  ---------
  layout : TiledLayout 
    The grid layout of the sources.
  overlaps : tuple of ints
    The overlaps along the grid axes.
  max_shifts : tuple or list of tuple of ints
    The maximal shifts along the axes directions.
  clip : number or None
    If not None, clip the soruces at this value when calculating the alignment.
  background : number or None
    If not None, if the values in the overlap region are less than this number make alignment return -inf quality as there is no signal to use for alignment.
  processes : int or 'serial' 
    Number of processor to use for parallel processing, if 'serial' process in serial.
  verbose : bool 
    If True, print progress information.
  
  Returns
  -------
  layout : TiledLayout class
    The updated layout after alignment.
  """
  if not isinstance(layout, Layout):
    raise ValueError("A TiledLayout class is expected as input!");
  
  if verbose:
    timer = tmr.Timer();
    print('Alignment: Aligning %r on gird!' % layout);
 

  alignments = layout.alignments;
  n_alignments = len(alignments);
  
  #source_to_index = {s : i for i,s in enumerate(layout.sources)};
  
  axes = [];
  for i,a in enumerate(layout.alignments):
    #alignment axis
    pos_pre = np.array(a.pre.tile_position);
    pos_post= np.array(a.post.tile_position);
    axes.append(np.where(pos_pre - pos_post != 0)[0][0]);
  
  
  _align = ft.partial(_align_layout_on_tiling, n_alignments=n_alignments, 
                      overlaps=overlaps, max_shifts=max_shifts, 
                      clip=clip, background=background, verbose=verbose)
  
  if not isinstance(processes, int) and processes != 'serial':
    processes = mp.cpu_count();
  
  if processes == 'serial':
    results = [_align(a.pre, a.post, ax, i) for a,ax,i in zip(alignments, axes, range(n_alignments))];
  else:
    layout.sources_as_virtual();
    alignments = layout.alignments;
    with concurrent.futures.ProcessPoolExecutor(processes) as executor:
      results = executor.map(_align, [a.pre for a in alignments], [a.post for a in alignments], axes, range(n_alignments));
    results = list(results) 
    #layout.sources_as_real()
  
  for a,r in zip(layout.alignments, results):
    shift, quality = r;    
    a.shift = shift;
    a.quality = quality;
    
  if verbose:
    timer.print_elapsed_time('Alignment: Aligning %r on grid' % layout);
  
  return layout;

@ptb.parallel_traceback
def _align_layout_on_tiling(pre, post, axis, aid, n_alignments, overlaps, max_shifts, clip, background, verbose):
  if verbose:
    #id1 = source_to_index[a.pre]; id2 = source_to_index[a.post];
    print('Alignment: aligning source %d with %d along axis %d, alignment pair %d/%d' % (pre.id, post.id, axis, aid, n_alignments));  
  return align_2_sources_along_axis(pre, post, axis=axis, overlap=overlaps[axis], max_shifts=max_shifts, 
                                    clip=clip, background=background, verbose=verbose);









def align_layout_axis(layout, axis = 2, depth = 10, max_shifts = 10, axis_range = None, clip = None, background = None, processes = None, verbose = False):
  """Aligns sources in a layout in a single axis direction only.
  
  Arguments
  ---------
  layout: Layout class
    The layout in which to align the 3d sources in z-direction.
  axis : int
    The axis along to aling the layout.  
  depth : int or list of ints
    The approximate overlaps of the images in the different dimensions to use for mip projection.
    Only the depth parameter along the relevant axis is used.
  max_shifts : tuple of ints
    The minmal and maximal shift in to consider.
  axis_range : tuple of int or None
    Use only a sub set of the axis range to speed up processing.
  clip : number or None
    If not None, clip the soruces at this value when calculating the alignment.
  background : number or None
    If not None, if the values in the overlap region are less than this number make alignment return -inf quality as there is no signal to use for alignment. 
  processes : int or 'serial' 
    Number of processor to use for parallel processing, if 'serial' process in serial.
  verbose : bool
    Print progress information.
  
  Returns
  -------
  layout : Layout 
    The layout with updated axis-alignments.
    
  Note
  ----
  To speed up the calculation, a mip projection is used in the direction of the
  tiling
  """
  if not isinstance(layout, Layout):
    raise ValueError('The layout is expected to be a GridLayout');
    
  if verbose:
    timer = tmr.Timer();
    print('Alignment: aligning sources in layout along axis=%d!' % axis)
  
  #format the shifts
  if not isinstance(depth, (list, tuple)):
    depth = (depth,) * layout.ndim;
  
  max_shifts = _format_max_shifts(max_shifts, layout.ndim);
  n_alignments = layout.n_alignments;
  
  
  _align = ft.partial(_align_layout_axis, 
                      n_alignments=n_alignments, axis=axis, axis_range=axis_range, 
                      depth=depth, max_shifts=max_shifts, 
                      clip=clip, background=background, verbose=verbose);
  
  if not isinstance(processes, int) and processes != 'serial':
    processes = mp.cpu_count();
  
  if processes == 'serial':
    results = [_align(a.pre, a.post, i) for i,a in enumerate(layout.alignments)];
  else:
    layout.sources_as_virtual();                        
    alignments = layout.alignments;
    print('align_axis')
    with concurrent.futures.ProcessPoolExecutor(processes) as executor:
      results = executor.map(_align, [a.pre for a in alignments], [a.post for a in alignments], range(n_alignments));
    results = list(results);                       
  
    #layout.sources_as_real();
  
  for a,r in zip(layout.alignments, results):
    shift, quality = r;
    a_shift = list(a.shift);
    a_shift[axis] = shift[axis];
    a.shift = tuple(a_shift);
    a.quality = quality;
  
  if verbose:
    timer.print_elapsed_time('Alignment: aligning sources in layout along axis=%d!' % axis)
  
  return layout;

@ptb.parallel_traceback
def _align_layout_axis(src1, src2, aid, n_alignments, axis, axis_range, depth, max_shifts, clip, background, verbose):
  if verbose:
    print('Alignment: aligning %r with %r along axis %d, alignment pair %d/%d !' % (src1.tile_position, src2.tile_position, axis, aid, n_alignments));
  
  #reduce source size
  sl = (slice(None),) * src1.ndim;
  if not axis_range in [None, all]:
    p1 = list(src1.position);
    sl1 = list(sl);
    sl1[axis] = slice(axis_range[0] - p1[axis], axis_range[1] - p1[axis])
    sl1 = tuple(sl1);          
    src1 = Slice(source = src1, slicing = sl1);                 
    
    p2 = list(src2.position);
    sl2 = list(sl);
    sl2[axis] = slice(axis_range[0] - p2[axis], axis_range[1] - p2[axis])
    sl2 = tuple(sl2);
    src2 = Slice(source = src2, slicing = sl2);               
      
  #mip axis
  t1 = src1.tile_position;
  t2 = src2.tile_position;
  if not t1 is None and not t2 is None:
    mip_axis = np.where(np.array(t1) - t2 != 0)[0][0];
  else:
    mip_axis = axis;
  
  if mip_axis == axis:
    #take the smallest overlapping dim
    r1 = Region(position = src1.position, shape = src1.shape);
    r2 = Region(position = src2.position, shape = src2.shape);
    overlap1, _ = _overlap_with_shifts(r1, r2, max_shifts = max_shifts);
    mip_shape = None; 
    mip_axis = None; 
    for d,s in enumerate(overlap1.shape):
      if d != axis:
        if mip_shape is None:
          mip_shape = s;
          mip_axis = d;
        elif mip_shape < s:
          mip_shape = s;
          mip_axis = d;
  
  mip_depth = depth[mip_axis];
  
  result = align_2_sources_along_axis_mip(src1, src2, axis = mip_axis, depth = mip_depth, max_shifts = max_shifts, clip = clip, background = background, verbose = False);
  
  if verbose:
    shift, quality = result;
    print('Alignment: aligning %r with %r along axis %d, alignment pair %d/%d done, shift = %r, quality = %.2e!' % (src1.tile_position, src2.tile_position, axis, aid, n_alignments, shift, quality));
         
  return result;



def align_layout_rigid_mip(layout, depth = 10, max_shifts = 10, ranges = None, clip = None, background = None, processes = None, verbose = False):
  """Aligns sources in a layout in a single axis direction only.
  
  Arguments
  ---------
  layout: Layout class
    The layout in which to align the 3d sources in z-direction.
  depth : int or list of ints
    The approximate overlaps of the images in the different dimensions to use for mip projection.
    Only the depth parameter along the relevant axis is used.
  max_shifts : tuple of ints
    The minmal and maximal shift in to consider.
  axis_range : tuple of int or None
    Use only a sub set of the axis range to speed up processing.
  clip : number or None
    If not None, clip the soruces at this value when calculating the alignment.
  background : number or None
    If not None, if the values in the overlap region are less than this number make alignment return -inf quality as there is no signal to use for alignment. 
  processes : int or 'serial' layout.alig    
    Number of processor to use for parallel processing, if 'serial' process in serial.
  verbose : bool
    Print progress information.
  
  Returns
  -------
  layout : Layout 
    The layout with updated axis-alignments.
    
  Note
  ----
  To speed up the calculation, a mip projection is used in the direction of the tiling.
  The tiling dimension are assumed to be alinged with the first image dimensions.
  """    
  if verbose:
    timer = tmr.Timer();
    print('Alignment: rigidly aligning sources in layout: %r!' % layout)
  
  #format the shifts
  if not isinstance(depth, (list, tuple)):
    depth = (depth,) * layout.ndim;
            
  if not isinstance(ranges, list):
    ranges = [ranges] * layout.ndim;    
  
  max_shifts = _format_max_shifts(max_shifts, layout.ndim);
  n_alignments = layout.n_alignments;
  
  _align = ft.partial(_align_layout_ridgid_mip, 
                      n_alignments=n_alignments,
                      depth=depth, max_shifts=max_shifts, ranges=ranges,
                      clip=clip, background=background, verbose=verbose);
  
  if not isinstance(processes, int) and processes != 'serial':
    processes = mp.cpu_count();
  
  if processes == 'serial':
    results = [_align(a.pre, a.post, i) for i,a in enumerate(layout.alignments)];
  else:
    layout.sources_as_virtual();                        
    alignments = layout.alignments;
    print('align_axis')
    with concurrent.futures.ProcessPoolExecutor(processes) as executor:
      results = executor.map(_align, [a.pre for a in alignments], [a.post for a in alignments], range(n_alignments));
    results = list(results);                       
  
    #layout.sources_as_real();
  
  for a,r in zip(layout.alignments, results):
    shift, quality = r;
    a.shift = shift;
    a.quality = quality;
  
  if verbose:
    timer.print_elapsed_time('Alignment: rigidly aligning sources in layout: %r!' % layout)
  
  return layout;


@ptb.parallel_traceback
def _align_layout_ridgid_mip(src1, src2, aid, n_alignments, depth, max_shifts, ranges, clip, background, verbose):
  if verbose:
    print('Alignment: aligning %r with %r, alignment pair %d/%d !' % (src1.tile_position, src2.tile_position, aid, n_alignments));
     
  #mip axis
  t1 = src1.tile_position;
  t2 = src2.tile_position;
  if not t1 is None and not t2 is None:
    mip_axis = np.where(np.array(t1) - t2 != 0)[0][0];
  else:
    mip_axis = None;
  
  if mip_axis is None:
    #take the smallest overlapping dim
    r1 = Region(position = src1.position, shape = src1.shape);
    r2 = Region(position = src2.position, shape = src2.shape);
    overlap1, _ = _overlap_with_shifts(r1, r2, max_shifts = max_shifts);
    mip_shape = None; 
    mip_axis = None; 
    for d,s in enumerate(overlap1.shape):
        if mip_shape is None:
          mip_shape = s;
          mip_axis = d;
        elif mip_shape < s:
          mip_shape = s;
          mip_axis = d;
  
  mip_depth = depth[mip_axis];
  
  #reduce sources to ranges along non-mip axes
  if ranges != [None] * len(ranges):
    sl1 = (); sl2 = ();
    p1 = src1.position; p2 = src2.position;          
    for d,r in enumerate(ranges):
      if d != mip_axis and r is not None:
        sl1 += (slice(r[0] - p1[d], r[1] - p1[d]),)
        sl2 += (slice(r[0] - p2[d], r[1] - p2[d]),)
      else:
        sl1 += (slice(None),);
        sl2 += (slice(None),);      

    src1 = Slice(source = src1, slicing = sl1);
    src2 = Slice(source = src2, slicing = sl2);                  
                   
              
  result = align_2_sources_along_axis_mip(src1, src2, axis = mip_axis, depth = mip_depth, max_shifts = max_shifts, clip = clip, background = background, verbose = False);
  
  if verbose:
    shift, quality = result;
    print('Alignment: aligning %r with %r, alignment pair %d/%d done, shift = %r, quality = %.2e!' % (src1.tile_position, src2.tile_position, aid, n_alignments, shift, quality));
         
  return result;






def _format_max_shifts(max_shifts, ndim):
  """Helper to format the max_shift specifications.
  """
  if isinstance(max_shifts, int):
    max_shifts = [(-max_shifts, max_shifts)] * ndim;
  if isinstance(max_shifts, tuple):
    max_shifts = [max_shifts] * ndim;
  if isinstance(max_shifts, np.ndarray):
    max_shifts = list(max_shifts);
  if not isinstance(max_shifts, list):
    raise ValueError('max_shifts expected to be int, tuple, list or array, found %r!' % (max_shifts));
  if len(max_shifts) != ndim:
    raise ValueError('max_shifts len = %d expected to be of same dimension as the source = %d!' % (len(max_shifts), ndim));
  for d,ms in enumerate(max_shifts):
    if isinstance(ms, int):
      ms = (-ms, ms);
    elif isinstance(ms, list):
      ms = tuple(ms);
    if len(ms) != 2:
      raise ValueError('max_shifts entry at dimension %d expected to be of the firm (min,max), found %r!' % (d, ms));
    if max_shifts[d][0] > max_shifts[d][1]:
      raise ValueError('max_shifts entry at dimension %d expected to be ordered (min <= max), found %r!' % (d, ms));
    max_shifts[d] = ms;
  #max_shifts = np.array(max_shifts, dtype = int);
  return max_shifts;


#########################################################################
### Placement
#########################################################################

def filter_alignments(alignments, min_quality = -np.inf):
  """Filter out alignments that fall below a certain quality level.
  
  Arguments
  ---------
  alignments : list of Alignments classes
    The pairwise alignments between images to optimize globally.
  min_quality : float
    The minimal quality of the alignment.
 
  Returns
  -------
  alignments : list of Alignment classes
    The filtered alignments.
  """
  if min_quality is None:
    return alignments;
  
  max_quality = None;
  if hasattr(min_quality, '__len__'):
    if len(min_quality) == 0:
      return alignments;
    if len(min_quality) == 1:
      min_quality = min_quality[0];
    else:
      max_quality = min_quality[1];
      min_quality = min_quality[0];                               

  new_alignments = [];
  for a in alignments:
    if a.quality > min_quality:
      if max_quality is not None:
        if a.quality < max_quality:
          new_alignments.append(a);
      else:
        new_alignments.append(a);
  
  return new_alignments;


def alignments_from_source_positions(alignments):
  """Update alignment shifts from source positions and shapes
  
  Arguments
  ---------
  alignments : list of Alignments classes
    The alignments pairs of the sources.

  Returns
  -------
  alignments : list of Alignments classes
    The updated alignments. 
  """
  for a in alignments:
    a.shift = (0,) * a.pre.ndim;
  return alignments;
  

def positions_from_tree(alignments, sources = None, min_quality = None, fixed_source = None, lower_to_origin = False):
  """Update source positions from alignments.
  
  Arguments
  ---------
  alignments : list of Alignments classes
    The alignments pairs of the sources.
  sources : list of Source classes or None
    Optional sources to update the positions for, if None extracted from alignment classes.
  min_quality : float
    The minimal quality of the alignment.
  fixed_source : Source class, int or None
    Optional source to kept fixed. If None the first source is placed at the origin.
  lower_to_origin : bool
    If True the lower corner of the aligned images is set to zero.
  
  Returns
  -------
  positions : list of tuple of ints
    The source positions.
    
  Note
  ----
  The result will be a single consistent solution based on a minimal paths between the first and the other sources of the layout.
  """
  #TODO: base on spanning tree with best quality measure
  alignments = filter_alignments(alignments, min_quality = min_quality);
  
  if sources is None:
    sources = sources_from_alignments(alignments);
  nsources = len(sources);
  
  ndim = alignments[0].ndim
  source_to_index = { s : i for i,s in enumerate(sources)};
  
  #construct minimal tree to connect sources
  g = gt.Graph(directed = True);    
  g.add_vertex(nsources);
  p = g.new_edge_property('vector<int>');
  for a in alignments:
    ipre = source_to_index[a.pre];
    ipost = source_to_index[a.post];
    shift = a.displacement;
    e  = g.add_edge(ipre, ipost);
    p[e] = shift;
    e = g.add_edge(ipost, ipre);
    p[e] = tuple(-s for s in shift);
  
  #calculate positions from tree
  start_id = 0;
  if fixed_source is not None:
    start_id = source_to_index[fixed_source];
  
  positions = np.zeros((nsources, ndim), dtype=int);
  for i in range(nsources):
    vlist, elist = gtt.shortest_path(g, g.vertex(start_id), g.vertex(i));
    for e in elist:
      positions[i] += p[e];
  
  #correct for origin and fixed source
  if fixed_source is not None:
    fixed_id = source_to_index[fixed_source];
    fixed_position = fixed_source.position;
    positions = positions - positions[fixed_id] + fixed_position;

  if lower_to_origin:
    min_pos = np.min(positions, axis = 0);
    positions -= min_pos;  
  
  positions = [tuple(p for p in pos) for pos in positions];
  return positions;  


def positions_from_optimization(alignments, sources = None, min_quality = None, fixed_source = None, lower_to_origin = False): # optimize positions !
  """Use least squares optimization to find globally optimal source positions from pairwise alignments.
  
  Arguments
  ---------
  alignments : list of Alignments classes
    The pairwise alignments between images to optimize globally.
  sources : list of Source classes or None
    The sources to optimze the positions for, if None determine from the alignments.
  min_quality : float
    The minimal quality of the alignment.
  fixed_source : Source class, int or None
    Optional source to kept fixed. If None the first source is kept fixed.
  lower_to_origin : bool
    If True, the lower corner of the aligned images is set to zero.
  
  Returns
  -------
  sources : list of Source classes
    The sources with the optimized positions.
    
  Note
  ----
  The error function is sum (x_i + s_ij - x_j)^2 with x_i the image positions and s_ij the pairwise shifts.
  """
  #TODO: incorporate qaulity measure in optimization
  alignments = filter_alignments(alignments, min_quality = min_quality);
  
  nalignments = len(alignments);
  if nalignments <= 0:
    if sources is None:
      return None;
    else:
      return [s.position for s in sources];
    
  #sources
  if sources is None:
    sources = sources_from_alignments(alignments);
  source_to_index = { s: i for i,s in enumerate(sources)};  
  
  if not connected(alignments, sources):
    raise RuntimeError('Sources need to be connected for optimized placement!')
  
  # construct the mappings between node ids and index 1:nimages
  pre_indices = np.unique([source_to_index[a.pre] for a in alignments]);
  post_indices = np.unique([source_to_index[a.post] for a in alignments]);
  node_to_index = np.unique(np.hstack([pre_indices, post_indices]));
  nnodes = len(node_to_index);
  
  index_to_node = -np.ones(np.max(node_to_index)+1, dtype = int);                  
  index_to_node[node_to_index] = np.arange(nnodes);
  
  ndim = sources[0].ndim;
  n = ndim * nalignments;
  m = ndim * (nnodes - 1); # first image is assumed to be fixed at zero

  # derivative of the error gives constraints s - M x == 0
  # s are the displacements, x the centers of the images, M is derived from the error terms

  # s
  s = np.zeros(n);
  k = 0;
  for a in alignments:
    sh = a.displacement;
    for d in range(ndim):
      s[k] = sh[d];
      k = k + 1;
  
  # M
  M = np.zeros((n,m));
  k = 0;
  for a in alignments:
   for d in range(ndim):
      pre_node = index_to_node[source_to_index[a.pre]];
      if pre_node > 0:
         M[k, (pre_node - 1) * ndim + d] = -1;

      post_node = index_to_node[source_to_index[a.post]];
      if post_node > 0:
         M[k, (post_node - 1) * ndim + d] = 1;
      k = k + 1;
  
  #print s
  #print M
  #print np.linalg.pinv(M)
  
  # find the centers of the images via pseudo inverse
  positions = np.dot(np.linalg.pinv(M), s);  
  positions = np.hstack([np.zeros(ndim), positions]);
  positions = np.reshape(positions, (-1, ndim));
  positions = np.asarray(np.round(positions), dtype = int);
  
  #correct for origin and fixed source
  if fixed_source is not None:
    fixed_id = source_index(sources, fixed_source);
    fixed_position = fixed_source.position;
  else:
    fixed_id = 0;
    fixed_position = sources[0].position;
  positions = positions - positions[fixed_id] + fixed_position;
  
  if lower_to_origin:
    min_pos = np.min(positions, axis = 0);
    positions -= min_pos;  
  
  positions = [tuple(p for p in pos) for pos in positions];
  return positions;


def place_layout(layout, method = 'optimization', min_quality = None, lower_to_origin = False, verbose = False):
  """Place the sources in a layout in a consistent way.
  
  Arguments
  ---------
  layout : Layout class
    The layout in which the Sources will be placed in a consistent way.
  method : 'optimization' or 'tree'
    The method to use to place the sources.
  min_quality : float
    The minimal quality of the alignments to include in the placement process.
  lower_to_origin : bool
    If True, the lower corner of the aligned sources is set to zero.
  
  Returns
  -------
  layout : Layout class
    The layout with the optimized positions of the sources.
  """
  #if not isinstance(layout, Layout):
  #  raise ValueError("A Layout class is expected as input!");
  
  methods = ['optimization', 'tree'];
  if method not in methods:
    ValueError('Method %r not in %r' % (method, methods));
    
  if verbose:
    timer = tmr.Timer();
    print('Placement: placing %r!' % layout);
  
  #determine connected components
  components, component_sources = layout.connected_components(min_quality = min_quality, with_sources = True);                                                      
  #print components, component_sources
  
  component_positions = [];                               
  for component, sources in zip(components, component_sources):
    fixed_source = sources[(len(sources)-1)//2];
    if method == 'optimization':
      positions = positions_from_optimization(alignments = component, sources = sources, fixed_source = fixed_source, min_quality = None, lower_to_origin = False);
    elif method == 'tree':
      positions = positions_from_tree(alignments = component, sources = sources, fixed_source = fixed_source, min_quality = None, lower_to_origin = False);

    component_positions.append(positions);
  
  #order positions according to layout                                  
  positions = [];                                
  for s in layout.sources:                                
   for sources, pos in zip(component_sources, component_positions):
     for cs,p in zip(sources, pos):
       if s == cs:
         positions.append(p);
         break;
  #print(positions) 
  
  if lower_to_origin:
    positions = np.array(positions, dtype=int);     
    positions -= np.min(positions, axis = 0);
    positions = [tuple(p for p in pos) for pos in positions];             

  layout.set_source_positions(positions = positions, update_alignments = False);
  
  if verbose:
    timer.print_elapsed_time('Placement: placing %r' % layout)
  
  return layout;



def place_layout_axis(layout, axis = 2, method = 'optimization', min_quality = None, lower_to_origin = False, verbose = False):
  """Places the sources in a layout along a single axis only.
  
  Arguments
  ---------
  layout: Layout class
    The layout of the stacks.
  method : 'optimization' or 'tree'
    The method to use to place the sources.
  min_quality : float
    The minimal quality of the alignment.
  lower_to_origin : bool
    If True the lower corner of the aligned images is set to zero.    
  
  Returns
  -------
  layout : Layout 
    The layout with updated z-alignments.  
  """
  if not isinstance(layout, Layout):
    raise ValueError("A Layout class is expected as input!");
    
  if verbose:
    timer = tmr.Timer();
    print('Placement: placing %r along axis %d!' % (layout, axis))
  
  sources_1d = [SourceRegion(position = s.position[axis:axis+1], shape = s.shape[axis:axis+1]) for s in layout.sources];  
  source_to_1d = {s : s1 for s,s1 in zip(layout.sources, sources_1d)};
  
  alignments_1d = [Alignment(pre = source_to_1d[a.pre], post = source_to_1d[a.post], shift = a.shift[axis:axis+1], quality = a.quality) for a in layout.alignments];

  layout_1d = Layout(sources = sources_1d, alignments = alignments_1d);
  
  place_layout(layout_1d, method = method, min_quality = min_quality, lower_to_origin = lower_to_origin);
  
  for s,p in zip(layout.sources, layout_1d.source_positions()):
    position = list(s.position);
    position[axis] = p[0];
    s.position = tuple(position);
    
  for a in layout.alignments:
    shift = list(a.shift);
    shift[axis] = 0;
    a.shift = tuple(shift);
  
  if verbose:
    timer.print_elapsed_time('Placement: placing %r along axis %d!' % (layout, axis))
  
  return layout;


########################################################################################
### Stitching
########################################################################################


def embedding(sources, shape = None, position = None):
  """Splits a set of co-axial sources into a minimal set of non-overlaping regions.
  
  Arguments
  ---------
  sources : list of Source or Region classes
    The sources to embed in a full image.
  shape : tuple of ints or None
    Optional fixed shape of the full image, if None use the minimal shape that fits all sources.
  position : tuple of ints or None
    Optional position form which to start the stitching. Together with shape this can be used to restrict the stitching region. 
    If None, use the minimal position that fits all the contributing sources.
    
  Returns
  -------
  shape : tuple of int
    The shape that encapsulates all the regions.
  position : tuple of int
    The lowest corner of all the regions.
  regions : list of Overlap classes.
    The regions of different overlaps of the individual sources.    
  
  Note
  ----
  The result can be used to stitch the images.
  """
  regions = [];
  for s in sources:
    region = Overlap(position = s.position, shape = s.shape, sources = [s]);
    regions = _add_overlap_region(regions, region);
  #print regions
  #print [r.position for r in regions]
  #print [r.shape for r in regions]
  
  
  if position is None:
    position = tuple(np.min([r.position for r in regions], axis = 0));
    
  if shape is None:
    shape = tuple(np.max([r.upper for r in regions], axis = 0));
  shape = tuple(max(s,0) for s in shape);
  #print position, shape
  
  # reduce to position
  new_regions = [];
  for i,r in enumerate(regions):
    r.lower = tuple(p if l < p else l for l,p in zip(r.lower, position));
    r.upper = tuple(p if u < p else u for u,p in zip(r.upper, position));

    if np.all([u > l for u,l in zip(r.upper, r.lower)]):
      new_regions.append(r);
  regions = new_regions;
  
  #reduce to shape
  new_regions = [];
  ps = np.array(position, dtype=int) + shape;
  for i,r in enumerate(regions):
    r.lower = tuple(p if l > p else l for l,p in zip(r.lower, ps));
    r.upper = tuple(p if u > p else u for u,p in zip(r.upper, ps));

    if np.all([u > l for u,l in zip(r.upper,r.lower)]):
      new_regions.append(r);
  regions = new_regions;
  
  return position, shape, regions;


def _overlap(region1, region2):
  """Helper to determine overlap between two regions."""
  ovl = np.max([region1.lower, region2.lower], axis = 0);
  ovu = np.min([region1.upper, region2.upper], axis = 0);
  
  if np.any(ovu - ovl - 1 < 0):
    return None;
  else:
    return Overlap(lower = ovl, upper = ovu);


def _split_region(r, o):
  """Split region into covering rectangles including the overlap region."""
  split = [o];
  
  rl,ru = r.lower, r.upper;
  ol,ou = o.lower, o.upper;
  
  #split along axes
  for d in range(r.ndim):
    if rl[d] < ol[d]:
      l = ol[:d] + rl[d:];
      u = ou[:d] + (ol[d],) + ru[d+1:];
      split.append(Overlap(lower = l, upper = u));
    
    if ou[d] < ru[d]:
      l = ol[:d] + (ou[d],) + rl[d+1:];
      u = ou[:d] + ru[d:];
      split.append(Overlap(lower = l, upper = u));
   
  return split;   


def _add_overlap_region(regions, region):
  """Helper to determine overlap regions."""
  # try too add the full new region first
  regsadd = [region];

  # the non-overlapping rectangles to be checked for overlap with regsadd
  regscheck = regions;

  # rectangles that will not have any further overlap with regsadd
  regsnew   = [];
  
  while len(regscheck) > 0 and len(regsadd) > 0:
    # chek the next one in list
    rc = regscheck[0];
    #print 'add:', regsadd
    #print 'check:', regscheck
    #print 'new:', regsnew
      
    #check if overlap with any of the regions to be added
    found = False;
    for a in range(len(regsadd)):
      ra = regsadd[a];
      ov = _overlap(rc, ra);
      #print 'chk,add %d:' % a, rc, ra
      #print 'ovl %d:' % a, ov
      
      if ov is not None: 
        #print 'overlap: ', rc, ra, ov
        
        # split region to check
        split = _split_region(rc, ov);
        for s in split:
          s.sources = rc.sources;
        
        #add id of immage to add to overlapping region 
        #-> cannot overlap with other region -> safe to put into new list
        sources = split[0].sources;
        sources += tuple(s for s in ra.sources if s not in sources);
        split[0].sources = sources
        #print 'split1:', split        
        
        regsnew.append(split[0]); #.copy()); 
        
        # other non-overlapping regions from split need to be checked again
        #regscheck = [s.copy() for s in split[1:]] + regscheck[1:];
        regscheck = split[1:] + regscheck[1:];
        
        #split added region
        split = _split_region(ra, ov)[1:];
        for s in split:
          s.sources = ra.sources;
        #print 'split2:', split        
        
        # add all non-verlapping regions to the 'to be added' list
        #regsadd = regsadd[:a] + [r.copy() for r in split[1:]] + regsadd[a+1:];
        regsadd = regsadd[:a] + split + regsadd[a+1:];
          
        # start cehcking anew
        found = True;
        break;
            
    if not found:   # -> check region not overlapping -> add to new list and remove from checklist
      regsnew.append(rc);
      regscheck = regscheck[1:];
        
  regsnew = regsnew + regscheck + regsadd;
  #print 'done add:', regsnew
  
  return regsnew;


def stitch_by_function(layout, sink = None, function = np.max):
  """Stitch sources according to shifts applying a specific function in the overlapping regions.
  
  Arguments
  ---------
  layout : Layout class
    The layout to use for the sources.
  sink : array like or None
    The sink to write the result to.
  function : function
    The function to apply in overlapping regions, e.g. np.max or np.mean.
  
  Returns
  -------
  stitched : array
    The stitched array.
  """
  # determine all the overlap regions
  position, shape, regions = layout.embedding();
  
  # stitch image
  if 'axis' in insp.getargspec(np.max).args:
    function = ft.partial(function, axis = 0);
  
  if sink is None:
    stitched = np.zeros(shape, dtype = layout.dtype, order = layout.order);
  else:
    stitched = sink;
  
  for r in regions:
    nsources = len(r.sources);
    
    if nsources > 1:
      rd = [s[sl] for s,sl in zip(r.sources, r.source_slicings())];
      rd = function(rd);
    else:
      rd = r.sources[0][r.source_slicings()[0]];
    
    stitched[r.local_slicing(position = position)] = rd;
  
  return stitched;


def stitch_weights(shape):
  """Returns the weights of a source of a given shape to use in stitching routines.
  
  Arguments
  ---------
  shape : tuple of ints
    The shape of the source.
  
  Returns
  -------
  weights : array
    The weigthsfor the pixels of the source.
  """ 
  ranges = [range(s) for s in shape];
  mesh = np.meshgrid(*ranges, indexing = 'ij');
  mesh = [np.min([m, np.max(m) - m], axis = 0) for m in mesh];
  weights = np.min(mesh, axis = 0) + 1;  
  return weights;
  

def stitch_by_function_with_weights(layout, sink = None, function = np.sum, weight_function = stitch_weights):
  """Stitch sources applying a specific weighting function in the overlapping regions.
  
  Arguments
  ---------
  layout : Layout class
    The layour of the sources.
  sink : array like or None
    The sink to write the result to.
  function : function
    The function to apply in overlapping regions, e.g. np.sum or np.mean.
  weight_function : function
    A function that returns an array of pixel weights given the source shape.
  
  Returns
  -------
  stitched : array
    The stitched array.
  """
  # determine all the overlap regions
  position, shape, regions = layout.embedding();
  
  # stitch image
  if 'axis' in insp.getargspec(function).args:
    function = ft.partial(function, axis = 0);
  
  if sink is None:
    stitched = np.zeros(shape, dtype = layout.dtype, order = layout.order);
  else:
    stitched = sink;
  
  #determine weights
  shapes = [s.shape for s in layout.sources];
  if shapes.count(shapes[0]) == len(shapes):
    same_shape = True;
    w = weight_function(shapes[0]);
  else:
    same_shape = False;
    source_to_index = {s : i for i,s in enumerate(layout.sources)};
    w = [weight_function(s) for s in shapes];
  
  
  for r in regions:
    nsources = len(r.sources);
    #print nsources, r
    
    if nsources > 1:
      rd = np.zeros((nsources,) + r.shape);
      wd = np.zeros((nsources,) + r.shape);
      
      for i,s,sl in zip(range(len(r.sources)), r.sources, r.source_slicings()):
        #print i, s, sl
        rd[i] = s[sl];
        if same_shape:
          wd[i] = w[sl];
        else:
          wd[i] = w[source_to_index[s]][sl];        
        
      rd = function(rd, wd);
    else:
      s = r.sources[0];
      rd = s[r.source_slicings()[0]];
    
    stitched[r.local_slicing(position = position)] = rd;  
  
  return stitched;


def stitch_by_mean(layout, sink = None):
  """Stitch sources according to shifts applying mean function in the overlap regions.
  
  Arguments
  ---------
  layout : Layout class
    The layout of the sources.
  sink : array like or None
    The sink to write the result to.
  
  Returns
  -------
  stitched : array
    The stitched array.
  """
  return stitch_by_function(layout, sink = sink, function = np.mean);


def stitch_by_max(layout, sink = None):
  """Stitch sources according to shifts applying max function in the overlap regions.
  
  Arguments
  ---------
  layout : Layout class
    The layout of the sources.
  sink : array like or None
    The sink to write the result to.
  
  Returns
  -------
  stitched : array
    The stitched array.
  """ 
  return stitch_by_function(layout, sink = sink, function = np.max);
  

def stitch_by_min(layout, sink = None):
  """Stitch sources according to shifts applying min function in the overlap regions.
  
  Arguments
  ---------
  layout : Layout class
    The layout of the sources.
  sink : array like or None
    The sink to write the result to.
  
  Returns
  -------
  stitched : array
    The stitched array.
  """ 
  return stitch_by_function(layout, sink = sink, function = np.min);


def stitch_by_interpolation(layout, sink = None):
  """Stitch sources according to shifts applying linear interpolation in the overlap regions.
  
  Arguments
  ---------
  layout : Layout class
    The layout of the sources.
  sink : array like or None
    The sink to write the result to.
  
  Returns
  -------
  stitched : array
    The stitched array.
  """ 
  def weigthed_mean(data, weights):
    return np.mean(data * weights, axis = 0) / np.mean(weights, axis = 0);
  return stitch_by_function_with_weights(layout, sink = sink, function = weigthed_mean, weight_function = stitch_weights);

  
def stitch_by_interpolation_adjust_max(layout, sink = None):
  """Stitch sources according to shifts applying linear interpolation in the overlap regions.
  
  Arguments
  ---------
  layout : Layout class
    The layout of the sources.
  sink : array like or None
    The sink to write the result to.
  
  Returns
  -------
  stitched : array
    The stitched array.
  """ 
  def weigthed_mean_adjust(data, weights):
    m = np.max(data);
    s = np.mean(data * weights, axis = 0) / np.mean(weights, axis = 0);
    ms = np.max(s);
    if ms != 0:
      s *= float(m) / ms;
    return s;
  
  return stitch_by_function_with_weights(layout, sink = sink, function = weigthed_mean_adjust, weight_function = stitch_weights);


def _stitching_function_from_method(method):
  """Helper to convert stitching method string to function"""
  method_map = {'interpolation'            :  stitch_by_interpolation,
                'interpolation-adjust-max' :  stitch_by_interpolation_adjust_max,
                'max'                      :  stitch_by_max,
                'min'                      :  stitch_by_min,
                'mean'                     :  stitch_by_mean};
  
  if method not in method_map.keys():
    ValueError('Method %r not in %r' % (method, method_map.keys()));  
  
  return method_map[method];


def stitch_layout(layout, sink = None, method = 'interpolation', verbose = False):
  """Stitch a layout according to its current alignment.
  
  Arguments
  ---------
  layout : Layout class
    The layout of the sources.
  sink : array like or None
    The sink to write the result to.
  method : str
    The method to use for the stitching: 'interpolation', 'max', 'min', 'mean'
  verbose : bool
    If True, print progress information.
  
  Returns
  -------
  stitched : array
    The stitched array.
  """
  if verbose:
    timer = tmr.Timer();
    print('Stitching: Stitching %r with method %s.' % (layout, method));

  function = _stitching_function_from_method(method);
  result = function(layout, sink = sink);
  
  if verbose:
    timer.print_elapsed_time('Stitching: Stitching %r with method %s' % (layout, method));
  
  return result;


########################################################################################
### Visualization
########################################################################################

def overlay_sources(sources, colors = None, percentile = 98, normalize = True):
  """Overlays the sources to check thier placement.
  
  Arguments
  ---------
  layout : Layout class
    The layout with the sources to overlay.
  colors : list of tuple of floats or color names
    The optional RGB colors to use.
  percentile : int
    Use this percentile as upper cutoff in the resulting image to enhance contrast.
  normalize : bool
    If True normalize image to floats between 0 and 1.
  
  Returns
  -------
  image : array
    A color image. 
  """
  layout = Layout(sources = sources);
  return overlay_layout(layout, colors = colors, percentile = percentile, normalize = normalize);


def layout_coloring(layout, colors = None, color_ids = None):
  sources = layout.sources;
  nsources = len(sources); 
  
  if color_ids is None:
    #find sources that overlap
    edges = [];
    for i,s in enumerate(sources):
      p1 = np.array(s.position, dtype=int);
      s1 = s.shape;
      for j in range(i+1,nsources):
        p2 = np.array(sources[j].position, dtype=int);
        s2 = sources[j].shape;
        if np.all(np.max([p1,p2], axis = 0) < np.min([p1+s1, p2+s2], axis = 0)):
          edges.append((i,j));
    
    # find graph coloring of overlap structure
    g = gt.Graph(directed = False);
    g.add_vertex(n = nsources);
    g.add_edge_list(edges);  
    color_ids = np.array(gtt.sequential_vertex_coloring(g).a);
  
  if colors == 'ids':
    return color_ids;
  
  ncols = np.max(color_ids) + 1;
  if colors is None:
    if ncols <= 2:
      colors = [[1, 0, 1], [0, 1, 0]];
    elif ncols <= 4:
      colors = [[0.5, 0.5, 0], [0, 0.5, 0.5], [0.5, 0, 0], [0, 0, 0.5]];
    else: 
      colors = [[0.25, 0,     0   ],
                [0,    0.25,  0   ],
                [0,    0,     0.25],
                [0.25, 0.25,  0   ],
                [0,    0.25,  0.25],
                [0.25, 0,     0.25],
                [0.125,0.25,  0   ],
                [0.125,0,     0.25]];
  colors = [col.color(c, alpha = False, as_int = True) for c in colors];
  colors = np.pad(colors[:ncols], ((0,max(0, ncols - len(colors))),(0, 0)), 'wrap');
  return colors[color_ids];


def overlay_layout(layout, colors = None, percentile = 98, normalize = True, color_ids = None):
  """Overlays the sources to check their placement.
  
  Arguments
  ---------
  layout : Layout class
    The layout with the sources to overlay.
  colors : list of tuple of floats or color names
    The optional RGB colors to use.
  percentile : int
    Use this percentile as upper cutoff in the resulting image to enhance contrast.
  normalize : bool
    If True normalize image to floats between 0 and 1.
  
  Returns
  -------
  image : array
    A color image. 
  """
  #full shape
  full_shape = tuple(layout.extent);
  full_lower = layout.lower;
  
  source_colors = layout_coloring(layout, colors=colors, color_ids=color_ids);
  if colors == 'ids':
    image = [np.zeros(full_shape) for i in range(max(source_colors)+1)];
  else:
    image = np.zeros(full_shape + (3,));
  
  # construct full image
  sources = layout.sources;
  for s,c in zip(sources, source_colors):
    l = s.lower; u = s.upper;
    r = tuple(slice(ll - fl , uu - fl) for ll,uu,fl in zip(l, u, full_lower));
    if colors == 'ids':
      image[c][r] += s[:];
    else:
      r += (slice(None),);
      image[r] += np.multiply.outer(s[:], c);
  
  if percentile is not None:
    p = np.percentile(image, percentile); 
    image[image > p] = p;
  if normalize:
    image = image / image.max();
  return image;


def plot_sources(sources, colors = None, percentile = 98, normalize = True):
  """Overlays and plots sources in a layout to check alignment.
  
  Arguments
  ---------
  layout: Layout class
    The layout to use for plotting.
  colors : list of colors or None
    The optional RGB colors to use.
  percentile : int
    Use this percentile as upper cutoff in the resulting image to enhance contrast.
  normalize : bool
    If True normalize image to floats between 0 and 1.
  
  Returns
  -------
  image : array
    A color image of the overlayed sources.
  """
  layout = Layout(sources = sources);
  return plot_layout(layout, colors = colors, percentile = percentile, normalize = normalize)


def plot_layout(layout, colors = None, percentile = 98, normalize = True, color_ids = None):
  """Overlays and plots sources in a layout to check alignment.
  
  Arguments
  ---------
  layout: Layout class
    The layout to use for plotting.
  colors : list of colors or None
    The optional RGB colors to use.
  percentile : int
    Use this percentile as upper cutoff in the resulting image to enhance contrast.
  normalize : bool
    If True normalize image to floats between 0 and 1.
  
  Returns
  -------
  image : array
    A color image of the overlayed sources.
  """
  img = overlay_layout(layout, colors = colors, percentile = percentile, normalize = normalize, color_ids = color_ids);
  if img.ndim == 3:
    plt.imshow(np.transpose(img, [1,0,2])[:,:,:], origin = 'lower');
    plt.tight_layout()
  else:
    p3d.plot(img);
  

def plot_regions(regions, sources = None, cmap = plt.cm.rainbow, annotate = True, axes = [0,1]):
  """Overlays and plots regions to check the alignment.
  
  Arguments
  ---------
  regions : list of Region classes.
    The regions to plot.
  cmap : colormap
    The color map to use to color the regions.
  annotate : bool
    Use annotaton or not.
  """
  if len(regions) == 0:
    return;
  
  if sources is None:
    sources = list(np.unique(np.hstack([r.sources for r in regions])));
  
  sources_to_ids = { s : i for i,s in enumerate(sources)};
    
  ndim = regions[0].ndim;
  if ndim != 2:
    warnings.warn('The regions are plotted in 2d using axes %r but are %dd!' % (axes, ndim));
    
  if axes is None:
    axes = [0,1];
  
  ax = plt.gca();
  rmin = np.zeros(ndim);
  rmax = np.zeros(ndim);
  for i,r in enumerate(regions):
    rec = plt.Rectangle(np.array(r.lower)[axes], r.upper[axes[0]] - r.lower[axes[0]], r.upper[axes[1]] - r.lower[axes[1]], fill = True, alpha = 0.3, color = cmap(float(i)/len(regions)));
    ax.add_patch(rec);
    if annotate:
      ids = [sources_to_ids[s] for s in r.sources]
      ax.annotate(str(tuple(ids)), xy=rec.get_xy(), xytext=(0, 0), textcoords='offset points', 
                  color='w', ha='center', fontsize=8,
                  bbox=dict(boxstyle='round, pad=.5', fc=(.1, .1, .1, .92), ec=(1., 1., 1.), lw=1, zorder=1));
    rmin = np.min([rmin, r.lower], axis = 0);
    rmax = np.max([rmax, r.upper], axis = 0);
                                          
  #limits
  plt.xlim((rmin[axes[0]], rmax[axes[0]]));
  plt.ylim((rmin[axes[1]], rmax[axes[1]])); 


def plot_alignments(alignments, sources = None, axes = [0,1], annotate = True, min_quality = -np.inf, cmap =  plt.cm.hot):
  """Plots the alignments with their quality"""
  
  ndim = alignments[0].ndim;
  if ndim != 2:
    warnings.warn('The regions are plotted in 2d using axes %r but are %dd!' % (axes, ndim));
    ndim = 2;
    
  if axes is None:
    axes = [0,1];  
  
  q = np.array([a.quality for a in alignments]);
  
  q_max = np.max(q);
  if q_max == -np.inf:
    q_max = 0;
    q_min = -1;
  else:
    q_min = np.min(q);
    if q_min == -np.inf:
      q_min = np.min(q[q > -np.inf]);
    q_min = max(min_quality, q_min);
    if q_max <= q_min:
      q_max = q_min + 1;

  if sources is None:
    sources = sources_from_alignments(alignments);
  
  #plot
  ax = plt.gca();
  rmin = np.zeros(ndim);
  rmax = np.zeros(ndim);
  
  for s in sources:
    #plot the source boundary
    lower = np.array(s.lower)[axes];
    upper = np.array(s.upper)[axes];
    
    rec = plt.Rectangle(lower, upper[0] - lower[0], upper[1] - lower[1], fill = True, alpha = 0.3, color = 'gray');
    ax.add_patch(rec);
    #if annotate:
    #  ax.annotate(str(s.id), xy=rec.get_xy(), xytext=(0, 0), textcoords='offset points', 
    #              color='w', ha='center', fontsize=8,
    #              bbox=dict(boxstyle='round, pad=.5', fc=(.1, .1, .1, .92), ec=(1., 1., 1.), lw=1, zorder=1));
    rmin = np.min([rmin, lower], axis = 0);
    rmax = np.max([rmax, upper], axis = 0);
  
  plt.xlim((rmin[axes[0]], rmax[axes[0]]));
  plt.ylim((rmin[axes[1]], rmax[axes[1]])); 


  for a in alignments:
    #print a, a.pre.lower, a.pre.upper
    p1 = 0.5 * (np.array(a.pre.lower)[axes] + np.array(a.pre.upper)[axes]);
    p2 = 0.5 * (np.array(a.post.lower)[axes] + np.array(a.post.upper)[axes]);
    if a.quality > -np.inf:
      c = cmap((float(a.quality) - q_min)/(q_max - q_min));
    else:
      c = 'black';
    
    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color = c, linewidth = 1);
    if annotate:
      ax.annotate("%.2e" % a.quality, xy=0.5*(p1+p2), xytext=(0, 0), textcoords='offset points', 
                  color='w', ha='center', fontsize=8,
                  bbox=dict(boxstyle='round, pad=.5', fc=(.1, .1, .1, .92), ec=(1., 1., 1.), lw=1, zorder=1));



#TODO: join overlap with real code
def layout_along_axis_mip(src1, src2, axis = 2, depth = 10, max_shifts = 10, ranges = None, verbose = False):
  """Layout corresponding to a mip projected alignment
  
  Arguments
  ---------
  Returns
  -------
  """  
  #format the shifts
  ndim = src1.ndim;
  if not isinstance(depth, (list, tuple)):
    depth = (depth,) * ndim;
            
  if not isinstance(ranges, list):
    ranges = [ranges] * ndim;    
  
  max_shifts = _format_max_shifts(max_shifts, ndim);

  #mip axis
  t1 = src1.tile_position;
  t2 = src2.tile_position;
  if not t1 is None and not t2 is None:
    mip_axis = np.where(np.array(t1) - t2 != 0)[0][0];
  else:
    mip_axis = None;
  
  if mip_axis is None:
    #take the smallest overlapping dim
    r1 = Region(position = src1.position, shape = src1.shape);
    r2 = Region(position = src2.position, shape = src2.shape);
    overlap1, _ = _overlap_with_shifts(r1, r2, max_shifts = max_shifts);
    mip_shape = None; 
    mip_axis = None; 
    for d,s in enumerate(overlap1.shape):
        if mip_shape is None:
          mip_shape = s;
          mip_axis = d;
        elif mip_shape < s:
          mip_shape = s;
          mip_axis = d;
  
  mip_depth = depth[mip_axis];
  
  #reduce sources to ranges along non-mip axes
  if ranges != [None] * len(ranges):
    sl1 = (); sl2 = ();
    p1 = src1.position; p2 = src2.position;          
    for d,r in enumerate(ranges):
      if d != mip_axis and r is not None:
        sl1 += (slice(r[0] - p1[d], r[1] - p1[d]),)
        sl2 += (slice(r[0] - p2[d], r[1] - p2[d]),)
      else:
        sl1 += (slice(None),);
        sl2 += (slice(None),);      

    src1 = Slice(source = src1, slicing = sl1);
    src2 = Slice(source = src2, slicing = sl2);                  
  
  #mip
  mip_depth = depth[mip_axis];
  max_shifts = max_shifts[:mip_axis] + max_shifts[mip_axis+1:];
  
  s1 = src1.shape;
  s2 = src2.shape;
  
  # max intensity projections 
  sub1 = [slice(None)] * ndim;
  sub1[mip_axis] = slice(max(0, s1[mip_axis] - mip_depth), None);
  sub1 = tuple(sub1);
  
  sub2 = [slice(None)] * ndim;
  sub2[mip_axis] = slice(None, min(mip_depth, s2[mip_axis]));
  sub2 = tuple(sub2);          
              
  # calculate max projection along axis  
  mip1 = max_intensity_projection(src1[sub1], axis = mip_axis);
  mip2 = max_intensity_projection(src2[sub2], axis = mip_axis);                               
  
  #add position information
  p1 = src1.position[:mip_axis] + src1.position[mip_axis+1:];
  p2 = src2.position[:mip_axis] + src2.position[mip_axis+1:];
  #print axis, p1, p2
  
  mip1 = Source(mip1, position = p1, tile_position = src1.tile_position);
  mip2 = Source(mip2, position = p2, tile_position = src2.tile_position);
  #print mip1, mip2              

  return Layout(sources = [mip1, mip2]);
 
 
def overlay_along_axis_mip(src1, src2, axis = 2, depth = 10, max_shifts = 10, ranges = None, verbose = False):
  layout = layout_along_axis_mip(src1, src2, axis=axis, depth=depth, max_shifts=max_shifts, ranges=ranges, verbose=verbose);
  return overlay_layout(layout)


def plot_along_axis_mip(src1, src2, axis = 2, depth = 10, max_shifts = 10, ranges = None, verbose = False):
  layout = layout_along_axis_mip(src1, src2, axis=axis, depth=depth, max_shifts=max_shifts, ranges=ranges, verbose=verbose);
  return plot_layout(layout)
 

########################################################################################
### Tests
########################################################################################

def _test():
  import ClearMap.Alignment.StitchingBase as stb;
  reload(stb);
  
  #overlaps and embeddings
  r1 = stb.Region(lower = (0,0), upper = (100,100));
  r2 = stb.Region(lower = (80,20), upper = (180, 120));
  r = stb.embedding([r1,r2])
  print(r)
  
  import numpy as np
  import ClearMap.Tests.Files as tfs
  data = np.load(tfs.vasculature_pre)[:,:100,:100];
  
  #divide data
  data1 = data[:210,:,:]
  data2 = data[200:,:,:];
  
  reload(stb)
  s1 = stb.Source(source=data1, position = (0,0,0));
  s2 = stb.Source(source=data2, position = (210,0,0));
  stb.align_2_sources(s1, s2, max_shifts = 20)
  
  
  #Slicing layouts
  l = stb.Layout(sources = [s1, s2], alignments = [stb.Alignment(pre=s1, post=s2)])
  
  s = l.slice_along_axis(50, axis = 2)
  
  s.align(max_shifts = 20, verbose = True)
  s.place(verbose = True)
  s.plot_alignments()
  
  d = s.stitch(verbose = True, method = 'max')
  stb.p3d.plot(d)
  
  np.all(d == data[:,:,50])
  
  #Tiled layouts
  reload(stb)
  l = stb.TiledLayout([data1, data2], overlaps = 15);
  l.plot_regions()
  
  l.align(max_shifts = (-10,30), verbose=True)
  l.plot_alignments()

  l.sources
  l.place(lower_to_origin=True, verbose=True)
  l.sources
  l.plot_alignments()
  
  d = l.stitch(method='interpolation', verbose=True)
  np.all(d == data)
  stb.dv.plot(d)
  
  s = l.slice_along_axis(50, axis = 2);
  s.align(verbose = True)
  s.place(verbose = True)
  d = l.stitch(method='interpolation', verbose=True)
  
  #2d Tiles
  data = np.load(tfs.vasculature_pre)[:,:,:100];
  data.shape
  
  #divide data
  data1 = data[:220,:220,:];
  data2 = data[200:,:215,:];
  data3 = data[:208,200:,:];
  data4 = data[193:,198:,:];
  
  tiling = [[data1, data3],[data2, data4]];
  #this should result 
  
  reload(stb)
  l = stb.TiledLayout(tiling, overlaps = (10,10))
  print(l.tile_positions)
  l.center_tile_source()
  l.source_positions()
  
  l.align_on_tiling(max_shifts = (-25,25), verbose = True)
  l.place(method = 'optimization', lower_to_origin = True, verbose = True)
  l.plot_alignments()  
  
  s = l.stitch(method = 'max', verbose = True)
  stb.dv.plot(s)
  np.all(s == data)
  
  # Tiles from files
  expression = stb.te.Expression('test_<X,I,2>_<Y,I,4>.tif');
  
  import ClearMap.IO.IO as io
  for i in range(len(tiling)):
    for j in range(len(tiling[i])):
      io.write(expression.string({'X' : i, 'Y' : j}), tiling[i][j])  
  
  reload(stb)
  l = stb.TiledLayout(expression = expression)  
  print(l.tile_positions)
  print([src.location for src in l.sources])
  
  l.align_on_tiling(max_shifts = (-25,25), verbose = True)
  l.place(method = 'optimization', lower_to_origin = True, verbose = True) 
  
  s = l.stitch(method = 'max', verbose = True)
  
  stb.dv.plot(s)
  
  np.all(s == data)
  
  #cleanup
  for i in range(len(tiling)):
    for j in range(len(tiling[i])):
      stb.io.delete_file(expression.string({'X' : i, 'Y' : j}))  
  