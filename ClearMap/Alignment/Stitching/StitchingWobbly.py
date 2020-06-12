# -*- coding: utf-8 -*-
"""
StitchingWobbly
===============

Wobbly stitching module handles the alginment of large volumetric data sets.

The module alings stacks allowing them to wobble around a wobble axis, i.e. 
due to oscillatory movements during image aquisition.
"""
__author__    = 'Christoph Kirst <christoph.kirst.ck@gmail.com>'
__license__   = 'GPLv3 - GNU General Pulic License v3 (see LICENSE)'
__copyright__ = 'Copyright © 2020 by Christoph Kirst'
__webpage__   = 'http://idisco.info'
__download__  = 'http://www.github.com/ChristophKirst/ClearMap2'

import numpy as np
import functools as ft
import multiprocessing as mp
import concurrent.futures

import graph_tool as gt
import graph_tool.topology as gtt


import ClearMap.IO.IO as io
import ClearMap.IO.Slice as slc

import ClearMap.Alignment.Stitching.StitchingRigid as strg
import ClearMap.Alignment.Stitching.Tracking as trk

import ClearMap.ParallelProcessing.ParallelTraceback as ptb

import ClearMap.Utils.Timer as tmr
import ClearMap.Utils.TagExpression as te

###############################################################################
###  Layout
###############################################################################

class WobblySource(strg.Source):
  """Class to handle source data and positions of wobbly stacks."""
  
  ISOLATED = -2
  INVALID = -1
  VALID = 0
  FIXED = 2
  
  status_to_description = {ISOLATED : 'isolated',
                           INVALID  : 'invalid',
                           VALID    : 'valid',
                           FIXED    : 'fixed'}
  
  def __init__(self, source, wobble = None, status = None, axis = 2, position = None, tile_position = None):
    """Source class construtor.
    
    Arguments
    ---------
    source: string, array or Source class
      The image source.
    position : list of tuple of ints or None
      The positions of the source's 'lower' corner of the source.
    wobble : list of list of ints or None
      The positions of the individual planes in this wobbly source.
    """
    strg.Source.__init__(self, source = source, position = position, tile_position = tile_position);
    
    self._axis = int(axis);    
    
    if wobble is None:
      shape = super(WobblySource, self).shape;
      self._wobble = np.zeros((shape[self._axis], len(shape) - 1), dtype = int);
    else:
      self._wobble = np.array(wobble, dtype = int);  
  
    if status is None:
      shape = super(WobblySource, self).shape;
      self._status = np.full(shape[self._axis], self.VALID, dtype = int);
    else:
      self._status = np.array(status, dtype = int);  
  
  
  @property
  def name(self):
    return 'Wobbly-' + self.source.name;
  
  
  @property
  def axis(self):
    """The axis along which the source is assumed to be wobbly.
    
    Returns
    -------
    axis : int
      The wobble axes.
    """
    return self._axis;  
  
  @property
  def coordinate(self):
    return self._position[self.axis];
  
  @property
  def height(self):
    return self._source.shape[self.axis];
  
  @property
  def wobble(self):
    """The wobblyness of this source.
    
    Returns
    -------
    wobble : array of ints
      The deviations from the source position along the wobble axes.
    """
    return self._wobble;  
  
  @property
  def wdim(self):
    return self._source.ndim - 1;

  
  @wobble.setter
  def wobble(self, wobble):
    if wobble.shape[0] != self.height or wobble.ndim != self.wdim:
      raise ValueError('Number of wobbles %d is not equal the number of planes = %d along the wobble axis %d!' % (wobble.shape[0], self.height, self.axis));
    self._wobble = np.array(wobble, dtype = int);
  
  def wobble_from_positions(self, positions):
    start = self.coordinate;
    stop = start + self.height;
    self._wobble[:] = positions[start:stop];
    #set status
    finite = np.all(np.isfinite(positions[start:stop]), axis=1);
    non_finite = np.logical_not(finite); 
    self._status[non_finite] = self.INVALID;  
    #self._status[finite] = self.VALID;                   
  
  
  @property
  def status(self):
    """The status of each slice of this source.
    
    Returns
    -------
    status : array of ints
      The status for each position along the wobble axes.
    """
    return self._status;    
  
  @property
  def valids(self):
    return self._status >= self.VALID;
  
  
  ### Geometry
  
  @property
  def lower_wobbly(self):
    """The lower corner of the wobbly source.
    
    Returns
    -------
    lower : tuple of int
      The coordinates of the lower corner of the source.
    """
    wobble_min = np.min(self._wobble, axis=0);
    return self._wobble_to_position(wobble_min, self.coordinate);
  
  
  @property
  def upper_wobbly(self):
    """The upper corner of the source.
    
    Returns
    -------
    upper : tuple of int
      The coordinates of the upper corner of the source.
    """
    wobble_max = np.max(self._wobble, axis=0);
    position = self._wobble_to_position(wobble_max, self.coordinate);
    shape = self.source.shape;                                
    return tuple(p + s for p,s in zip(position, shape));                                      
  
  
  @property
  def positions(self):
    """The positions of the lower corners of all slices along the wobble axis.
    
    Returns
    -------
    positions : array
      The coordinates of the lower corner of the slices along the wobble axis.
    """
    wobble = self.wobble;
    axis = self.axis;
    coordinate = self.coordinate;
    positions = np.concatenate([wobble[:,:axis],
                                np.arange(len(wobble))[:,np.newaxis] + coordinate, 
                                wobble[:,axis:]], axis = 1)
    return positions;
  
  
  def coordinate_to_local(self, coordinate):
    """Converts a wobble axis coordinate to the a local coordinate wrt to the sources origin.
    
    Arguments
    ---------
    coordinate : int
      The non-local coordinate.
    
    Returns
    -------
    loacl_coordinate : int
      The local coordinate within this source.
    """
    position = self.coordinate;
    shape = self.height;
    if coordinate < position or coordinate >= position + shape:
      raise RuntimeError('Coordinate %d out of range (%d,%d)!' % (coordinate, position, position + shape))
    return coordinate - position;
  
    
  def coordinate_from_local(self, local_coordinate):
    """Converts a local wobble axis coordiante to the non-local coordinate.
    
    Arguments
    ---------
    local_coordinate : int
      The local coordiante within the source.
    
    Returns
    -------
    coordiante : int
      The non-local coordiante.
    """
    position = self.coordainte;
    shape = self.height;
    if local_coordinate < 0 or local_coordinate >= shape:
      raise RuntimeError('Coordinate %d out of range (%d,%d)!' % (local_coordinate, position, position + shape))
    return local_coordinate + position;

  
  def position_at_coordinate(self, coordinate):
    """Returns the wobbly position of the source at the specified coordinate along the wobble axis.
    
    Arguments
    ---------
    coordinate : int
      The coordinate along the wobble axis.
    
    Returns
    -------
    position : tuple of int.
      The non-local position of the specified coordainte slice.
    """ 
    local_coordinate = self.coordinate_to_local(coordinate);                                          
    wobble = self._wobble[local_coordinate];
    return self._wobble_to_position(wobble, coordinate); 
  
 
  def wobble_at_coordinate(self, coordinate):
    """Returns the wobbly position of the source at the specified coordinate along the wobble axis.
    
    Arguments
    ---------
    coordinate : int
      The coordinate along the wobble axis.
    
    Returns
    -------
    position : tuple of int.
      The non-local position of the specified coordainte slice.
    """ 
    local_coordinate = self.coordinate_to_local(coordinate);   
    return self._wobble[local_coordinate];
  
  #status  
  
  def status_at_coordinate(self, coordinate):
    local_coordinate = self.coordinate_to_local(coordinate);    
    return self._status[local_coordinate];

  def set_status_at_coordinate(self, coordinate, status):
    local_coordinate = self.coordinate_to_local(coordinate);    
    self._status[local_coordinate] = status;
  
  def is_valid(self, coordinate):
    return 0 <= coordinate - self.coordinate < self.height and self.status_at_coordinate(coordinate) >= self.VALID;
  
  def set_invalid(self, coordinate):
    if 0 <= coordinate - self.coordinate < self.height:
      self.set_status_at_coordinate(coordinate, self.INVALID);
  
  def set_isolated(self, coordinate):
    if 0 <= coordinate - self.coordinate < self.height:
      self.set_status_at_coordinate(coordinate, self.ISOLATED);
  
  
  def fix_isolated(self, exclude_borders = False):
    """Fix the positons of isolated slices."""
    status = self.status;
    wobble = self.wobble;
    n_status = len(status);
    isolated = np.array(status == self.ISOLATED, dtype=int);
    isolated = np.pad(isolated, (1,1), 'constant');
    delta = np.diff(isolated);
    starts = np.where(delta > 0)[0];
    ends = np.where(delta < 0)[0];
    
    #whole stack has no isolated slices
    if len(starts) == 0:
      return  
                   
    #if whole stack is isolated
    if len(starts) == 1 and starts[0] == 0 and len(ends) == 1 and ends[0] == n_status:
      status[:] = self.ISOLATED;
      return;             
                   
    #find left and right bounds for isolated stretches
    for s,e in zip(starts, ends):
      #exclude borders
      if exclude_borders:
        if s == 0 or e == n_status:
           status[s:e] = self.ISOLATED;
           continue;
      
      #find next valid  in each direction
      if s > 0 and status[s-1] >= self.VALID:
        left = wobble[[s-1]];
      else:
        left = None;                     
      if e < n_status and status[e] >= self.VALID:                      
        right = wobble[[e]];
      else:
        right = None;
      if left is None and right is None:
        status[s:e] = self.ISOLATED;
      else:
        if left is None:
          wobble[s:e] = right;
        elif right is None:
          wobble[s:e] = left;
        else: # linearly interpolate
          wobble[s:e] = np.array(np.round((right-left) * 1.0 / (e-s+1) * np.arange(1, e-s+1)[:, np.newaxis] + left), dtype = int);
        status[s:e] = self.FIXED;
        
  
  def smooth_positions(self, smooth = dict(method = 'window', window = 'bartlett', window_length = 10)):
    positions = smooth_positions(self.positions, self.valids, smooth=smooth);                                      
    return positions;
  
  ### Helper
  
  def _wobble_to_position(self, wobble, coordinate):
    """Transform a wobble and axis coordainte to the full position.
    
    Arguments
    ---------
    wobble : tuple
      The wobble to add.
    coordainte : int
      The coordinate along the axis
    
    Returns
    -------
    position : tuple
      The poisotn with added wobble.
    """
    axis = self.axis;
    return tuple(wobble[:axis]) + (coordinate,) + tuple(wobble[axis:]);
  
  
  ### Other
  
  def array_wobbly(self):
    """Returns the array in the wobbly form with zeros at empty positions.
    
    Returns
    -------
    array : array
      The data of the array.
    """
    axis = self.axis;
    extent = self.extent;
    ndim = len(extent);
    array = np.zeros(extent, dtype=self.dtype, order=self.order);
    lower_wobble = np.min(self.wobble, axis = 0);
    
    slicing = (slice(None),) * (ndim - 1);
    for c in range(extent[axis]):
      slicing_source = slicing[:axis] + (c,) + slicing[axis:];
      data = self.source[slicing_source];
      
      shape = data.shape;
      position = self.wobble[c] - lower_wobble;
      slicing_data = tuple(slice(p,p+s) for p,s in zip(position, shape));
      slicing_data = slicing_data[:axis] + (c,) + slicing_data[axis:]; 
      array[slicing_data] = data;
    
    return array;
  
  
  def __copy__(self):
    cls = self.__class__
    new = cls.__new__(cls)
    new.__dict__.update(self.__dict__)
    new._wobble = self._wobble.copy();
    return new




class WobblyAlignment(strg.Alignment):
  
  NOSIGNAL = -5
  NOMINIMA = -4
  UNALIGNED = -3
  UNTRACED = -2
  INVALID = -1
  VALID = 0
  MEASURED = 1
  ALIGNED = 2
  FIXED = 3
  
  status_to_description = {NOSIGNAL  : 'no signal',
                           NOMINIMA  : 'no minima',
                           UNALIGNED : 'unaligned',
                           UNTRACED  : 'untraced',
                           INVALID   : 'invalid',
                           VALID     : 'valid',
                           MEASURED  : 'measured',
                           ALIGNED   : 'aligned',
                           FIXED     : 'fixed'}
  
  def __init__(self, pre = None, post = None, shifts = None, displacements = None, qualities = None, status = None, axis = 2,  shift = None, displacement = None, quality = None):
    strg.Alignment.__init__(self, pre=pre, post=post, shift=shift, displacement=displacement, quality=quality);
    
    #overlap region
    ovlp = strg.overlap(strg.Region(position = pre.position[axis:axis+1], shape = pre.shape[axis:axis+1]),
                       strg.Region(position = post.position[axis:axis+1], shape = post.shape[axis:axis+1]));
                         
    if ovlp == None:
      raise ValueError('The two sources do not overlap along the wobble axis!');
                      
    n = ovlp.shape[0]
    ndim = pre.ndim;
    
    if displacements is None:
      d = tuple(p - q for p,q,d in zip(post.position, pre.position, range(ndim)) if d != axis);
      if shifts is None:
        displacements = np.ones((n, pre.ndim-1), dtype = int) * d;
      else:
        displacements = np.array(shifts, dtype = int) + d;
                                
    self._displacements = displacements;
    
    if qualities is None:
      qualities = np.ones(n) * (-np.inf);
    self.qualities = qualities;
    
    if status is None:
      status = np.full(n, self.VALID, dtype = int);
    self.status = status;
    
    self.axis = axis;                         
  
  @property
  def displacements(self):
    return self._displacements;
    
  @displacements.setter
  def displacements(self, value):
    if len(value) != self.upper_coordinate - self.lower_coordinate:
      raise ValueError('Dimension mismatch %d != %d' % (len(value), self.upper_coordinate - self.lower_coordinate));
    self._displacements = value;
  
  @property
  def lower_coordinate(self):
    return max(self.pre.coordinate, self.post.coordinate);
  
  @property
  def upper_coordinate(self):
    return min(self.pre.coordinate + self.pre.height, self.post.coordinate + self.post.height)
  
  def coordinate_to_local(self, coordinate):
    lower, upper = self.lower_coordinate, self.upper_coordinate
    if not (lower <= coordinate < upper):
      raise ValueError('Invalid coordinate!');
    else: 
      return coordinate - lower;
  
  
  @property
  def shifts(self):
    axis = self.axis;
    displacements = self._displacements;
    pre_pos = self.pre.position;
    post_pos = self.post.position;
    pre_pos = pre_pos[:axis] + pre_pos[axis+1:];
    post_pos = post_pos[:axis] + post_pos[axis+1:];  
    shifts = displacements - post_pos + pre_pos;
    return shifts;
  
  @shifts.setter
  def shifts(self, value):
    if len(value) != self.upper_coordinate - self.lower_coordinate:
      raise ValueError('Dimension mismatch %d != %d' % (len(value), self.upper_coordinate - self.lower_coordinate));
    axis = self.axis;
    pre_pos = self.pre.position;
    post_pos = self.post.position;
    pre_pos = pre_pos[:axis] + pre_pos[axis+1:];
    post_pos = post_pos[:axis] + post_pos[axis+1:];  
    self._displacements = np.array(value) + post_pos - pre_pos;
  
                   
  def align_wobbly_axis(self, **kwargs):
    shifts, qualities = align_wobbly_axis(self.pre, self.post, axis = self.axis, **kwargs);
    self.shifts = shifts;
    self.qualities = qualities;                                        
  
  
  def displacement_at_coordinate(self, coordinate):
    return self._displacements[self.coordinate_to_local(coordinate)]
    
  def quality_at_coordinate(self, coordinate):
    return self.qualities[self.coordinate_to_local(coordinate)]
    
  def status_at_coordinate(self, coordinate):
    return self.status[self.coordinate_to_local(coordinate)]
  
  def set_status_at_coordinate(self, coordinate, status):     
    self.status[self.coordinate_to_local(coordinate)]
  

  def valids(self, min_quality = -np.inf):
    valids = self.status >= self.VALID;
    if min_quality:
      valids = np.logical_and(valids, self.qualities > min_quality);
    return valids;                             
  
  def smooth_displacements(self, min_quality = -np.inf, **kwargs):
    displacements = smooth_displacements(self.displacements, self.valids(min_quality=min_quality), **kwargs);
    #self.displacements = displacements                                         
    return displacements;

  def fix_unaligned(self):
    """Linearly interpolate between unaligned coordinates"""
    status = self.status;
    displacements = self.displacements;
    qualities = self.qualities;
    n_status = len(status);
    unaligned = np.array(status == self.UNALIGNED, dtype=int);
    unaligned = np.pad(unaligned, (1,1), 'constant');                  
    delta = np.diff(unaligned);
    starts = np.where(delta > 0)[0];
    ends = np.where(delta < 0)[0];                     
    
    #whole stack is aligned
    if len(starts) == 0:
      return            
    #whole stack is unalinged
    if len(starts) == 1 and starts[0] == 0 and len(ends) == 1 and ends[0] == n_status:
      status[:] = self.INVALID;
      return;             
                   
    #find left and right bounds for isolated stretches
    for s,e in zip(starts, ends):
      #find next valid  in each direction
      if s > 0 and status[s-1] >= self.VALID:
        left = displacements[[s-1]];
      else:
        left = None;                     
      if e < n_status and status[e] >= self.VALID:                      
        right = displacements[[e]];
      else:
        right = None;
      if left is None and right is None:
        status[s:e] = self.INVALID;
      else:
        if left is None:
          displacements[s:e] = right;
          qualities[s:e] = qualities[e];
        elif right is None:
          displacements[s:e] = left;
          qualities[s:e] = qualities[s-1];
        else: # linearly interpolate
          displacements[s:e] = np.array(np.round((right-left) * 1.0 / (e-s+1) * np.arange(1, e-s+1)[:, np.newaxis] + left), dtype = int);   
          qs = qualities[s-1];
          qe = qualities[e];
          if np.isfinite(qs) and np.isfinite(qe):                       
            qualities[s:e] = (qe - qs) / (e-s+1) * np.arange(1, e-s+1) + qs;
          elif np.isfinite(qe):
            qualities[s:e] = qe;
          else:
            qualities[s:e] = qs;
        status[s:e] = self.FIXED;
              

  
  def overlay_wobbly(self, overlap = True):
    axis = self.axis;
    shifts = self.shifts;
    n_slices = len(shifts);
    shifts = shifts[self.status >= self.VALID]
    min_shifts = np.min(np.array(shifts), axis = 0);
    max_shifts = np.max(np.array(shifts), axis = 0);
    min_shifts = tuple(min_shifts[:axis]) + (0,) + tuple(min_shifts[axis:]); 
    max_shifts = tuple(max_shifts[:axis]) + (0,) + tuple(max_shifts[axis:]);          
    ndim = len(min_shifts);
    
    if overlap:
      o1,o2 = strg._overlap_with_shifts(self.pre, self.post, max_shifts=[(m,n) for m,n in zip(min_shifts, max_shifts)]);
    
      i1 = self.pre[o1.local_slicing(self.pre)]
      i2 = self.post[o2.local_slicing(self.post)]
    
      p1 = o1.lower;
      p2 = o2.lower;
      s1 = o1.shape;
      s2 = o2.shape;
    else:
      i1 = self.pre;
      i2 = self.post;      

      p1 = self.pre.position;
      p2 = self.post.position;
      s1 = self.pre.shape;
      s2 = self.post.shape;
    
    #paddings
    pad1 = ();
    off2 = ();
    shape = ();
    for d in range(ndim):
      pad1 += ((max(0, p1[d] - (p2[d] + min_shifts[d])), max(0, p2[d] + s2[d] + max_shifts[d] - (p1[d] + s1[d]))),);
      off2 += (max(0, p2[d] + min_shifts[d] - p1[d]),);
      shape += (max(s1[d] + pad1[d][0] + pad1[d][1], off2[d] + max_shifts[d] - min_shifts[d] + s2[d]),);
    
    ovl = [np.zeros(shape, dtype=self.pre.dtype), np.zeros(shape, dtype=self.post.dtype)];
    slice_i = [slice(None)] * ndim;
    
    pad1i = pad1[:axis] + pad1[axis+1:];
    for i in range(n_slices):
      if i%100 == 0:
        print('Generating overlay slice %d/%d!' % (i,n_slices))
      
      if self.status[i] >= self.VALID:
        slice_i[axis] = i;
        shift = tuple(self.shifts[i]);
        shift = shift[:axis] + (0,) + shift[axis:];
        pad2 = tuple((o + s - m, sh - (o + s - m) - sp) for o,s,m,sh,sp in zip(off2, shift, min_shifts, shape, s2));
        pad2i = pad2[:axis] + pad2[axis+1:];
        ovl[0][slice_i] = np.pad(i1[slice_i], pad1i, 'constant')
        ovl[1][slice_i] = np.pad(i2[slice_i], pad2i, 'constant')
  
    return ovl;  
  
  def plot_overlay_wobbly(self):
    return strg.p3d.plot([self.overlay_wobbly()]);
   
  def overlay_mip_wobbly(self, overlap = True, mip_axis = None, percentile = 98, normalize = True):
    ovl_mip = self.overlay_wobbly(overlap=overlap);
    
    #max project
    if mip_axis is None:
      mip_axis = strg._mip_axis(self.pre, self.post);
    
    ovl_mip = [np.max(o, axis=mip_axis) for o in ovl_mip];
    for o in ovl_mip:
      p = np.percentile(o, percentile);
      o[o>p]=p;
    
    import ClearMap.Visualization.Color as col
    colors = [[1, 0, 1], [0, 1, 0]];
    colors = [col.color(c, alpha = False, as_int = True) for c in colors];
    
    image = np.zeros(ovl_mip[0].shape + (3,));
    for o,c in zip(ovl_mip,colors):
      image += np.multiply.outer(o /o.max(),c);
     
    if normalize:
      for c in range(3):
        image[:,:,c] /= image[:,:,c].max(); 
      
    return image;
    
  def plot_mip_wobbly(self, overlap = True, mip_axis = None, percentile = 98):
    image = self.overlay_mip_wobbly(overlap=overlap, mip_axis=mip_axis, percentile=percentile);   
    import matplotlib.pyplot as plt
    plt.imshow(np.transpose(image, [1,0,2])[:,:,:], origin = 'lower');
    plt.tight_layout()


class WobblyLayout(strg.TiledLayout):
  """Layout to handle stitching of wobbly sources."""
  
  def __init__(self, sources = None, expression = None, tile_axes = None, tile_shape = None, tile_positions = None, positions = None, overlaps = None, alignments = None, axis = 2, position = None, shape = None, dtype = None, order = None):
    """WobblyStackLayout constructor.
      
    Arguments
    ---------
    expression : str
      Regular expression of source names.
    tile_axes : tuple of strings
      The names and ordering of the grid axes of the named groups in the regular expression. If None use the names and order as they appear in expression.
    tile_shape : tuple of ints or None
      Shape of the grid. If None determine automatically.
    tile_positions : list of tuple of ints or None
      List of grid positions to consider, if None use all available.
    positions : list of tuples of ints
      The positions of the individual sources, if None use overlaps to position sources.
    overlaps : tuple of ints or None
      Overlaps of the individual sources in each grid dimension. If None assume overlap is zero.
    shape : tuple of int or None
      The fixed shape of this Layout, if None the minimal size to fit all sources will be used.
    position : tuple of int or None
      The fixed position of this layout, if None the lower corner to fit all sources will be used.
    dtype: dtype or None
      The data type to use for this layout, if None use the dtype of the first source.
    axis : int
      The wobbly axis of the sources.
    """
    # initialize classes
    strg.TiledLayout.__init__(self, sources = sources, expression = expression, tile_axes = tile_axes, tile_shape = tile_shape, tile_positions = tile_positions, positions = positions, overlaps = overlaps, alignments = alignments, position = position, shape = shape, dtype = dtype, order = order);
    
    # convert sources to WobblySources
    sources = self.sources;
    self.sources = [WobblySource(source = s, axis = axis) for s in sources];

    alignments = [];     
    sources_to_wobbly_sources = {s : w for s,w in zip(sources, self.sources)};          
    for a in self.alignments:
      pre = sources_to_wobbly_sources[a.pre];
      post = sources_to_wobbly_sources[a.post];
      displacement = a.displacement;
      quality = a.quality;
      alignments.append(WobblyAlignment(pre=pre, post=post, axis=axis, displacement=displacement, quality=quality));
    self.alignments = alignments;
    
    self.axis = int(axis); 
  
  
  @property    
  def lower_wobbly(self):
    """Calculates the lower position of the entire layout.
    
    Returns
    -------
    lower : tuple of ints
      The lower position of the full layout.
    """
    return tuple(np.min([s.lower_wobbly for s in self.sources], axis = 0));
  
  
  @property   
  def upper_wobbly(self):
    """Calculates the upper position of the entire layout.
    
    Returns
    -------
    upper : tuple of ints
      The upper position of the full layout.
    """
    return tuple(np.max([s.upper_wobbly for s in self.sources], axis = 0));
  
  @property
  def origin_wobbly(self):
    return tuple(min(p,0) for p in self.lower_wobbly);
  
  @property
  def shape_wobbly(self):
    return tuple(u - o for u,o in zip(self.upper_wobbly, self.origin_wobbly))
  
  
  def set_positions(self, positions):
    """Set the positions of all wobbly slices and sources."""
    for s,p in zip(self.sources, positions):
      s.wobble_from_positions(p);
  
                             
  def slice_along_axis_wobbly(self, coordinate):
    """Returns a layout corresponding to a slice along the wobble axis in this layout.
    
    Arguments
    ---------
    coordinate : int
      The coordinate at which to take the slice.
    axis : int
      The axis to take the slice in.
    
    Returns
    -------
    layout : Layout class
      The sliced layout.
      
    Note
    ----
    The underlying sources are converted to virtual for parallel stitching.
    """
    axis = self.axis;
    ndim = self.ndim;
    
    #filter sources in slice
    sources = [source for source in self.sources if source.is_valid(coordinate)];
           
    #slice sources        
    sliced_sources = [];
    for source in sources:
      position = source.wobble_at_coordinate(coordinate);
      slicing = (slice(None),) * axis + (coordinate - source.coordinate,) + (slice(None),) * (ndim-1-axis);                           
      sliced_sources.append(strg.Source(source = slc.Slice(source=source.source.as_virtual(), slicing=slicing), position=position, tile_position=source.tile_position));
       
    if self._shape is not None:
      shape = self._shape[:axis] + self._shape[axis+1];
    else:
      shape = None;
    
    if self._position is not None:
      position = self._position[:axis] + self._position[axis+1];
    else:
      position = None;
    
    return strg.Layout(sources = sliced_sources, shape = shape, position = position, dtype = self._dtype, order = self._order);
  
  
  def layouts_along_axis_wobbly(self, coordinates = None):
    """Returns a list of Layouts representing the placed wobbly sources in each wobbly-axis slice of this layout.
    
    Arguments
    ---------
    coordinates : list of ints, all, or None
      The positions of the slices along the wobble axis. If all or None take all possible slices.
    
    Returns
    -------
    slices : list of SlicedLayout classes
      The layouts in each wobble-axis-plane.
      
    Note
    ----
    The slices layouts can be used for stitching of the wobbly stacks.
    """
    #create stitching planes
    if coordinates is all or coordinates is None:
      coordinates = range(self.lower[self.axis], self.upper[self.axis])
    
    return [self.slice_along_axis_wobbly(c) for c in coordinates];
 

  def plot_wobble(self):
    import matplotlib.pyplot as plt
    fig = plt.gcf();
    axs = [plt.subplot(1,2,i+1) for i in range(2)];            
    for i,a in enumerate(self.alignments):
      invalid = a.status < a.VALID
      d = np.array(a.displacements, dtype = float)
      d[invalid] = np.nan 
      #print invalid.sum()     
      x = np.arange(a.lower_coordinate, a.upper_coordinate)
      plt.subplot(1,2,1);    
      _ = plt.plot(x, d[:,0], label = '%d: %r-%r' % (i, a.pre.identifier, a.post.identifier)) #analysis:ignore
    
      plt.subplot(1,2,2, sharex = axs[0]);            
      _ = plt.plot(x, d[:,1], label = '%d: %r-%r' % (i, a.pre.identifier, a.post.identifier)) #analysis:ignore
     
    def on_pick(event):
      for ax in axs:
        for curve in ax.get_lines():
          if curve.contains(event)[0]:
            print(curve.get_label())
    
    fig.canvas.mpl_connect('motion_notify_event', on_pick)           
    plt.show()


  def alignment_info(self, tile_position, coordinate, plot = True, use_displacements = True, **kwargs):
    """Gathers all alignment info for a slice of a certain tile."""
    
    #get status
    s = self.source_from_tile_position(tile_position);
    status = s.status_at_coordinate(coordinate);

    #get all connecting alignemtns and thier status
    a_status = [];
    alignments = self.alignments_from_tile_position(tile_position);
    for a in alignments:
      a_status.append((a.pre.identifier, a.post.identifier, a.status_at_coordinate(coordinate)));

    print('Source status: %r' % WobblySource.status_to_description[status]);
    for a in a_status:         
      print('Alignment status: %r->%r: %r' % (a[0], a[1], WobblyAlignment.status_to_description[a[2]]))
    #return status, a_status    

    #plot overlay to all neighbours
    if plot:
      axis = self.axis;
      ndim = self.ndim;
      sources = [s];
      for a in alignments:
         if a.pre not in sources:
           sources.append(a.pre);
         if a.post not in sources:
           sources.append(a.post);
     
      #slice sources        
      sliced_sources = [];
      for source in sources:
        if use_displacements:
          if source == s:
            position = tuple(0 for i in range(ndim-1));
          else:
            for a in alignments:
              if source == a.pre:
                position = tuple(-p for p in a.displacement);                        
                break;
              if source == a.post:
                position = a.displacement;
                break;
            position = position[:axis] + position[axis+1:];
        else:
          position = source.wobble_at_coordinate(coordinate); 
                                              
        slicing = (slice(None),) * axis + (coordinate - source.coordinate,) + (slice(None),) * (ndim-1-axis);                           
        sliced_sources.append(strg.Source(source = slc.Slice(source=source.source.as_virtual(), slicing=slicing), position=position, tile_position=source.tile_position));
         
      sliced_layout = strg.Layout(sources = sliced_sources, shape = None, position = None, dtype = self.dtype, order = self.order);
      strg.plot_layout(sliced_layout, **kwargs)
      
    return sliced_layout                                     

    


###############################################################################
###  Alignment
###############################################################################


class Verbose(object):
  flags = {
    'save'   : 0b010,
    'figure' : 0b100
  }
  
  def __init__(self, verbose = True, save = None, directory = None):
    if isinstance(verbose, Verbose):
      self.verbose  = verbose.verbose;
      self.save = verbose.save;
      self.directory = verbose.directory;
    else:
      self.verbose = verbose;
      self.save = save;
      self.directory = directory;
    
  def has_flag(self, flag):
    if flag is None:
      if isinstance(self.verbose, bool):
        return self.verbose;
      else:
        return self.verbose > 0;
    if isinstance(self.verbose, bool):
      return False;
    else:
      return self.verbose & self.flags[flag] > 0;

  def copy(self):
    new = type(self)();
    new.__dict__.update(self.__dict__);
    return new;

  def __eq__(self, other):
    return self.verbose == other;
    
  def full_filename(self, filename):
    if self.directory is None:
      return filename;
    else:
      return io.join(self.directory, filename);
      
  def create_directory(self, prefix = None):
    if self.directory is None:
      import datetime
      directory = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S/');
      if prefix is not None:
        directory = '%s_%s' % (prefix, directory);
      self.diretory = directory;
    if not io.is_directory(self.directory):
      io.create_directory(self.directory);
    return self.directory;


def verbose_has_flag(verbose, flag):
  return Verbose(verbose).has_flag(flag);

    
#TODO: use global plane wise coordinates if subsampling !
def align_layout(layout, axis_range = None, max_shifts = 10, axis_mip = None,
                 validate = None, prepare = 'normalization',
                 validate_slice = None, prepare_slice = None,
                 find_shifts = 'minimization',
                 verbose = False, processes = None):
  
  axis = layout.axis;
  alignments = layout.alignments;
  
  if verbose:
    timer = tmr.Timer();
    print('Alignment: aligning %d pairs of wobbly sources.' % (len(alignments)));
    verbose = Verbose(verbose);
    if verbose.has_flag('save'):
       verbose.create_directory(prefix='WobblyAlginment');
  
  _align = ft.partial(align_wobbly_axis, axis=axis, axis_range=axis_range, axis_mip=axis_mip, max_shifts=max_shifts,
                                         prepare=prepare, validate=validate, 
                                         prepare_slice=prepare_slice, validate_slice=validate_slice, 
                                         find_shifts=find_shifts,
                                         verbose=verbose);
  
  if not isinstance(processes, int) and processes != 'serial':
    processes = mp.cpu_count();
  
  if processes == 'serial':
    results = [_align(a.pre, a.post) for a in alignments];
  else:
    layout.sources_as_virtual();
    with concurrent.futures.ProcessPoolExecutor(processes) as executor:
      results = executor.map(_align, [a.pre for a in alignments], [a.post for a in alignments]);
    results = list(results);                 
  
  for a,r in zip(layout.alignments, results):
    a.shifts = r[0];
    a.qualities = r[1];   
    a.status = r[2];                
  
  if verbose:
    timer.print_elapsed_time('Alignment: aligning %d pairs of wobbly sources' % (len(alignments)));


@ptb.parallel_traceback
def align_wobbly_axis(source1, source2, axis = 2, axis_range = None, max_shifts = 10, axis_mip = None,
                      validate = None, prepare = 'normalization',
                      validate_slice = None, prepare_slice = None,
                      find_shifts = 'minimization',
                      with_errors = False, with_overlaps = False, verbose = True):
  """Create shifts along the wobble axis, estimate smooth shifts and mark invalid slices, accounts for jumps in minima using multiple minima."""                      
  
  if verbose:
    timer = tmr.Timer();
    print('Alignment: wobbly alignment %r->%r along axis %d' % (source1.identifier, source2.identifier, axis));
  
  #prepare methods dicts
  validate, prepare, validate_slice, prepare_slice, find_shifts = \
    [dict(method=m) if isinstance(m, str) else m for m in (validate, prepare, validate_slice, prepare_slice, find_shifts)];
  
  if axis_mip:
    if not isinstance(axis_mip, tuple):
      axis_mip = (axis_mip, axis_mip);
    
  #overlap etc
  ndim = source1.ndim;
  p1 = source1.position;
  p2 = source2.position;
  s1 = source1.shape;
  s2 = source2.shape;
  
  p1a = p1[axis]; 
  p2a = p2[axis]; 
  
  start = max(p1a, p2a);
  stop = min(p1a + s1[axis], p2a + s2[axis]);
  if start > stop:
    raise ValueError('The sources do not overlap along axis %d!' % axis);
  n_slices = stop - start;              
  #print n_slices, start, stop              
  
  #sampling
  if not isinstance(axis_range, tuple):
    axis_range = (axis_range,);
  if len(axis_range) < 3:
    axis_range += (None,) * (3-len(axis_range));            
  a_start, a_stop, a_step = axis_range if axis_range else (None,None,None)
  a_start = start if a_start is None else a_start;
  a_stop  = stop  if a_stop  is None else a_stop;
  a_step  = 1     if a_step  is None else a_step;
  a_start = max(start, a_start);
  a_stop = min(stop, a_stop);               
  #print a_start, a_stop, a_step, start, stop
  
  #max shifts formatting  
  max_shifts = strg._format_max_shifts(max_shifts, ndim);
  max_shifts = max_shifts[:axis] + max_shifts[axis+1:];                                                   
                    
  #slices for fft
  sl1 = strg.Region(position = p1[:axis] + p1[axis+1:], shape = s1[:axis] + s1[axis+1:]);
  sl2 = strg.Region(position = p2[:axis] + p2[axis+1:], shape = s2[:axis] + s2[axis+1:]);                  
  slice1,slice2, pad1,pad2, slice_no_pad1,slice_no_pad2, shift_min,shift_max, fft_roi = strg._slicing_and_padding_for_fft(sl1, sl2, max_shifts);
  #print slice1,slice2, pad1,pad2, slice_no_pad1,slice_no_pad2, shift_min,shift_max, fft_roi
  #sdim = len(shift_min);
                                                                                                                                                            
  #full slicings                                                                                                 
  slice1_full = slice1[:axis] + (slice(a_start - p1a, a_stop - p1a),) + slice1[axis:];
  slice2_full = slice2[:axis] + (slice(a_start - p2a, a_stop - p2a),) + slice2[axis:];
  #print(slice1_full, slice2_full)                     
  #pad1_full = pad1[:axis] + [(0,0)] + pad1[axis:];
  #pad2_full = pad2[:axis] + [(0,0)] + pad2[axis:];
                                                                                                                                                           
  i1 = np.array(source1[slice1_full], dtype=float);
  i2 = np.array(source2[slice2_full], dtype=float);                
  #print i1.shape, i2.shape       
  
  #initialize the error and status results
  status = WobblyAlignment.INVALID * np.ones(n_slices, dtype=int);
  error_shape = (n_slices,) + tuple(-s.start if s.start is not None else s.stop for s in fft_roi);                                    
  errors = np.zeros(error_shape);                
                   
  #validate entire stacks 
  if validate:
    valid = _validate(i1, **validate);             
    if verbose and not valid:
      print('Alignment: Source %r is not valid!' % (source1.identifier,)); 
    if valid:
      valid = _validate(i2, **validate);
      if verbose and not valid:
        print('Alignment: Source %r is not valid!' % (source1.identifier,));
    if not valid:
      status[:] = WobblyAlignment.NOSIGNAL; 
      results = _shifts_qualities_status(errors, status, correct_shift=shift_min, **find_shifts);
      if with_errors:
        results += (errors,);
      if with_overlaps:
        results += ((i1,i2),);
      return results;                            

  #keep original copy for validation               
  if validate_slice:
    i1raw = i1.copy();
    i2raw = i2.copy();

  #prepare
  if prepare:
    i1 = _prepare(i1, **prepare);
    i2 = _prepare(i2, **prepare);
  
  #weights
  shape1 = i1.shape[:axis] + i1.shape[axis+1:]
  w1 = np.pad(np.zeros(shape1), pad1, 'constant');          
  w1[slice_no_pad1] = 1;
  w1fft = np.fft.fftn(w1);
  
  w2 = np.pad(np.zeros(shape1), pad1, 'constant');             
  w2[slice_no_pad2] = 1;
  w2fft = np.fft.fftn(w2);
   
  #norm                    
  nrm  = np.fft.ifftn(w1fft * np.conj(w2fft));                     
  nrm  = np.abs(nrm[fft_roi]);
  eps =  2.2204e-16;
  nrm[nrm < eps] = eps;             
    
  #align slices              
  for i, a in enumerate(range(start, stop)):
    if verbose and i % 100 == 0:
      print('Alignment: Wobbly alignment %r->%r along axis %d: slice %d / %d' % (source1.identifier, source2.identifier, axis, i, a_stop-a_start));
    
    if a < a_start or a >= a_stop or (a-a_start) % a_step != 0:
      status[i] = WobblyAlignment.UNALIGNED;
      continue;
    
    if axis_mip:
      mip_start = max(0, a - a_start - axis_mip[0]);
      mip_end = max(0, a - a_start + axis_mip[1]);
      slice1_a = (slice(None),) * axis + (slice(mip_start,mip_end),) + (slice(None),) * (ndim-1-axis)
      slice2_a = (slice(None),) * axis + (slice(mip_start,mip_end),) + (slice(None),) * (ndim-1-axis)
      i1a = np.max(i1[slice1_a], axis=axis);
      i2a = np.max(i2[slice2_a], axis=axis);
    else:
      slice1_a = (slice(None),) * axis + (a - a_start,) + (slice(None),) * (ndim-1-axis)
      slice2_a = (slice(None),) * axis + (a - a_start,) + (slice(None),) * (ndim-1-axis)
   
      i1a = i1[slice1_a];
      i2a = i2[slice2_a];
    
    if validate_slice:
      i1rawa = i1raw[slice1_a];
      valid = _validate(i1rawa, **validate_slice);
      if verbose and not valid:
        print('Alignment: Slice %d with coordainte %d in source %r is not valid!' % (a - a_start, a, source1.identifier)); 
      if valid:
        i2rawa = i2raw[slice2_a];
        valid = _validate(i2rawa, **validate_slice);
        if verbose and not valid:
          print('Alignment: Slice %d with coordainte %d in source %r is not valid!' % (a - a_start, a, source2.identifier)); 
      if not valid:
        status[i] = WobblyAlignment.NOSIGNAL;
        continue;                        
    
    if prepare_slice:
      i1a = _prepare(i1a, **prepare_slice);
      i2a = _prepare(i2a, **prepare_slice);
    
    i1a = np.pad(i1a, pad1, 'constant');
    i2a = np.pad(i2a, pad2, 'constant');                  
    
    # fft
    i1fft = np.fft.fftn(i1a); 
    i2fft = np.fft.fftn(i2a);
    s1fft = np.fft.fftn(i1a * i1a);  
    s2fft = np.fft.fftn(i2a * i2a);
    wssd = w1fft * np.conj(s2fft) + s1fft * np.conj(w2fft) - 2 * i1fft * np.conj(i2fft);
    wssd = np.fft.ifftn(wssd);        
    wssd = wssd[fft_roi];
    
    # normalize
    wssd = np.abs(wssd);           
    wssd = wssd / nrm;
              
    # save least square errors             
    errors[i] = wssd;
    status[i] = WobblyAlignment.MEASURED
    
  if verbose:
    timer.print_elapsed_time('Alignment: Wobbly slice alignment %r->%r along axis %d done' % (source1.identifier, source2.identifier, axis));
  
    if verbose_has_flag(verbose, 'save'):
      filename = verbose.full_filename('errors_%r_%r.npy' % (source1.identifier, source2.identifier));
      np.save(filename, errors);
      verbose.save = '%r_%r' % (source1.identifier, source2.identifier)
                
  results = _shifts_qualities_status(errors, status, add_shift=shift_min, verbose=verbose, **find_shifts);
  
  if verbose:
    timer.print_elapsed_time('Alignment: Wobbly alignment %r->%r along axis %d done' % (source1.identifier, source2.identifier, axis));
                              
  if with_errors:
    results += (errors,);
  if with_overlaps:
    results += ((i1raw,i2raw),);
  return results;  



def prepare_normalization(array, clip = None, normalize = True):
  # clip images for better alignment performance  
  if clip is not None:                 
    #clip
    if isinstance(clip, (list, tuple)):
      if clip[0] is not None:
        array[array < clip[0]] = clip[0];
      if clip[1] is not None:
        array[array > clip[1]] = clip[1];
    else:
      array[array > clip] = clip;
  
  #normalize the full image
  if normalize:
    array -= np.mean(array);
    array *= 1.0/np.sqrt(np.sum(array*array));

  return array;

def _prepare(array, method='normalization', **kwargs):
  if method == 'normalization':
    return prepare_normalization(array, **kwargs);
  else:
    raise ValueError('Preparation method %r not valid!' % method);
                

def validate_foreground(array, valid_range = (800,None), size = None, fraction = None, verbose = True):
  #check if overlaps are background
  if valid_range is None:
    return True;
  
  low, high = valid_range;
  if low is None and high is None:
    return True;
  
  if low is not None and high is not None:
    foreground = np.sum(np.logical_and(low < array, array < high));
  elif low is not None:
    foreground = np.sum(low <= array);       
  else:
    foreground = np.sum(array <= high);
  
  if fraction is not None:
    size = fraction * array.size;
    
  if size is None:
    valid = foreground > 0;
    if verbose and not valid:
      print('Alignment: All %d pixels are background in range %r!' % (array.size, valid_range));
  else:
    valid = foreground >= size;
    if verbose and not valid:
      print('Alignment: Not enough foreground pixels %d < %d in range %r!' % (foreground, size, valid_range));
    
  return valid;


def _validate(array, method='foreground', **kwargs):
  if method == 'foreground':
    return validate_foreground(array, **kwargs);
  else:
    raise ValueError('Validation method %r not valid!' % method); 



import skimage.feature as skif

def detect_local_minima(error, distance=1):
  minima = skif.peak_local_max(-error, min_distance=distance, exclude_border=True);
                              
  if len(minima) > 0:                              
    shifts = [tuple(m) for m in minima];
    qualities = [error[s] for s in shifts];
  else:
    shifts = [(0,) * error.ndim];
    qualities = [-np.inf];
  
  return shifts, qualities                   


def _detect_minima(array, method='local_minima', **kwargs):
  if method == 'local_minima':
    return detect_local_minima(array, **kwargs);
  else:
    raise ValueError('Method %r not valid for minima detection!' % method);
                


def shifts_from_minimization(errors, status):
  n = len(status);
  qualities = -np.inf * np.ones(n);    
  shifts = np.zeros((n,errors.ndim-1),dtype=int);
  
  # find minimal shifts
  for e,s,i in zip(errors, status, range(n)): 
    if s == WobblyAlignment.MEASURED:
      shift = np.argmin(e);
      shift = tuple(np.unravel_index(shift, e.shape)); 

      shifts[i] = shift;                                                
      qualities[i] = -(e[shift]);
      status[i] = WobblyAlignment.ALIGNED;           
            
  return shifts, qualities, status        
 


def shifts_from_tracing(errors, status, cutoff=None, new_trajectory_cost=None, minima='local_minima', verbose=False, **kwargs):
  verbose = Verbose(verbose);  
  
  #defaults
  n = len(status);
  qualities = -np.inf * np.ones(n);    
  shifts = np.zeros((n,errors.ndim-1),dtype=int);
  
  #measured entries
  measured = np.where(status == WobblyAlignment.MEASURED)[0];
  if len(measured) == 0:
    return shifts, qualities, status;
                           
  #minima detection
  mins = [_detect_minima(error, method=minima, **kwargs) for error in errors[measured]];             
  
  #invalid minima
  for i,m in zip(measured, mins):
    if len(m[1]) == 1 and not np.isfinite(m[1][0]):
      status[i] = WobblyAlignment.NOMINIMA     
      #print('no min')           
  mins = [m for m in mins if np.isfinite(m[1][0])];
          
  #valid regions
  measured = status == WobblyAlignment.MEASURED
  valids = np.logical_or(measured, status == WobblyAlignment.UNALIGNED);
  valids = np.array(valids, dtype=int);
  valids = np.asarray(np.pad(valids, (1,1), 'constant'));                     
  starts = np.where(np.diff(valids) > 0)[0];
  ends = np.where(np.diff(valids) < 0)[0];   
  #print starts, ends
  
  if len(starts) == 0:
    return shifts, qualities, status;

  if new_trajectory_cost is None:
     new_trajectory_cost = np.sqrt(np.sum(np.power(errors[0].shape, 2)));

  n_measured = 0;
  for s,e in zip(starts, ends):
    #account for subsampling
    measured_se = np.where(measured[s:e])[0];
    n_measured_se = len(measured_se);                    
    if n_measured_se == 0:
      continue;
                 
    positions = [mins[i][0] for i in range(n_measured, n_measured + n_measured_se)];
    n_measured += n_measured_se;
    
    trajectories = trk.track_positions(positions, new_trajectory_cost=new_trajectory_cost, cutoff=cutoff)
    
    if verbose.has_flag('figure'):
      import matplotlib as mpl   #analysis:ignore
      from mpl_toolkits.mplot3d import Axes3D #analysis:ignore
      import matplotlib.pyplot as plt
      fig = plt.figure(200); plt.clf();
      fig.gca(projection='3d') 
      for t in trajectories:
        plt.plot([positions[p[0]][p[1]][0] for p in t], [positions[p[0]][p[1]][1] for p in t], [p[0] for p in t])    
      plt.title('Tracked trajectories')
    
    if verbose.has_flag('save'):
      filename = verbose.full_filename('trajectories_%s_%d_%d.npy' % (verbose.save, s, e));
      np.save(filename, trajectories);
      filename = verbose.full_filename('positions_%s_%d_%d.npy' % (verbose.save, s, e));
      np.save(filename, positions);
    
    #successivley add longer trajectories 
    #TODO: could search local error landscape for best error, etc
    n_opt = 0;
    t_opt = []; 
    while n_opt < n_measured_se:
      #find longest
      lens = np.array([len(t) for t in trajectories]);             
      iopt = np.where(lens == np.max(lens))[0];                
      if len(iopt) > 1:
        q = [np.sum([errors[t[0]][tuple(positions[t[0]][t[1]])] for t in trajectories[i]]) for i in iopt];
        iopt = iopt[np.argmin(q)];
      else:
        iopt = iopt[0];
      t_opt.append(trajectories[iopt]);
      n_opt += len(t_opt[-1]);
    
      #remove non relevant trajcetories
      ts = t_opt[-1][0][0]
      te = t_opt[-1][-1][0];
      trajectories = [t for t in trajectories if (t[0][0] < ts and t[-1][0] < ts) or (t[0][0] > te and t[-1][0] > te)]
      if len(trajectories) == 0:
        break;
    
    if verbose.has_flag('figure'):
      fig = plt.figure(201); plt.clf();
      fig.gca(projection='3d') 
      for t in t_opt:
        plt.plot([positions[p[0]][p[1]][0] for p in t], [positions[p[0]][p[1]][1] for p in t], [p[0] for p in t])    
      plt.title('Optimal trajectory')
    
    if verbose.has_flag('save'):
      #print(verbose.save, (s,e));
      filename = verbose.full_filename('trajectory_opt_%s_%d_%d.npy' % (verbose.save, s, e));
      np.save(filename, t_opt);
    
    #update results
    measured_se += s;
    status[measured_se] = WobblyAlignment.UNTRACED
    for t in t_opt:
      for p in t:
        l, m = p;
        i = measured_se[l];
        shifts[i] = positions[l][m];
        qualities[i] = -errors[i][tuple(shifts[i])];
        status[i] = WobblyAlignment.ALIGNED;              

  return shifts, qualities, status;


def _shifts_qualities_status(errors, status, method='minimization', add_shift=None, **kwargs):
  """Helper to calculate shifts, qualities and status from alignment errors."""
  if method is None:
    method = 'minimization';
  
  if method == 'minimization':
    method = shifts_from_minimization
  elif method == 'tracing':
    method = shifts_from_tracing;
  else:
    raise ValueError('Method %r not a vaild for shift detection!' % method);
  
  #strg.dv.plot(errors)
  shifts, qualities, status = method(errors, status, **kwargs);
  #print shifts, qualities, status
  
  if add_shift is not None:
    shifts = [tuple(s + m for s,m in zip(shift, add_shift)) for shift in shifts];
  
  return shifts, qualities, status



def inspect_align_layout(alignment, verbose):
  """Parse the infomration saved during a align_layout.
  
  Returns
  -------
  errors : array 
    The error landscape for each slice.
  minima : array
   Coordinates of the detected minima  
  trajectories : list
    List of coordaintes of the detected trajectories.
  trajectories_optimal : list
    List of the optimal trajectories.
  """

  verbose = Verbose(verbose);  
  verbose.save = '%r_%r' % (alignment.pre.identifier, alignment.post.identifier);
  
  #error
  error_file = verbose.full_filename('errors_%s.npy' % verbose.save);
  error = np.load(error_file)
  #p3d.plot(error_file)
  
  #minima
  positions_expression = verbose.full_filename(te.Expression('positions_%s_<s>_<e>.npy' % verbose.save))
  positions_files = io.file_list(positions_expression)  

  minima = [];
  for p in positions_files:
    values = positions_expression.values(p);
    s = values['s']; e = values['e'];
    positions = np.load(p)
    pp = np.vstack([np.array([np.array(m + (z,), dtype=int) for m in  mm], dtype=int) for z,mm in zip(range(s,e),positions)]);
    minima.append(pp);
  minima = np.vstack(minima)
  
  #potential trajectories
  trajectory_expression = verbose.full_filename(te.Expression('trajectories_%s_<s>_<e>.npy' % verbose.save))
  trajectory_files = io.file_list(trajectory_expression)  
  paths = [];  
  for t in trajectory_files:
    values = trajectory_expression.values(t);
    s = values['s']; e = values['e'];
    trajectories = np.load(t)
    positions = np.load(positions_expression.string(values))    
    z_positions = np.arange(s,e);
    for trajectory in trajectories:
      paths.append(np.array([np.array(positions[p[0]][p[1]] +(z_positions[p[0]],)) for p in trajectory]));
  
  #p3d.list_line_plot_3d(paths[1])
   
  #optimal trajectory
  trajectory_expression = verbose.full_filename(te.Expression('trajectory_opt_%s_<s>_<e>.npy' % verbose.save))
  trajectory_files = io.file_list(trajectory_expression)  
  opt_paths = [];  
  for t in trajectory_files:
    values = trajectory_expression.values(t);
    s = values['s']; e = values['e'];
    trajectories = np.load(t);
    positions = np.load(positions_expression.string(values))
    z_positions = np.arange(s,e);
    for trajectory in trajectories:
      opt_paths.append(np.array([np.array(positions[p[0]][p[1]] +(z_positions[p[0]],)) for p in trajectory]));
  
  return error, minima, paths, opt_paths





###############################################################################
### Placement
###############################################################################


def place_layout(layout, min_quality = None, method = 'optimization', 
                 smooth = None, smooth_optimized = None, fix_isolated = True,                       
                 lower_to_origin = True, processes = None, verbose = False):
  """Place a layout with the WobblyAlignments."""
  
  #prepare methods dicts
  smooth, smooth_optimized = [dict(method=m) if isinstance(m, str) else m for m in (smooth, smooth_optimized)];
  
  #place tiles in each slice first
  sources = layout.sources;
  alignments = layout.alignments;
  axis = layout.axis;
  
  #TODO: fix all the upper lower etc defs to not only work with lower_to_origin layout ?
  n_slices = layout.extent[axis];                   
  n_sources = len(sources);
  if n_sources == 0 or n_slices == 0:
    return;

  if verbose:
    timer = tmr.Timer();
    print('Placement: placing positions in %d slices!' % (n_slices));
  
  #compose the slice info
  source_to_index = {s : i for i,s in enumerate(sources)};
  positions = np.array([s.position[:axis] + s.position[axis+1:] for s in sources]);
  alignment_pairs = np.array([(source_to_index[a.pre], source_to_index[a.post]) for a in alignments]);
  n_alignments = len(alignment_pairs);
  ndim = len(positions[0]);
            
  #displacmeents and qualities
  displacements = np.full((n_slices, n_alignments, ndim), np.nan);
  qualities = np.full((n_slices, n_alignments), -np.inf);  
  status = np.full((n_slices, n_alignments), WobblyAlignment.INVALID, dtype = int);                               
  for i,a in enumerate(alignments):
    # fill in undersampled gaps
    a.fix_unaligned();
    
    # smooth
    l = a.lower_coordinate;
    u = a.upper_coordinate;
    if smooth:               
      displacements[l:u,i] = a.smooth_displacements(min_quality=min_quality, **smooth);  
    else:
      displacements[l:u,i] = a.displacements;
    qualities[l:u,i] = a.qualities;                  
    status[l:u,i] = a.status;
  
  #np.save('displacements.npy', displacements);
  #np.save('qualities.npy', qualities);
  #np.save('status.npy', status);
  
  #place each slice
  _place = ft.partial(_place_slice, positions=positions, alignment_pairs=alignment_pairs, min_quality=min_quality);
  
  if not isinstance(processes, int) and processes != 'serial':
    processes = mp.cpu_count();
  
  if processes == 'serial':
    results = [_place(d,q,s) for d,q,s in zip(displacements, qualities, status)];
  else:
    with concurrent.futures.ProcessPoolExecutor(processes) as executor:
      results = executor.map(_place, displacements, qualities, status);
    results = list(results);                 
  
  positions_new = np.array([r[0] for r in results]);             
  components = [r[1] for r in results];   
  
  #np.save('positions_new.npy', positions_new.swapaxes(0,1));
  #TODO: transform status from alignments to source staus ?  

               
  if verbose:
    timer.print_elapsed_time('Placement: placing positions in %d slices done!' % (n_slices));                                 
  
  #mark and remove isolated tiles
  for s,components_slice in enumerate(components):
    for c in components_slice:
      if len(c) == 1:
        layout.sources[c[0]].set_isolated(coordinate = s); 
  components = [[c for c in components_slice if len(c) > 1] for components_slice in components];              
  
  #optimize positions
  if method == 'optimization':
    if verbose:
      print('Placement: optimizing wobbly positions!')
    positions_optimized = _optimize_slice_positions(positions_new, components, processes=processes, verbose=verbose);
  else:
    if verbose:
      print('Placement: combining wobbly positions!')      
    positions_optimized = _straighten_slice_positions(positions_new, components, layout.tile_positions);
  positions_optimized = positions_optimized.swapaxes(0,1);
  
  #np.save('positions_optimized_1.npy', positions_optimized);

  #TODO: after fixing isolated !!!! or including status !!!
  #smoooth optimized positions
  if smooth_optimized:
     for p in positions_optimized:
       valids = np.all(np.isfinite(p), axis=1); 
       #include status validation here !                       
       p[:] = smooth_positions(p, valids=valids, **smooth_optimized);
                                                    
  #zero origin
  if lower_to_origin:
    positions_optimized_valid = np.ma.masked_invalid(positions_optimized);
    min_pos = np.array(np.min(np.min(positions_optimized_valid, axis = 0), axis = 0));
    positions_optimized -= min_pos;
   
  if verbose:
    timer.print_elapsed_time('Placement: placing wobbly layout done!');
  
  #np.save('positions_optimized_2.npy', positions_optimized);
  
  layout.set_positions(positions_optimized);
  
  if fix_isolated:                      
    for source in layout.sources:
      source.fix_isolated();
  
  #return positions_optimized;

@ptb.parallel_traceback
def _place_slice(displacements, qualities, status, positions, alignment_pairs, min_quality=-np.inf):
  
  positions = positions.copy();
                           
  #filter alignments by quality
  valid = status >= WobblyAlignment.VALID;
  if min_quality:
    valid = np.logical_and(valid, qualities > min_quality);

  alignment_pairs = alignment_pairs[valid];
  displacements = displacements[valid];
  qualities = qualities[valid];
                         
  #connected components
  component_ids, component_pairs, component_displacements = _connected_components(positions, alignment_pairs, displacements);
  
  for pairs,displ in zip(component_pairs, component_displacements):
    _place_slice_component(positions, pairs, displ);                                    
                                   
  return positions, component_ids;


def _connected_components(positions, alignment_pairs, displacements):
  """Returns the connected components of the alignments."""
  n_sources = len(positions);
    
  #determine connected compoenents
  g = gt.Graph(directed = False);    
  g.add_vertex(n_sources);
  for a in alignment_pairs:
    g.add_edge(a[0], a[1]);
  connected_components, hist = gtt.label_components(g);
  connected_components = np.array(connected_components.a);                            
  n_components = len(hist);
  #print connected_components, hist, len(hist), np.max(hist)   
  
  # create components
  component_pairs = [];
  component_displacements = [];
  component_ids = [];                            
  for i in range(n_components):
    ids = np.where(connected_components == i)[0];
    pairs = [];
    displ = [];           
    for a,d in zip(alignment_pairs, displacements):
      if a[0] in ids:
        pairs.append(a);
        displ.append(d);           
     
    component_pairs.append(pairs);
    component_displacements.append(displ); 
    component_ids.append(ids);                             
                         
  return component_ids, component_pairs, component_displacements;
  
  
def _place_slice_component(positions, alignment_pairs, displacements, fixed = None):
  """Optimize positions for a connected component."""
  nalignments = len(alignment_pairs);
  if nalignments == 0:
    return positions;
  
  # construct the mappings between node ids and index 1:nimages
  pre_indices = np.unique([p[0] for p in alignment_pairs])
  post_indices = np.unique([p[1] for p in alignment_pairs]);
  node_to_index = np.unique(np.hstack([pre_indices, post_indices]));
  index_to_node = { i : n for n,i in enumerate(node_to_index)}    
  nnodes = len(node_to_index);
    
  ndim = len(positions[0]);
  n = ndim * nalignments;
  m = ndim * (nnodes - 1); # first image is assumed to be fixed at zero

  # derivative of the error gives constraints s - M x == 0
  # s are the displacements, x the centers of the images, M is derived from the error terms

  # s
  s = np.zeros(n);
  k = 0;
  for a, sh in zip(alignment_pairs, displacements):
    for d in range(ndim):
      s[k] = sh[d];
      k = k + 1;
  
  # M
  M = np.zeros((n,m));
  k = 0;
  for a in alignment_pairs:
    pre_node = index_to_node[a[0]];
    post_node = index_to_node[a[1]];                      
    for d in range(ndim):
      if pre_node > 0:
         M[k, (pre_node - 1) * ndim + d] = -1;
      if post_node > 0:
         M[k, (post_node - 1) * ndim + d] = 1;
      k = k + 1;
  
  #print s
  #print M
  #print np.linalg.pinv(M)
  
  # find the centers of the images via pseudo inverse
  positions_optimized = np.dot(np.linalg.pinv(M), s);  
  positions_optimized = np.hstack([np.zeros(ndim), positions_optimized]);
  positions_optimized = np.reshape(positions_optimized, (-1, ndim));
  positions_optimized = np.asarray(np.round(positions_optimized), dtype = int);
  
  #correct for origin and fixed source
  if fixed is not None:
    fixed_id = fixed;
  else:
    fixed_id = np.min(alignment_pairs);
  fixed_position = positions[fixed_id];
  positions_optimized = positions_optimized - positions_optimized[index_to_node[fixed_id]] + fixed_position;
  
  #update positions
  positions[node_to_index] = positions_optimized;



def _cluster_components(components):
  """Find the connected components of the cluster components"""
  c_lens = [len(c) for c in components];
  c_ids = np.cumsum(c_lens);
  c_ids = np.hstack([0, c_ids]);
  n_components = np.sum(c_lens);
  
  def is_to_c(s, i):
    return c_ids[s] + i
    
  def c_to_si(c):
    s = np.searchsorted(c_ids, c, side='right')-1;
    i = c - c_ids[s];
    return s,i
  
  g = gt.Graph(directed = False);    
  g.add_vertex(n_components);
  
  for s in range(1, len(components)):  
    for i,ci in enumerate(components[s-1]):
      for j,cj in enumerate(components[s]):
        for c in ci:
          if c in cj:
            g.add_edge(is_to_c(s-1,i),is_to_c(s,j));
            break;
  
  connected_components, hist = gtt.label_components(g);
  connected_components = np.array(connected_components.a);
  n_components = len(hist);
  components_full = [np.where(connected_components==i)[0] for i in range(n_components)];
  
  #remove isolated nodes
  components_full = [c for c in components_full if len(c) > 1] 
                     
  return components_full, is_to_c, c_to_si


def _optimize_slice_positions(positions, components, processes = None, verbose = False):
  """Helper to optimize the positions of the slices on top of each other"""
  
  #Setting:
  #refer to the slice components as 'clusters'

  #positions is a list of the tile positions in each slice
  #positions[slice, tile] is a ndim array of the tile position in slice s
  #positions of non-existent tiles are set to [npinf] * ndim
  
  #components is a list of lists indicating the clusters in each slice
  #components[slice] = [cluster1, cluster2, ...]
  #each cluster is a list of tile ids. tiles not in a slice are not listed.
  
  #Optimization:
  #minimize displacements of all sources between the slices
  n_slices = len(components);
  ndim = len(positions[0,0]);
  
  #compute connected components of the clusters
  cluster_components, si_to_c, c_to_si = _cluster_components(components);                                                                                                               
  #cluster_components is a list of lists of ints indicating the cluster ids
  #that belong to the connected compoenents of the clusters
  #cluster_components[0] = [c1, c2, ...] with cluster ids c1,c2,...
  #print cluster_components   
  n_components = len(cluster_components)
  if verbose:
    print('Placement: found %d components to optimize!' % n_components);
  
  #optimize positions for each cluster component
  for cci, cluster_component in enumerate(cluster_components):
    #Error functon:
    # E = \sum_s \sum_{i \in C_s} \sum_{j \in C_{s+1}} \sum_{k\in C_{s,i} \cup C_{s+1,j}} (x_{s,k} + s_{s,i} - (x_{s+1,k} + s_{s+1,j}))^2
    # x_{s,k} is the position of the k-th tile in the s-th slice
    # s_{s,i} is the shift of the i-th cluster C_{s,i} in slice s
    # C_s is the set of clusters in slice s
    # s0 = argmin_s(|C_s|>0), i0 = argmin(C_\bar{s}) is the first cluster
    # s_{s0,i0} = 0 is fixed as the overall shift is arbitrary otherwise.
    
    # derivative of error gives constraints x - M s == 0
    # and cluster shifts are given as the pseudo inverse: s = M^\dagger x
    
    #Notation: 
    # slice indices: t = s+1, r = s-1
    # C_{s,i} has an id c, c_to_si and si_to_c convert between s,i and c
    # The clusters in this connected component of clusters are enumerated by d
    # starting at the second cluster as the first cluster's shift is fixed
    # d_to_si, si_to_d convert between them.
    
    n_clusters = len(cluster_component);
    n_s = (n_clusters - 1); # first s == 0
    if verbose:
      print('Placement: optimizing component %d/%d with %d clusters!' % (cci, n_components, n_clusters));
                 
     
    #construct map : slice -> cluster ids
    slice_to_cluster_ids = [()] * n_slices;
    for c in cluster_component:
       s,i = c_to_si(c);                  
       slice_to_cluster_ids[s] += (i,);
    
    
    #construct generic id d to si maps
    si_to_d = {}; 
    d_to_si = {};
    for d,c in enumerate(cluster_component[1:]):
      s,i = c_to_si(c);
      si_to_d[(s,i)] = d;
      d_to_si[d] = (s,i);
    
    s0,i0 = c_to_si(cluster_component[0])     
    
    
    # construct x, M
    X = [io.sma.zeros(n_s) for d in range(ndim)];
    M = [io.sma.zeros((n_s, n_s)) for d in range(ndim)];
    for ci, c in enumerate(cluster_component[1:]): 
      #if verbose and ci % 100 == 0:
      #  print('Placement: constructing constraints %d/%d!' % (ci, n_clusters))
      
      s,i = c_to_si(c);
      C_si = components[s][i];
      d = si_to_d[(s,i)];
      #print s,i,C_si,d
      
      #if s < n_slices - 1:
      #for c2 in cluster_component:
      t = s + 1;
      if t < n_slices:
        for j in slice_to_cluster_ids[t]:
          C_tj = components[t][j];
          is_first = s0 == t and i0 == j;
          if not is_first:
            f = si_to_d[(t,j)];
            #print t,j,C_tj,is_first,f
          for k in C_si:
            if k in C_tj:
              for e in range(ndim):
                X[e][d] += positions[s,k,e] - positions[t,k,e];
                M[e][d,d] += 1;
                if not is_first:
                  M[e][d,f] -= 1;
      
      r = s - 1;
      if r >= 0:
        for j in slice_to_cluster_ids[r]:  
          C_rj = components[r][j];
          is_first = s0 == r and i0 == j;
          if not is_first:
            f = si_to_d[(r,j)];
          for k in C_si:
            if k in C_rj:
              for e in range(ndim):
                X[e][d] -= positions[r,k,e] - positions[s,k,e];
                M[e][d,d] += 1;
                if not is_first:
                  M[e][d,f] -= 1;
    
    
    if verbose:
      print('Placement: done constructing constraints for component %d/%d!' % (cci, n_components))
    # find the shifts of the clusters via pseudo inverse
    #print X
    #print M
    #print np.linalg.pinv(-M)
    
    if isinstance(processes, int):
      M = [io.sma.smm.insert(m) for m in M];
      X = [io.sma.smm.insert(x) for x in X];           
      with concurrent.futures.ProcessPoolExecutor(min(processes, ndim)) as executer:
        shifts = executer.map(_optimize_shifts, M, X);
      shifts = list(shifts);
      shifts = np.array(shifts).T;
    else:
      shifts = [np.linalg.lstsq(-M[e], X[e], rcond=None)[0] for e in range(ndim)];
      shifts = np.asarray(np.round(shifts), dtype=int).T;
    
    #update positions of the tiles
    for c in cluster_component[1:]:
      s,i = c_to_si(c);
      C_si = components[s][i];
      d = si_to_d[(s,i)];
      #print s,i,d,C_si, shifts[d]
      for k in C_si:
        positions[s,k] += shifts[d]; 
                 
    if verbose:
      print('Placement: component %d/%d optimized!' % (cci, n_components))             
  
  
  #note overall shifts between components is not touched but might be based on
  #keeping ovrall distance.
  return positions;


def _optimize_shifts(MM,XX):
  M = io.sma.smm.get(MM);
  X = io.sma.smm.get(XX);                  
  
  #ss = np.dot(np.linalg.pinv(-M), X);
  ss = np.linalg.lstsq(-M, X, rcond=None)[0];
  #ss = scipy.sparse.linalg.lsqr(-M, X)[0];

  io.sma.smm.free(MM);
  io.sma.smm.free(XX);                     
  
  return np.asarray(np.round(ss), dtype=int);


def _straighten_slice_positions(positions, components, tile_positions):
  """Straighten the center tiles in each connected cluster component"""
  
  #The cluster components always split between different tiles 
  #so we can straighten the center tile in each cluster compoenent.
  
  n_slices = len(components);
  
  #compute connected components of the clusters
  cluster_components, si_to_c, c_to_si = _cluster_components(components);

  for cluster_component in cluster_components:

    slice_ids = [];    
    for c in cluster_component:
      s,i = c_to_si(c);
      if s not in slice_ids:
        slice_ids.append(s);
      
      C_si = components[s][i];
      tile_ids = [];
      tile_pos = [];
      for k in C_si:
        if k not in tile_ids:
          tile_pos.append(tile_positions[k]);
          tile_ids.append(k);
    
    center_tile_position = strg._center_tile(tile_pos);
    for i,t in enumerate(tile_pos):
      if center_tile_position == t:
        center_tile = i;
        break;
    center_slice = slice_ids[(len(slice_ids)-1)//2]; 
    center_position = positions[center_slice, center_tile];
    
    for k in tile_ids:
      for s in range(n_slices):
        positions[s,k] += center_position - positions[s,center_tile]
  
  return positions;

    
def smooth_binary(x, width=1):
  """Remove displacements smaller than a certain width."""
  width = width + 1; #width -> range
  if len(x) < width:
    width = len(x);
  
  x = x.copy();
  n = len(x);
  
  #smooth open border
  x[:width] = np.median(x[:width]);
  x[-width:] = np.median(x[-width:]);       
         
  for w in range(width,1,-1):         
    starts = range(n-w);
    ends = range(w,n);
    for s,e in zip(starts, ends):
      if x[s] == x[e]:
        x[s:e] = x[s];
            
  return x;
  

def smooth_window(x, window_length = 10, window = 'bartlett', binary = None):
  """Convolutional smoothing filter"""
  if window_length > len(x):
    window_length = len(x);
  
  if window:
    windows = ['flat', 'hanning', 'hamming', 'bartlett', 'blackman'];
    if not window in windows:
      raise ValueError('Window not in %r!' % windows);
    if window == 'flat': #moving average
      w = np.ones(window_length)
    else:
      w = getattr(np, window)(window_length)
    w /= w.sum();
              
    x = np.pad(x, (window_length, window_length), 'edge')
    y = np.convolve(w, x, mode='same')[window_length:-window_length];
  
    y =  np.array(np.round(y), dtype = int);
  else:
    y = x.copy();                 
  
  if binary:
    y = smooth_binary(y, width=binary);
  
  return y;                     


def smooth_positions(positions, valids, method = 'window', **kwargs):
  """Smooth positions in valid regions."""
  return smooth_displacements(positions, valids=valids, method=method, **kwargs);


def smooth_displacements(displacements, valids, method = 'window', **kwargs):   
  """Smooth displacements in valid regions."""
  displacements_smooth = displacements.copy();
  if method is None:
    return displacements_smooth;
  
  #find valid slices  
  valids = np.asarray(np.pad(valids, (1,1), 'constant'), dtype = int);
  starts = np.where(np.diff(valids) > 0)[0];
  ends = np.where(np.diff(valids) < 0)[0];   
                   
  if method == 'window':
    smooth = ft.partial(smooth_window, **kwargs)
  else:
    raise ValueError('Smoothing method %r not valid!' % method);
  
  #smooth each interval
  ndim = displacements.ndim;
  for s,e in zip(starts, ends):
    for d in range(ndim):
      smooth_displacements = smooth(displacements[s:e,d]);
      displacements_smooth[s:e,d] = smooth_displacements;
                   
  return displacements_smooth;



def fix_unaligned(displacements, status, qualities):
  """Linearly interpolate between unaligned coordinates"""
  n_status = len(status);
  unaligned = np.array(status == WobblyAlignment.UNALIGNED, dtype=int);
  unaligned = np.pad(unaligned, (1,1), 'constant');                  
  delta = np.diff(unaligned);
  starts = np.where(delta > 0)[0];
  ends = np.where(delta < 0)[0];                     
  
  #whole stack is aligned
  if len(starts) == 0:
    return displacements, status           
  
  #whole stack is unalinged
  if len(starts) == 1 and starts[0] == 0 and len(ends) == 1 and ends[0] == n_status:
    status[:] = WobblyAlignment.INVALID;
    return displacements, status           
                 
  #find left and right bounds for isolated stretches
  for s,e in zip(starts, ends):
    #find next valid  in each direction
    if s > 0 and status[s-1] >= WobblyAlignment.VALID:
      left = displacements[[s-1]];
    else:
      left = None;                     
    if e < n_status and status[e] >= WobblyAlignment.VALID:                      
      right = displacements[[e]];
    else:
      right = None;
    if left is None and right is None:
      status[s:e] = WobblyAlignment.INVALID;
    else:
      if left is None:
        displacements[s:e] = right;
        qualities[s:e] = qualities[e];
      elif right is None:
        displacements[s:e] = left;
        qualities[s:e] = qualities[s-1];
      else: # linearly interpolate
        displacements[s:e] = np.array(np.round((right-left) * 1.0 / (e-s+1) * np.arange(1, e-s+1)[:, np.newaxis] + left), dtype = int);   
        qs = qualities[s-1];
        qe = qualities[e];
        if np.isfinite(qs) and np.isfinite(qe):                       
          qualities[s:e] = (qe - qs) / (e-s+1) * np.arange(1, e-s+1) + qs;
        elif np.isfinite(qe):
          qualities[s:e] = qe;
        else:
          qualities[s:e] = qs;
      status[s:e] = WobblyAlignment.FIXED;
  
  return displacements, status

#TODO: fix this clean up placement in total including status info
def fix_isolated(self, exclude_borders=False):
  """Fix the positons of isolated slices."""
  status = self.status;
  wobble = self.wobble;
  n_status = len(status);
  isolated = np.array(status == self.ISOLATED, dtype=int);
  isolated = np.pad(isolated, (1,1), 'constant');
  delta = np.diff(isolated);
  starts = np.where(delta > 0)[0];
  ends = np.where(delta < 0)[0];
  
  #whole stack has no isolated slices
  if len(starts) == 0:
    return  
                 
  #if whole stack is isolated
  if len(starts) == 1 and starts[0] == 0 and len(ends) == 1 and ends[0] == n_status:
    status[:] = self.ISOLATED;
    return;             
                 
  #find left and right bounds for isolated stretches
  for s,e in zip(starts, ends):
    #exclude borders
    if exclude_borders:
      if s == 0 or e == n_status:
         status[s:e] = self.ISOLATED;
         continue;
    
    #find next valid  in each direction
    if s > 0 and status[s-1] >= self.VALID:
      left = wobble[[s-1]];
    else:
      left = None;                     
    if e < n_status and status[e] >= self.VALID:                      
      right = wobble[[e]];
    else:
      right = None;
    if left is None and right is None:
      status[s:e] = self.ISOLATED;
    else:
      if left is None:
        wobble[s:e] = right;
      elif right is None:
        wobble[s:e] = left;
      else: # linearly interpolate
        wobble[s:e] = np.array(np.round((right-left) * 1.0 / (e-s+1) * np.arange(1, e-s+1)[:, np.newaxis] + left), dtype = int);
      status[s:e] = self.FIXED;



###############################################################################
### Stitching
###############################################################################


def stitch_layout(layout, sink, method = 'interpolation', processes = None, verbose = True):
  """Stitches the wobbly sources in a wobbly layout.
  
  Arguments
  ---------
  layout: WobblyLayout class
    The layout of the stacks to stitch.
  method : 'interpolation', 'max', 'min', 'mean'
    The method to use to stitch the sources.  
  processes : int or 'serial' 
    Number of processor to use for parallel processing, if 'serial' process in serial.
  verbose : bool 
    If True, print progress information.
  
  Returns
  -------
  layout : Layout 
    The layout with updated z-alignments.  
  """
  #if not isinstance(layout, WobblyLayout):
  #  raise ValueError('Expecting a WobblyLayout instance as first argument!');
  
  if verbose:
    timer = tmr.Timer();
    print('Stitching: stitching wobbly layout.');
  
  #overall shape
  axis = layout.axis;
  origin = layout.origin_wobbly;
  shape = layout.shape_wobbly;
  #print axis, origin, shape
  
  # create sink
  #TODO: make layout a sink ! use io.create
  io.mmp.create(sink, shape=shape, dtype=layout.dtype, order=layout.order);
  #print shape
  
  #create slices
  coordinates = np.arange(origin[axis], origin[axis] + shape[axis]);
  layout_slices = layout.layouts_along_axis_wobbly(coordinates);
  n_slices = len(layout_slices);
  #print n_slices 
                
  if verbose:
    timer = tmr.Timer();
    print('Stitching: stitching %d sliced layouts.' % n_slices);                
  
  #sliced origin and shape
  full_region = strg.Region(position = origin[:axis] + origin[axis+1:], shape = shape[:axis] + shape[axis+1:])                   
  
  _stitch = ft.partial(_stitch_slice, n_slices=n_slices, sink=sink, method=method, 
                       axis=axis, full_region=full_region, verbose=verbose);
               
  #stitch the data
  if not isinstance(processes, int) and processes != 'serial':
    processes = mp.cpu_count();
  
  if processes == 'serial':
    [_stitch(l,i) for i,l in enumerate(layout_slices)];
  else:
    #for l in layout_slices:
    #  l.sources_as_virtual();
    with concurrent.futures.ProcessPoolExecutor(processes) as executor:
      executor.map(_stitch, layout_slices, range(n_slices));
    
  
  if verbose:
    timer.print_elapsed_time('Stitching: stitching wobbly layout done!');

  return sink;


@ptb.parallel_traceback
def _stitch_slice(slice_layout, slice_id, n_slices, sink, method, axis, full_region, verbose):
  if verbose:
    print('Stitching: stitching wobbly slice %d/%d' % (slice_id, n_slices));
  
  if len(slice_layout.sources) == 0:
    return;
  
  #sliicings
  slice_region = strg.Region(lower = slice_layout.origin, upper = slice_layout.upper);
  #print slice_region, full_region                          
                          
  overlap = strg._overlap(slice_region, full_region);
  if overlap is None:
    return;
  #print overlap
  
  overlap.sources = [slice_region, full_region];
  slice_slicing, full_slicing = overlap.source_slicings();
  full_slicing = full_slicing[:axis] + (slice_id,) + full_slicing[axis:];
  #print slice_slicing, full_slicing
                             
  #stitch
  stitched = strg.stitch_layout(slice_layout, method = method);
  
  #write to sink
  io.write(sink, stitched[slice_slicing], slicing = full_slicing);
  

#############################################################################################################
### Tests
#############################################################################################################

def _test():
  import ClearMap.Alignment.Stitching.StitchingWobbly as stw
  
  from importlib import reload
  reload(stw);
  
  #create some wobbly tiles
  import numpy as np
  import ClearMap.Tests.Files as tfs
  data = np.load(tfs.vasculature_pre)[:,:100,:100];
  
  #linear wobble
  nz = 3;
  data1 = data[:120,:,:nz]
  data2 = np.zeros((100,100,nz), dtype = data.dtype);
  wobble = [];
  for s in range(nz):
    x = 2 * s;
    data2[:,:,s] = data[100+x:200+x,:,s]
    wobble.append((x,0));
  wobble = np.array(wobble);
  
  import matplotlib.pyplot as plt
  plt.figure(1); plt.clf();
  plt.plot(wobble[:,0])
  
  l = stw.WobblyLayout([data1, data2], overlaps = 20);
  stw.align_layout(l, max_shifts = 20, verbose = True, processes = 'serial', validate = True, find_shifts = 'tracing')
  
  a = l.alignments[0];
  a.plot_overlay_wobbly()                  
  
  stw.place_layout(l, method='!optimization', smooth = None, lower_to_origin=True, verbose = True, processes = 'serial')

  s = stw.stitch_layout(l, sink = 'test.npy', method='max', processes = 'serial')
  
  stw.strg.p3d.plot(s)
  
  #true if not optimized
  np.all(stw.io.as_source(s)[:190,:,:] == data[:190,:,:nz])
  
  
  plt.figure(2); plt.clf();
  for s in l.sources:
    plt.plot(s.wobble[:,0] - np.min(s.wobble[:,0]))
  plt.plot(wobble[:,0] - np.min(wobble[:,0]))
  
  np.all(l.sources[1].wobble[:,0] - 100 == wobble[:,0] )
  
  stw.strg.p3d.plot(s)
  
  
  #Sin wobble
  import numpy as np
  import ClearMap.Tests.Files as tfs
  data = np.load(tfs.vasculature_pre)[:,:100,:100];
  
  nz = 30;
  data1 = data[:120,:,:nz]
  data2 = np.zeros((100,100,nz), dtype = data.dtype);
  wobble = [];
  for s in range(nz):
    x = int(10 * np.sin(s * 2 * np.pi/30));
    data2[:,:,s] = data[100+x:200+x,:,s]
    wobble.append((x,0));
  wobble = np.array(wobble);
  
  import matplotlib.pyplot as plt
  plt.figure(1); plt.clf();
  plt.plot(wobble[:,0])

  
  reload(stw.strg)
  reload(stw)
  l = stw.WobblyLayout([data1, data2], overlaps = 20);
  stw.align_layout(l, max_shifts = 20, verbose = True, processes = 'serial', validate = False)
  
  stw.place_layout(l, method='!optimization',  lower_to_origin=True, smooth = None, verbose = True, processes = 'serial')

  s = stw.stitch_layout(l, sink = 'test.npy', method='max', processes = 'serial')
  
  plt.figure(2); plt.clf();
  for s in l.sources:
    plt.plot(s.wobble[:,0] - np.min(s.wobble[:,0]))
  plt.plot(wobble[:,0] - np.min(wobble[:,0]))
  
  
  #True for non-optimized plaements
  np.all(stw.io.as_source(s)[:190,:,:] == data[:190,:,:nz])
  
  
  # wobble + axis alignment
  import ClearMap.Alignment.StitchingWobbly as stw
  reload(stw.strg)
  reload(stw)
  import numpy as np
  import ClearMap.Tests.Files as tfs
  data = np.load(tfs.vasculature_pre)[:,:100,:100];
  
  nz = 30; sh = 5;
  data1 = data[:120,:,:nz]
  data2 = np.zeros((100,100,nz), dtype = data.dtype);
  wobble = [];
  for s in range(nz):
    x = int(10 * np.sin(s * 2 * np.pi/30));
    data2[:,:,s] = data[100+x:200+x,:,s+sh]
    wobble.append((x,0));
  wobble = np.array(wobble);
  
  import matplotlib.pyplot as plt
  plt.figure(1); plt.clf();
  plt.plot(wobble[:,0])
  
  reload(stw.strg)
  reload(stw)
  l = stw.WobblyLayout([data1, data2], overlaps = 20);

  stw.strg.align_layout_axis(l, axis=2, depth=25, max_shifts=10, clip=None, background=None, processes=None, verbose=True)
  l.alignments
 
  plt.figure(10); plt.clf()
  l.alignments[0].plot_mip(depth = 10, max_shifts = [(-30,30),(-30,30),(-20,20)])
  
  stw.strg.place_layout_axis(l, axis = 2, method = 'optimization', min_quality = -np.inf, lower_to_origin = True, verbose = True)
  l.sources

  stw.align_layout(l, max_shifts = 20, verbose = True, processes = '!serial', validate = False, axis_range = (None, None, 3))
  a = l.alignments[0];
  a.plot_overlay_wobbly()                  
  
  
  plt.figure(10); plt.clf();
  plt.plot(l.alignments[0].displacements[:,0])
            
  stw.place_layout(l, method = 'optimization',  lower_to_origin=True, smooth = None, min_quality=-np.inf, processes = '!serial', verbose = True)

  s = stw.stitch_layout(l, sink = 'test.npy', method='max', processes = 'serial')
  
  stw.strg.dv.plot(s)
  
  #True for non-optimized plaements
  np.all(stw.io.as_source(s)[:190,:,sh:nz] == data[:190,:,sh:nz])
    
  
  plt.figure(2); plt.clf();
  for s in l.sources:
    plt.plot(s.wobble[:,0] - np.min(s.wobble[:,0]))
  plt.plot(wobble[:,0] - np.min(wobble[:,0]))
  
  
  
  # wobble + axis alignment + status
  import ClearMap.Alignment.StitchingWobbly as stw
  reload(stw.strg)
  reload(stw)
  import numpy as np
  import ClearMap.Tests.Files as tfs
  data = np.load(tfs.vasculature_pre)[:,:100,:100];
  
  nz = 50; sh = 5;
  data1 = data[:120,:,:nz]
  data2 = np.zeros((100,100,nz), dtype = data.dtype);
  wobble = np.zeros((nz+sh,2), dtype=int);
  for s in range(nz):
    x = int(10 * np.sin(s * 2 * np.pi/40));
    data2[:,:,s] = data[100+x:200+x,:,s+sh]
    wobble[s+sh] = (x,0);
  
  invalid = [13,14,15,16,47,48,49];                   
  for s in invalid:
    data2[:,:,s] = 0;                  
                   
                   
  import matplotlib.pyplot as plt
  plt.figure(1); plt.clf();
  plt.plot(wobble[:,0])
  plt.plot(invalid, np.zeros(len(invalid)), '*', c = 'r')
  
  
  reload(stw.strg)
  reload(stw)
  l = stw.WobblyLayout([data1, data2], overlaps = 20); l.sources[1].position = (100,0,sh);
  
  def plot_status(a, fig = 2):
    sm = a.smooth_displacements(min_quality = -np.inf, method='window', window='bartlett', window_length=10) 
    plt.figure(fig); plt.clf()               
    ax =  plt.subplot(2,2,1);        
    arange = np.arange(a.lower_coordinate, a.upper_coordinate);                     
    for i,d in enumerate(([a.status], [a.qualities], [wobble[arange,0], a.shifts[:,0], sm[:,0]], [wobble[arange,1], a.shifts[:,1], sm[:,1]])):
      plt.subplot(2,2,i+1, sharex = ax);
      for dd in d:             
        #plt.plot(arange, dd)
        plt.plot(dd)                 
                      
                      
  stw._validate(data2[:,:,13], **dict(method='foreground', valid_range = (1, None), size = None) )           
   
  reload(stw.strg)
  reload(stw)
  l = stw.WobblyLayout([data1, data2], overlaps = 20); l.sources[1].position = (100,0,sh);
  
  stw.align_layout(l, max_shifts = 15,  axis_range = (None, None, 1), axis_mip = 1,
                   validate = None,
                   prepare = 'normalization',
                   validate_slice = dict(method='foreground', valid_range = (1, None), size = None),
                   find_shifts = dict(method='tracing', cutoff=np.sqrt(2 * 3**2), debug = True),
                   verbose = True, processes = 'serial')

  a = l.alignments[0];          
  plot_status(a, fig=2)
  a.status[a.status < 0] = stw.WobblyAlignment.UNALIGNED
  a.fix_unaligned()               
  plot_status(a, fig=3)

  
  a = l.alignments[0];     
  results = stw.align_wobbly_axis(a.pre, a.post, max_shifts = 20,  axis_range = (None, None, 1), axis_mip = None,
                                  validate = None,
                                  prepare = 'normalization',
                                  validate_slice = dict(method='foreground', valid_range = (1, None), size = None),
                                  find_shifts = 'minimization',
                                  with_errors = True, with_overlaps = True, verbose = True)
  shifts, qualities, status, errors, ovlps = results;
  stw.strg.dv.plot((errors.transpose([1,2,0]),) + ovlps)
           
  a.plot_overlay_wobbly()                  
  
           
  stw.place_layout(l, method = '!optimization',  lower_to_origin=True, min_quality=-np.inf, 
                   smooth = None, smooth_optimized = None,
                   processes = '!serial', verbose = True)

  #plot the positions of the stacks
  import matplotlib as mpl   #analysis:ignore
  from mpl_toolkits.mplot3d import Axes3D #analysis:ignore
  import matplotlib.pyplot as plt
  fig = plt.figure(200); plt.clf();
  fig.gca(projection='3d') 
  for s in l.sources:
    plt.plot(s.wobble[:,0], s.wobble[:,1], np.arange(s.coordinate, s.coordinate + s.height))    
  plt.title('Source positions')


  plt.figure(300); plt.clf();            
  plt.plot(wobble[:,0])
  for i,s in enumerate(l.sources):
    plt.plot(s.wobble[:,0], label ='%d' % i)
  plt.legend()


  #non alignable planes
  flat = [28,29,30];                   
  for s in flat:
    data1[:,:,s] = 10
    data2[:,:,s] = 10;                  
                   
  stw.strg.dv.plot([data1[:,:,sh:], data2[:,:,:-sh]])
  
  reload(stw.strg)
  reload(stw)
  l = stw.WobblyLayout([data1, data2], overlaps = 20); l.sources[1].position = (100,0,sh);
  
  stw.align_layout(l, max_shifts = 20,  axis_range = (None, None, 1),
                   validate = None,
                   prepare = 'normalization',
                   validate_slice = dict(method='foreground', valid_range = (1, None), size = None),
                   find_shifts = dict(method='tracing', cutoff=np.sqrt(2 * 3**2), debug = False),
                   verbose = True, processes = 'serial')
  stw.place_layout(l, method = '!optimization',  lower_to_origin=True, min_quality=-np.inf, 
                   smooth = None, smooth_optimized = dict(method='window', window_length=10, binary = 2),
                   processes = '!serial', verbose = True)

  s = stw.stitch_layout(l, sink = 'test.npy', method='max', processes = 'serial')
  
  stw.strg.dv.plot(s)
  
  #True for non-optimized plaements
  np.all(stw.io.as_source(s)[:190,:,sh:nz] == data[:190,:,sh:nz])
    
  
  s = l.slice_along_axis_wobbly(32)
  t = stw.strg.stitch_layout(s, sink = None, method = 'max')
  stw.strg.dv.plot(t)
  plt.figure(10); plt.clf();
  plt.imshow(t.T, origin='lower')  
  
  plt.figure(2); plt.clf();
  for s in l.sources:
    plt.plot(np.arange(s.coordinate, s.coordinate + s.height), s.wobble[:,0] - np.min(s.wobble[:,0]))
  plt.plot(wobble[:,0] - np.min(wobble[:,0]))
  
  
  
  #TODO: min_overlap parameter in alignment to avoid boundary effects
  #TODO: option to reduce the shape of the overlaps used for alingment to speed things up
  
  
  ### Test on real data
  import numpy as np
  import ClearMap.IO.IO as io
  import ClearMap.Alignment.Stitching.StitchingRigid as stg
  import ClearMap.Alignment.Stitching.StitchingWobbly as stw
  import ClearMap.IO.Workspace as wsp
  
  directory = '/home/ckirst/Science/Projects/WholeBrainClearing/Vasculature/Experiment/Stitching_2018_06'
  expression = 'tiny_[<Y,2> x <X,2>]_C00.ome.npy'
  ws = wsp.Workspace(name = 'test', directory = directory, expression=expression); 
  io.file_list(ws.filename('expression'))
  
  l = stw.WobblyLayout(expression = ws.filename('expression'), tile_axes = ['X', 'Y'], overlaps = (25, 155));  
  
  # rigid alignment
  lr = stg.TiledLayout(expression = ws.filename('expression'), tile_axes = ['X', 'Y'], overlaps = (45, 155));  
  lr.alignments[0].plot_overlap()
  
  stg.align_layout_rigid_mip(lr, depth=[55, 165, None], max_shifts=[(-30,30),(-30,30),(-20,20)],
                             ranges = [None,None,None], background=(1000, 100), clip = 25000, 
                             verbose=True, processes='!serial')
  lr.alignments[0].plot_overlay()
  
  stg.place_layout(lr, method='optimization', min_quality=-np.inf, lower_to_origin=True, verbose=True)
  lr.alignments[0].plot_overlay()
  # plot result
  
  lr.plot_alignments();
