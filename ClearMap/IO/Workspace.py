# -*- coding: utf-8 -*-
"""
Workspace
=========

Workspace module and class to keep track of the data files of a project.
Using this module will simplify access to data and results using coherent
filenames accross experiments and samples.

One can think of a Workspace as a transparent data structure for ClearMap.

Note
----
Additional standard filenames can be added in the ftype_to_filename dict.
"""
__author__    = 'Christoph Kirst <ckirst@rockefeller.edu>'
__license__   = 'MIT License <http://www.opensource.org/licenses/mit-license.php>'
__copyright__ = 'Copyright (c) 2019 by Christoph Kirst'

#TODO: add DataJoint or NWB data formats 

import os
import numpy as np

from collections import OrderedDict as odict

import ClearMap.IO.IO as io

import ClearMap.Utils.TagExpression as te

import ClearMap.ParallelProcessing.DataProcessing.ArrayProcessing as ap

import ClearMap.Visualization.Plot3d as p3d


###############################################################################
### Filenames
###############################################################################

default_file_type_to_name = odict(
    raw                       = "/Raw/raw_<X,2>_<Y,2>.npy",
    autofluorescence          = "/Autofluorescence/auto_<X,2>_<Y,2>.npy",
    stitched                  = "stitched.npy", 
    layout                    = "layout.lyt",
    background                = "background.npy",
    resampled                 = "resampled.tif",
    resampled_to_auto         = 'elastix_resampled_to_auto',
    auto_to_reference         = 'elastix_auto_to_reference', 
    );

file_type_synonyms = dict(
    r  = "raw",
    a  = "autofluorescence",
    st = "stitched",
    l  = "layout",
    bg = "background",
    rs = "resampled"
    );
  
default_file_type_to_name_tube_map = default_file_type_to_name.copy();
default_file_type_to_name_tube_map.update(
    arteries                  = "/Raw/arteries_<X,2>_Y,2>.npy", 
    binary                    = "binary.npy",
    binary_status             = "binary_status.npy",
    skeleton                  = 'skeleton.npy',
    graph                     = "graph.gt",
    density                   = "density.tif"
    );
   
file_type_synonyms.update(
    b  = "binary",
    bs = "binary_status",
    g  = "graph",
    sk = "skeleton"
    );


default_file_type_to_name_cell_map = default_file_type_to_name.copy();
default_file_type_to_name_cell_map.update(
    raw                       = "/Raw/Z<Z,4>.tif",
    autofluorescence          = "/Autofluorescence/Z<Z,4>.tif",
    cells                     = 'cells.npy',
    density                   = "density.tif",
    );    

file_type_synonyms.update(
    c  = "cells"
    );
    

default_workspaces = odict(
    CellMap = default_file_type_to_name_cell_map,
    TubeMap = default_file_type_to_name_tube_map
    )
    

def filename(ftype, file_type_to_name = None, directory = None, expression = None, values = None, prefix = None, postfix = None, extension = None, debug = None):
  """Returns the standard file name to use for a result file.
  
  Arguments
  ---------
  ftype : str
    The type of the file for which the file name is requested.
  directory : str
    The working directory of the project.
  expression : str or None
    The tag expression to use if ftype is 'expression'.
  file_type_to_name : dict
    The file types to name mappings. If None, the default is used.
  values : dict or None
    The values to use in case a tag expression is given.
  prefix : str or None
    Optional prefix to the file if not None.
  postfix : str or list of str or None
    Optional postfix to the file if not None.
  extension : str or None
    Optional extension to replace existing one.
  debug : str, bool or None
    Optional string for debug files in wihch the string is added as postfix.
    If True, 'debug' is added.
  
  Returns
  -------
  filename : str
    The standard file name of the requested file type.
  """
  if file_type_to_name is None:
    file_type_to_name = default_file_type_to_name;
  
  if ftype is None:
    fname = directory;
  if ftype in file_type_synonyms.keys():
    ftype = file_type_synonyms[ftype];
  if ftype == 'expression' or expression is not None:
    expression = te.Expression(expression);
    fname = expression.string(values=values);
    #Note: expressions are used for raw data only atm -> no prefix, debug
    prefix = None;
    debug = None;   
  else:
    fname = file_type_to_name.get(ftype, None);
  e = te.Expression(fname);
  if len(e.tags) > 0: 
    fname = e.string(values=values);
    #Note: expressions are used for raw data only atm -> no prefix, debug
    prefix = None;
    debug = None;
  
  if fname is None:
    raise ValueError('Cannot find name for type %r!' % ftype);
  
  if prefix and prefix != '':
    if isinstance(prefix, list):
      prefix = '_'.join(prefix);
    fname = prefix + '_' + fname;
  
  if postfix is not None and postfix != '':
    if isinstance(postfix, list):
      postfix = '_'.join(postfix);
    fname = fname.split('.');
    fname = '.'.join(fname[:-1]) + '_' + postfix + '.' + fname[-1];   
  
  if debug:
    if not isinstance(debug, str):
      debug = 'debug';
    #fname = fname.split('.');
    #fname = '.'.join(fname[:-1]) + '_' + debug + '.' + fname[-1];   
    fname = debug + '_' + fname;
  
  if extension:
    fname = fname.split('.');
    fname = '.'.join(fname[:-1] + [extension]);                
  
  if directory:
    fname = io.join(directory, fname);
  
  return fname;                 


###############################################################################
### Workspace
###############################################################################

class Workspace(object):
  """Class to organize files."""
  
  def __init__(self, wtype = None, prefix = None, file_type_to_name = None, directory = None, debug = None, **kwargs):
    self._wtype = wtype;
    self._prefix = prefix;
    self.directory = directory;
    self._file_type_to_name = default_workspaces.get(wtype, default_file_type_to_name).copy();
    if file_type_to_name is not None:
      self._file_type_to_name.update(file_type_to_name);
    self._file_type_to_name.update(**kwargs);
    self._debug = debug;
    
  @property
  def wtype(self):
    return self._wtype;
  
  @wtype.setter
  def wtype(self, value):
    self.update(default_workspaces.get(value, default_file_type_to_name))
    self._wtype = value;   
    
  @property
  def prefix(self):
    return self._prefix;
  
  @prefix.setter
  def prefix(self, value):
    self._prefix = value;

  @property
  def directory(self):
    return self._directory;
  
  @directory.setter
  def directory(self, value):
    if value and len(value) > 0 and value[-1] == os.path.sep:
      value = value[:-1];
    self._directory = value;
    
  @property
  def file_type_to_name(self):
    return self._file_type_to_name;
  
  @file_type_to_name.setter
  def file_type_to_name(self, value):
    self._file_type_to_name = value;
    
  def update(self, *args, **kwargs):
    self._file_type_to_name.update(*args, **kwargs);
  
  @property
  def debug(self):
    return self._debug;
  
  @debug.setter
  def debug(self, value):
    if value is True:
      value = 'debug';
    if value is False:
      value = None;
    self._debug = value;
  
  def create_debug(self, ftype, slicing, debug = None, **kwargs):
    if debug is None:
      debug = self.debug;
    if debug is None:
      debug = 'debug';
    self.debug = None;
    source = io.as_source(self.filename(ftype, **kwargs));
    self.debug = debug;
    return io.write(self.filename(ftype, **kwargs), np.asarray(source[slicing], order='F'));
  
  def plot(self, ftype, **kwargs):
    return p3d.plot(self.filename(ftype, **kwargs));
  
  def load(self, filename):
    """Loads the configuration from disk"""
    d = np.load(filename)[0];
    self.__dict__.update(d);
#    
  def save(self, filename):
    """Saves the configuration to disk"""
    #prevent np to add .npy to a .workspace file
    fid = open(filename, "wb");
    np.save(fid, [self.__dict__]); 
    fid.close();
  
  def filename(self, ftype, file_type_to_name = None, directory = None, expression = None, values = None, prefix = None, extension = None, debug = None, **kwargs):
    if directory is None:
      directory = self.directory;
    if prefix is None:
      prefix = self.prefix;
    if file_type_to_name is None:
      file_type_to_name = self.file_type_to_name;
    if debug is None:
      debug = self.debug;
    return filename(ftype, file_type_to_name=file_type_to_name, 
                    directory=directory, expression=expression, 
                    values=values, prefix=prefix, extension=extension, 
                    debug=debug, **kwargs);
  
  def expression(self, *args,**kwargs):
    return te.Expression(self.filename(*args, **kwargs));

                  
  def extension(self, ftype, file_type_to_name = None, directory = None, expression = None, values = None, prefix = None, extension = None, debug = None, **kwargs):
    filename = self.filename(ftype=ftype,  file_type_to_name=file_type_to_name,
                             directory=directory, expression=expression,
                             values=values, prefix=prefix, extension=extension,
                             debug=debug, **kwargs);
    return io.extension(filename);                             
  
  def file_list(self, ftype, file_type_to_name = None, directory = None, expression = None, values = None, prefix = None, extension = None, debug = None, **kwargs):
    filename = self.filename(ftype=ftype,  file_type_to_name=file_type_to_name,
                             directory=directory, expression=expression,
                             values=values, prefix=prefix, extension=extension,
                             debug=debug, **kwargs)
    return io.file_list(filename);    
  
  def create(self, ftype, dtype = None, shape = None, order = None, 
                   file_type_to_name = None, directory = None, expression = None, 
                   values = None, prefix = None, extension = None, debug = None, **kwargs):
    filename = self.filename(ftype=ftype,  file_type_to_name=file_type_to_name,
                             directory=directory, expression=expression, 
                             values=values, prefix=prefix, 
                             debug=debug, **kwargs)
    io.create(filename, shape=shape, dtype=dtype, order=order)
    return filename
  
  def source(self, *args, **kwargs):
    return io.as_source(self.filename(*args, **kwargs));
  
  def read(self, *args, **kwargs):
   return ap.read(self.filename(*args, **kwargs));
 
  #def write(self, *args, **kwargs):
  # return ap.write(self.filename(*args, **kwargs));
  
  def __str__(self):
    #s = self.__class__.__name__;
    s = "Workspace";
    if self.wtype is not None:
      s = s + ('[%s]' % self.wtype); 
    if self.prefix is not None:
      s = s + ('(%s)' % self.prefix);
    if self.directory is not None:
      s = s + '{%s}' % self.directory;
    if self.debug is not None:
        s = s + '[' + self.debug +']'
    return s;
  
  def __repr__(self):
    return self.__str__()
  
  
  def info(self, tile_axes = None, check_extensions = True):
    s = self.__str__() + '\n';
    
    l = np.max([len(k) for k in self.file_type_to_name]);
    l = '%' + '%d' % l + 's';         
        
    for k,v in self.file_type_to_name.items():
      if len(te.Expression(v).tags) > 0:
        if check_extensions:
          files = self.file_list(k, extension='*');
          extensions = [io.file_extension(f) for f in files];
          extensions = np.unique(extensions);
          #print(extensions)
        else:
          extensions = [self.extension(k)];
        
        if len(extensions) == 0:
          s += l % k + ': no file\n';
        else:
          kk = k;
          for extension in extensions:
            expression = te.Expression(self.filename(k, extension=extension));
            tag_names = expression.tag_names();
            if tile_axes is None:
              tile_axes_ = tag_names;
            else:
              tile_axes_ = tile_axes;
            for n in tile_axes_:
              if not n in tag_names:
                raise ValueError('The expression does not have the named pattern %s' % n);
            for n in tag_names:
              if not n in tile_axes_:
                raise ValueError('The expression has the named pattern %s that is not in tile_axes=%r' % (n, tile_axes_));
            
            #construct tiling
            files = io.file_list(expression);
            if len(files) > 0:
              tile_positions = [expression.values(f) for f in files];
              tile_positions = [tuple(tv[n] for n in tile_axes_) for tv in tile_positions];
              tile_lower = tuple(np.min(tile_positions, axis = 0)); 
              tile_upper = tuple(np.max(tile_positions, axis = 0));
              tag_names = tuple(tag_names);
              
              if kk is not None:
                s += (l % kk) + ': ' 
                kk = None;
              else:
                s += (l % '') + '  '
              s+= ('%s {%d files, %r: %r -> %r}' % (expression.string()[len(self.directory)+1:], len(files), tag_names, tile_lower, tile_upper)) + '\n';   
      
      else:
        fname = self.filename(k);
        files = [];
        if io.is_file(fname):
          files += [fname];
        fname = self.filename(k, postfix = '*');
        files += io.file_list(fname);
        if len(files) > 0:
          files = [f[len(self.directory)+1:] for f in files]
          
          s += l % k + ': ' + files[0] + '\n'
          for f in files[1:]:
            s += l % '' + '  ' + f + '\n'
        else:
          s += l % k + ': no file\n';        
                    
    print(s);