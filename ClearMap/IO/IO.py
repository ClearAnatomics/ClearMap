# -*- coding: utf-8 -*-
"""
IO
==

IO interface to read files as sources.

This is the main module to distribute the reading and writing of 
individual data formats to the specialized sub-modules.
  
See :mod:`ClearMap.IO` for details.
"""
__author__    = 'Christoph Kirst <christoph.kirst.ck@gmail.com>'
__license__   = 'GPLv3 - GNU General Pulic License v3 (see LICENSE.txt)'
__copyright__ = 'Copyright © 2020 by Christoph Kirst'
__webpage__   = 'http://idisco.info'
__download__  = 'http://www.github.com/ChristophKirst/ClearMap2'


import importlib
import functools
import math

import numpy as np

import multiprocessing as mp
import concurrent.futures


import ClearMap.IO.Source as src
import ClearMap.IO.Slice as slc
import ClearMap.IO.TIF as tif
import ClearMap.IO.NRRD as nrrd
import ClearMap.IO.CSV as csv
import ClearMap.IO.NPY as npy
import ClearMap.IO.MMP as mmp
import ClearMap.IO.SMA as sma
import ClearMap.IO.MHD as mhd
try:
import ClearMap.IO.GT as gt
  gt_loaded = True
except ImportError:
  gt_loaded = False
import ClearMap.IO.FileList as fl
import ClearMap.IO.FileUtils as fu

import ClearMap.Utils.TagExpression as te
import ClearMap.Utils.Timer as tmr

import ClearMap.ParallelProcessing.ParallelTraceback as ptb


###############################################################################
### File manipulation
###############################################################################

from ClearMap.IO.FileUtils import (is_file, is_directory, file_extension,   #analysis:ignore
                                   join, split, abspath, create_directory, 
                                   delete_directory, copy_file, delete_file)

###############################################################################
### Source associations 
###############################################################################
from ClearMap.Utils.utilities import CancelableProcessPoolExecutor

source_modules = [npy, tif, mmp, sma, fl, nrrd, csv]
"""The valid source modules."""

file_extension_to_module = {"npy": mmp, "tif": tif, "tiff": tif, 'nrrd': nrrd,
                            'nrdh': nrrd, 'csv': csv}
if gt_loaded:
  file_extension_to_module['gt'] = gt
  source_modules += [gt]
"""Map between file extensions and modules that handle this file type."""        

###############################################################################
### Source to module conversions
###############################################################################

def source_to_module(source):
  """Returns IO module associated with a source.
  
  Arguments
  ---------
  source : object
    The source specification.
      
  Returns
  -------
  type : module
    The module that handles the IO of the source.
  """
  if isinstance(source, src.Source):
    return importlib.import_module(source.__module__);
  elif isinstance(source, (str, te.Expression)):
    return location_to_module(source);
  elif isinstance(source, np.memmap):
    return mmp;
  elif isinstance(source, (np.ndarray, list, tuple)) or source is None:
    if sma.is_shared(source):
      return sma
    else:
      return npy;
  else:
    raise ValueError('The source %r is not a valid source!' % source);


def location_to_module(location):
  """Returns the IO module associated with a location string.
  
  Arguments
  ---------
  location : object
    Location of the source.
      
  Returns
  -------
  module : module
    The module that handles the IO of the source specified by its location.
  """
  if fl.is_file_list(location):
    return fl;
  else:
    return filename_to_module(location);


def filename_to_module(filename):
  """Returns the IO module associated with a filename.
  
  Arguments
  ---------
  filename : str
    The file name.
      
  Returns
  -------
  module : module
    The module that handles the IO of the file.
  """       
  ext = fu.file_extension(filename);
  
  mod = file_extension_to_module.get(ext, None);
  if mod is None:
    raise ValueError("Cannot determine module for file %s with extension %s!" % (filename, ext));

  return mod;    

##############################################################################
### IO Interface
##############################################################################


#read write interface: specialized modules can assume the following
# read(source, slicing=None, **kwargs)
#  source is a valid source for the module as determined by the module's is_xxx function
# write(sink, data, slicing=None, *kwargs)
#  sink is a valid source for the module as determined by the module's is_xxx function
#  data is a Source class

def is_source(source, exists = True):
  """Checks if source is a valid source.
   
  Arguments
  ---------
  source : object
    Source to check. 
  exists : bool
    If True, check if source exists in case it has a location. 
      
  Returns
  -------
  is_source : bool
    True if source is a valid source.
  """
  if isinstance(source, src.Source):
    if exists:
      return source.exists();
    else:
      return True;  
  
  elif isinstance(source, str):
    try:
      mod = location_to_module(source);
    except:
      return False;
    if exists:
      return mod.Source(source).exists();
    else:
      return True;
    
  elif isinstance(source, np.memmap):
    return True;
    
  elif isinstance(source, (np.ndarray, list, tuple)):
    return True;
  
  else:
    return False;


def as_source(source, slicing = None, *args, **kwargs):
  """Convert source specification to a Source class.
  
  Arguments
  ---------
  source : object
    The source specification.
      
  Returns
  -------
  source : Source class
    The source class.
  """
  if not isinstance(source, src.Source):
    mod = source_to_module(source);
    source =  mod.Source(source, *args, **kwargs);
  if slicing is not None:
    source = slc.Slice(source=source, slicing=slicing);
  return source;


def source(source, slicing = None, *args, **kwargs):
  """Convert source specification to a Source class.
  
  Arguments
  ---------
  source : object
    The source specification.
      
  Returns
  -------
  source : Source class
    The source class.
  """
  return as_source(source, slicing=slicing, *args, **kwargs);


def ndim(source):
  """Returns number of dimensions of a source.
     
  Arguments
  ---------
  source : str, array or Source
    The source specification.
      
  Returns
  -------
  ndim : int
    The number of dimensions in the source.
  """
  source = as_source(source);
  return source.ndim;


def shape(source):
  """Returns shape of a source.
     
  Arguments
  ---------
  source : str, array or Source
    The source specification.
      
  Returns
  -------
  shape : tuple of ints
    The shape of the source.
  """
  source = as_source(source);
  return source.shape;


def size(source):
  """Returns size of a source.
     
  Arguments
  ---------
  source : str, array or Source
    The source specification.
      
  Returns
  -------
  size : int
    The size of the source.
  """
  source = as_source(source);
  return source.size;


def dtype(source):
  """Returns dtype of a source.
     
  Arguments
  ---------
  source : str, array or Source
    The source specification.
      
  Returns
  -------
  dtype : dtype
    The data type of the source.
  """
  source = as_source(source);
  return source.dtype; 


def order(source):
  """Returns order of a source.
     
  Arguments
  ---------
  source : str, array or Source
    The source specification.
      
  Returns
  -------
  order : 'C', 'F', or None
    The order of the source data items.
  """
  source = as_source(source);
  return source.order; 


def location(source):
  """Returns the location of a source.
     
  Arguments
  ---------
  source : str, array or Source
    The source specification.
      
  Returns
  -------
  location : str or None
    The location of the source.
  """
  source = as_source(source);
  return source.location; 


def memory(source):
  """Returns the memory type of a source.
     
  Arguments
  ---------
  source : str, array or Source
    The source specification.
      
  Returns
  -------
  memory : str or None
    The memory type of the source.
  """
  if sma.is_shared(source):
    return 'shared'
  else:
    return None;


def element_strides(source):
  """Returns the strides of the data array of a source.
  
  Arguments
  ---------
  source : str, array, dtype or Source
    The source specification.
      
  Returns
  -------
  strides : tuple of int
    The strides of the souce.
  """
  try:
    source = as_source(source);
    strides = source.element_strides;
  except:
    raise ValueError('Cannot determine the strides for the source!');   
  
  return strides;     


def buffer(source):
  """Returns an io buffer of the data array of a source for use with e,g python.
  
  Arguments
  ---------
  source : source specification
    The source specification.
      
  Returns
  -------
  buffer : array or memmap
    A buffer to read and write data.
  """
  try:
    source = as_source(source);
    buffer = source.as_buffer();
  except:
    raise ValueError('Cannot get a io buffer for the source!');   
  
  return buffer;  


#TODO: arg memory= to specify which kind of array is created, better use device=
#TODO: arg processes= in order to use ParallelIO -> can combine with buffer=
def read(source, *args, **kwargs):
  """Read data from a data source.
  
  Arguments
  ---------
  source : str, array, Source class
    The source to read the data from.
  
  Returns
  -------
  data : array
    The data of the source.
  """
  mod = source_to_module(source);
  return mod.read(source, *args, **kwargs);


def write(sink, data, *args, **kwargs):
  """Write data to a data source.
  
  Arguments
  ---------
  sink : str, array, Source class
    The source to write data to.
  data : array
    The data to write to the sink.
  slicing : slice specification or None
    Optional subslice to write data to. 
  
  Returns
  -------
  sink : str, array or Source class
    The sink to which the data was written.
  """
  mod = source_to_module(sink);
  return mod.write(sink, as_source(data), *args, **kwargs);


def create(source, *args, **kwargs):
  """Create a data source on disk.
  
  Arguments
  ---------
  source : str, array, Source class
    The source to write data to.
  
  Returns
  -------
  sink : str, array or Source class
    The sink to which the data was written.
  """
  mod = source_to_module(source);
  return mod.create(source, *args, **kwargs);


def initialize(source = None, shape = None, dtype = None, order = None, location = None, memory = None, like = None, hint = None, **kwargs):
  """Initialize a source with specified properties.
  
  Arguments
  ---------
  source : str, array, Source class
    The source to write data to.
  shape : tuple or None
    The desired shape of the source.
    If None, infered from existing file or from the like parameter.
    If not None and source has a valid shape shapes are tested to match.
  dtype : type, str or None
    The desired dtype of the source.
    If None, infered from existing file or from the like parameter.
    If not None and source has a valid dtype the types are tested to match.
  order : 'C', 'F' or None
    The desired order of the source.
    If None, infered from existing file or from the like parameter. 
    If not None and source has a valid order the orders are tested to match.
  location : str or None
    The desired location of the source.
    If None, infered from existing file or from the like parameter. 
    If not None and source has a valid location the locations need to match. 
  memory : 'shared' or None
    The memory type of the source. If 'shared' a shared array is created.
  like : str, array or Source class
    Infer the source parameter from this source.
  hint : str, array or Source class
    If parameters for source creation are missing use the ones from this
    hint source.
  
  Returns
  -------
  source : Source class
    The initialized source.
    
  Note
  ----
  The source is created on disk or in memory if it does not exists so processes
  can start writing into it.
  """
  if isinstance(source, (str, te.Expression)):
    location = source;
    source = None;
  
  if like is not None:
    like = as_source(like);
    if shape is None:
      shape = like.shape;
    if dtype is None:
      dtype = like.dtype;
    if order is None:
      order = like.order;

  if source is None:
    if location is None:
      shape, dtype, order = _from_hint(hint, shape, dtype, order);
      if memory == 'shared':
        return sma.create(shape=shape, dtype=dtype, order=order, **kwargs);
      else:
        return npy.create(shape=shape, dtype=dtype, order=order);
    else:
      try:
        source = as_source(location);
      except:
        try:
          shape, dtype, order = _from_hint(hint, shape, dtype, order);
          mod = location_to_module(location);
          return mod.create(location=location, shape=shape, dtype=dtype, order=order, **kwargs);
        except Exception as error:
          raise ValueError(f'Cannot initialize source for location {location} - {error}')
    
  if isinstance(source, np.ndarray):
    source = as_source(source);
  
  if not isinstance(source, src.Source):
    raise ValueError('Source specification %r not a valid location, array or Source class!' % source)
  
  if shape is not None and shape != source.shape:
    raise ValueError('Incompatible shapes %r != %r for the source %r!' % (shape, source.shape, source));
  if dtype is not None and dtype != source.dtype:
    raise ValueError('Incompatible dtype %r != %r for the source %r!' % (dtype, source.dtype, source));
  if order is not None and order != source.order:
    raise ValueError('Incompatible order %r != %r for the source %r!' % (order, source.order, source));
  if location is not None and abspath(location) != abspath(source.location):
    raise ValueError('Incompatible location %r != %r for the source %r!' % (location, source.location, source)); 
  if memory == 'shared' and not sma.is_shared(source):
    raise ValueError('Incompatible memory type, the source %r is not shared!' % (source,));  
  
  return source;


def _from_hint(hint, shape, dtype, order):
  """Helper for initialize."""
  if hint is not None:
    try:
      hint = as_source(hint);
      if shape is None:
        shape = hint.shape;
      if dtype is None:
        dtype = hint.dtype;
      if order is None:
        order = hint.order;
    except:
      pass;
  return shape, dtype, order;
  

def initialize_buffer(source, shape = None, dtype = None, order = None, location = None, memory = None, like = None, **kwargs):
  """Initialize a buffer with specific properties.
  
  Arguments
  ---------
  source : str, array, Source class
    The source to write data to.
  shape : tuple or None
    The desired shape of the source.
    If None, infered from existing file or from the like parameter.
    If not None and source has a valid shape shapes are tested to match.
  dtype : type, str or None
    The desired dtype of the source.
    If None, infered from existing file or from the like parameter.
    If not None and source has a valid dtype the types are tested to match.
  order : 'C', 'F' or None
    The desired order of the source.
    If None, infered from existing file or from the like parameter. 
    If not None and source has a valid order the orders are tested to match.
  location : str or None
    The desired location of the source.
    If None, infered from existing file or from the like parameter. 
    If not None and source has a valid location the locations need to match. 
  memory : 'shared' or None
    The memory type of the source. If 'shared' a shared array is created.
  like : str, array or Source class
    Infer the source parameter from this source.
 
  Returns
  -------
  buffer : array
    The initialized buffer to use tih e.g. cython.
    
  Note
  ----
  The buffer is created if it does not exists.
  """
  source = initialize(source, shape=shape, dtype=dtype, order=order, location=location, memory=memory, **kwargs)
  return source.as_buffer();


###############################################################################
### Utils
###############################################################################

def file_list(expression = None, file_list = None, sort = True, verbose = False):
  """Returns the list of files that match the tag expression.
  
  Arguments
  ---------
  expression :str
    The regular expression the file names should match.
  sort : bool
    If True, sort files naturally.
  verbose : bool
    If True, print warning if no files exists.
  
  Returns
  -------
  file_list : list of str
    The list of files that matched the expression.
  """
  return fl._file_list(expression=expression, file_list=file_list, sort=sort, verbose=verbose)


def max_value(source):
  """Returns the maximal value of the data type of a source.
  
  Arguments
  ---------
  source : str, array, dtype or Source
    The source specification.
      
  Returns
  -------
  max_value : number
    The maximal value for the data type of the source
  """
  if isinstance(source, (src.Source, np.ndarray)):
    source = source.dtype;
  
  if isinstance(source, str):
    try:
      source = np.dtype(source);
    except:
      pass;
  
  if not isinstance(source, (type, np.dtype)):
    source = dtype(source);

  try:
    max_value = np.iinfo(source).max;
  except:
    try: 
      max_value = np.finfo(source).max;
    except:
      raise ValueError('Cannot determine the maximal value for the type %r!' % source);   
  return max_value;                          
                             

def min_value(source):
  """Returns the minimal value of the data type of a source.
  
  Arguments
  ---------
  source : str, array, dtype or Source
    The source specification.
      
  Returns
  -------
  min_value : number
    The minimal value for the data type of the source
  """
  if isinstance(source, str):
    try:
      source = np.dtype(source);
    except:
      pass;
  
  if not isinstance(source, (type, np.dtype)):
    source = dtype(source);
  
  try:
    min_value = np.iinfo(source).min;
  except:
    try: 
      min_value = np.finfo(source).min;
    except:
      raise ValueError('Cannot determine the minimal value for the type %r!' % source);   
  return min_value;     



def convert(source, sink, processes = None, verbose = False, **kwargs):
  """Transforms a source into another format.
  
  Arguments
  ---------
  source : source specification
    The source or list of sources.
  sink : source specification
    The sink or list of sinks.
  
  Returns
  -------
  sink : sink speicication
    The sink or list of sinkfs.
  """      
  source = as_source(source);
  if verbose:
    print('converting %s -> %s' % (source, sink)) 
  mod = source_to_module(source);
  if hasattr(mod, 'convert'):
    return mod.convert(source, sink, processes=processes, verbose=verbose, **kwargs);
  else:
    return write(sink, source);



def convert_files(filenames, extension = None, path = None, processes = None, verbose = False, workspace=None,
                  verify=False):
  """Transforms list of files to their sink format in parallel.
  
  Arguments
  ---------
  filenames : list of str
    The filenames to convert
  extension : str
    The new file format extension.
  path : str or None
    Optional path speicfication.
  processes : int, 'serial' or None
    The number of processes to use for parallel conversion.
  verbose : bool
    If True, print progress information.
  
  Returns
  -------
  filenames : list of str
    The new file names.
  """      
  if not isinstance(filenames, (tuple, list)):
    filenames = [filenames];
  if len(filenames) == 0:
    return [];
  n_files = len(filenames); 
  
  if path is not None:
    filenames = [fu.join(path, fu.split(f)[1]) for f in filenames]; 
  sinks = ['.'.join(f.split('.')[:-1] + [extension]) for f in filenames];
  
  if verbose:
    timer = tmr.Timer()
    print('Converting %d files to %s!' % (n_files, extension));
  
  if not isinstance(processes, int) and processes != 'serial':
    processes = mp.cpu_count();
  
  #print(n_files, extension, filenames, sinks)
  _convert = functools.partial(_convert_files, n_files=n_files, extension=extension, verbose=verbose, verify=verify);
  
  if processes == 'serial':
    [_convert(source,sink,i) for i,source,sink in zip(range(n_files), filenames, sinks)];
  else:
    with CancelableProcessPoolExecutor(processes) as executor:
      executor.map(_convert, filenames, sinks, range(n_files))
      if workspace is not None:
        workspace.executor = executor
    if workspace is not None:
      workspace.executor = None
                  
  if verbose:
    timer.print_elapsed_time('Converting %d files to %s' % (n_files, extension));
  
  return sinks;


@ptb.parallel_traceback
def _convert_files(source, sink, fid, n_files, extension, verbose, verify=False):
  source = as_source(source);              
  if verbose:
    print('Converting file %d/%d %s -> %s' % (fid,n_files,source,sink))
  mod = file_extension_to_module[extension];    
  if mod is None:
    raise ValueError("Cannot determine module for extension %s!" % extension);
  mod.write(sink,source);
  if verify:
    src_mean = source.mean()
    sink_mean = mod.read(sink).mean()
    if not math.isclose(src_mean, sink_mean, rel_tol=1e-5):
      raise RuntimeError(f"Conversion of {source} to {sink} failed, means differ")


###############################################################################
### Helpers
###############################################################################

_shape = shape
_dtype = dtype
_order = order
_location = location
_memory = memory

###############################################################################
### Tests
###############################################################################

def _test():
  import ClearMap.IO.IO as io
  
  print(io.abspath('.'))
  #reload(io)
  
  


#def memmap(source): 
#def shared(source):
  



#TODO: 
#
#def copy(source, sink):
#    """Copy a data file from source to sink, which can consist of multiple files
#    
#    Arguments:
#        source (str): file name of source
#        sink (str): file name of sink
#    
#    Returns:
#        str: name of the copied file
#    
#    See Also:
#        :func:`copyImage`, :func:`copyArray`, :func:`copyData`, :func:`convert`
#    """     
#    
#    return copyData(source, sink);
#
#
#def convert(source, sink, **args):
#    """Transforms data from source format to sink format
#    
#    Arguments:
#        source (str): file name of source
#        sink (str): file name of sink
#    
#    Returns:
#        str: name of the copied file
#        
#    Warning:
#        Not optimized for large image data sets yet
#    
#    See Also:
#        :func:`copyImage`, :func:`combineImage`
#    """      
#
#    if source is None:
#        return None;
#    
#    elif isinstance(source, str):
#        if sink is None:        
#            return read(source, **args);
#        elif isinstance(sink, str):
#            #if args == {} and dataFileNameToType(source) == dataFileNameToType(sink):
#            #    return copy(source, sink);
#            #else:
#            data = read(source, **args); #TODO: improve for large data sets
#            return write(sink, data);
#        else:
#            raise RuntimeError('convert: unknown sink format!');
#            
#    elif isinstance(source, numpy.ndarray):
#        if sink is None:
#            return dataFromRegion(source, **args);
#        elif isinstance(sink,  str):
#            data = dataFromRange(source, **args);
#            return writeData(sink, data);
#        else:
#            raise RuntimeError('convert: unknown sink format!');
#    
#    else:
#      raise RuntimeError('convert: unknown srouce format!');
    
#
###############################################################################
## Other
###############################################################################
#
#def writeTable(filename, table):
#    """Writes a numpy array with column names to a csv file.
#    
#    Arguments:
#        filename (str): filename to save table to
#        table (annotated array): table to write to file
#        
#    Returns:
#        str: file name
#    """
#    with open(filename,'w') as f:
#        for sublist in table:
#            f.write(', '.join([str(item) for item in sublist]));
#            f.write('\n');
#        f.close();
#
#    return filename;
#




#############
# Temp


#def isFileExpression(source):
#    """Checks if filename is a regular expression denoting a file list
#    
#    Arguments:
#        source (str): source file name
#        
#    Returns:
#        bool: True if source is regular expression with a digit placeholder
#    """    
#
#    if not isinstance(source, str):
#      return False;    
#      
#    ext = fileExtension(source);
#    if not ext in dataFileExtensions:
#      return False
#    
#    #sepcified number of digits
#    searchRegex = re.compile('.*\\\\d\{(?P<digit>\d)\}.*').search
#    m = searchRegex(source); 
#    if not m is None:
#      return True;
#    
#    #digits without trailing zeros \d* or 
#    searchRegex = re.compile('.*\\\\d\*.*').search
#    m = searchRegex(source); 
#    if not m is None:
#      return True;
#    
#    #digits without trailing zeros \d{} or 
#    searchRegex = re.compile('.*\\\\d\{\}.*').search
#    m = searchRegex(source); 
#    if not m is None:
#      return True;
#      
#    return False;
#
#     
#def isDataFile(source, exists = False):
#    """Checks if a file has a valid data file extension useable in *ClearMap*
#     
#    Arguments:
#        source (str): source file name
#        exists (bool): if true also checks if source exists 
#        
#    Returns:
#        bool: true if source is an data file usable in *ClearMap*
#    """   
#    
#    if not isinstance(source, str):
#        return False;    
#    
#    fext = fileExtension(source);
#    if fext in dataFileExtensions:
#      if not exists:
#        return True;
#      else:
#        return exitsFile(source) or existsFileExpression(source);
#    else:
#        return False;
#        
#
#def isImageFile(source, exists = False):
#    """Checks if a file has a valid image file extension useable in *ClearMap*
#     
#    Arguments:
#        source (str): source file name
#        exists (bool): if true also checks if source exists 
#        
#    Returns:
#        bool: true if source is an image file usable in *ClearMap*
#    """   
#    
#    if not isinstance(source, str):
#        return False;    
#    
#    fext = fileExtension(source);
#    if fext in imageFileExtensions:
#      if not exists:
#        return True;
#      else:
#        return exitsFile(source) or existsFileExpression(source);
#    else:
#        return False;
#
#
#def isArrayFile(source, exists =False):
#    """Checks if a file is a valid array data file
#     
#    Arguments:
#        source (str): source file name
#        exists (bool): if true also checks if source exists 
#        
#    Returns:
#        bool: true if source is a array data file
#    """     
#    
#    if not isinstance(source, str):
#        return False;
#    
#    fext = fileExtension(source);
#    if fext in pointFileExtensions:
#      if not exists:
#        return True;
#      else:
#        return exitsFile(source) or existsFileExpression(source);
#    else:
#        return False;
#
#
#def isDataSource(source, exists = False):
#  """Checks if source is a valid data source for use in *ClearMap*
#   
#  Arguments:
#      source (str): source file name or array
#      exists (bool): if true also checks if source exists 
#      
#  Returns:
#      bool: true if source is an data source usable in *ClearMap*
#  """  
#  
#  return (not exists and source is None) or isinstance(source, numpy.ndarray) or isDataFile(source, exists = exists);
