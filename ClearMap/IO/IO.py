# -*- coding: utf-8 -*-
"""
IO
==

IO interface to read files as sources.

This is the main module to distribute the reading and writing of 
individual data formats to the specialized submodules.
  
See :mod:`ClearMap.IO` for details.
"""
__author__ = 'Christoph Kirst <christoph.kirst.ck@gmail.com>'
__license__ = 'GPLv3 - GNU General Public License v3 (see LICENSE.txt)'
__copyright__ = 'Copyright © 2020 by Christoph Kirst'
__webpage__ = 'https://idisco.info'
__download__ = 'https://github.com/ClearAnatomics/ClearMap'


import importlib
import functools
import math
import pathlib
import multiprocessing as mp
import warnings

import numpy as np


import ClearMap.IO.Source as src
import ClearMap.IO.Slice as slc
import ClearMap.IO.TIF as tif
import ClearMap.IO.NRRD as nrrd
import ClearMap.IO.CSV as csv
import ClearMap.IO.NPY as npy
import ClearMap.IO.MMP as mmp
import ClearMap.IO.SMA as sma
import ClearMap.IO.MHD as mhd
from ClearMap.Utils.exceptions import IncompatibleSource, SourceModuleNotFoundError

try:
    import ClearMap.IO.GT as gt
    gt_loaded = True
except ImportError:
    gt_loaded = False
import ClearMap.IO.FileList as fl
import ClearMap.IO.FileUtils as fu

import ClearMap.Utils.tag_expression as te
import ClearMap.Utils.Timer as tmr

import ClearMap.ParallelProcessing.ParallelTraceback as ptb

from ClearMap.Utils.utilities import CancelableProcessPoolExecutor


###############################################################################
# ## File manipulation
###############################################################################
# FIXME:
from ClearMap.IO.FileUtils import (is_file, is_directory, file_extension,   # analysis:ignore
                                   join, split, abspath, create_directory, 
                                   delete_directory, copy_file, delete_file)

###############################################################################
# ## Source associations
###############################################################################

source_modules = [npy, tif, mmp, sma, fl, nrrd, mhd, csv]
"""The valid source modules."""

file_extension_to_module = {'npy': mmp,
                            'tif': tif,
                            'tiff': tif,
                            'nrrd': nrrd,
                            'nrdh': nrrd,
                            'csv': csv,
                            'mhd': mhd}
if gt_loaded:
    file_extension_to_module['gt'] = gt
    source_modules += [gt]
"""Map between file extensions and modules that handle this file type."""        


###############################################################################
# ## Source to module conversions
###############################################################################
def source_to_module(source_):
    """
    Returns IO module associated with a source.

    Arguments
    ---------
    source_ : object
        The source specification.

    Returns
    -------
    type : module
        The module that handles the IO of the source.
    """
    if isinstance(source_, pathlib.Path):
        source_ = str(source_)

    if isinstance(source_, src.Source):
        return importlib.import_module(source_.__module__)
    elif isinstance(source_, (str, te.Expression)):
        return location_to_module(source_)
    elif isinstance(source_, np.memmap):
        return mmp
    elif isinstance(source_, (np.ndarray, list, tuple)) or source_ is None:
        if sma.is_shared(source_):
            return sma
        else:
            return npy
    else:
        raise ValueError(f'The source {source_} is not a valid source!')


def location_to_module(location_):
    """
    Returns the IO module associated with a location string.

    Arguments
    ---------
    location_ : str or te.Expression or pathlib.Path
        Location of the source.

    Returns
    -------
    module : module
        The module that handles the IO of the source specified by its location.
    """
    if isinstance(location_, pathlib.Path):
        location_ = str(location_)
    if fl.is_file_list(location_):
        return fl
    else:
        return filename_to_module(location_)


def filename_to_module(filename):
    """
    Returns the IO module associated with a filename.

    Arguments
    ---------
    filename : str
       The file name.

    Returns
    -------
    module : module
       The module that handles the IO of the file.
    """
    if isinstance(filename, pathlib.Path):
        filename = str(filename)

    ext = fu.file_extension(filename)

    mod = file_extension_to_module.get(ext, None)
    if mod is None:
        raise SourceModuleNotFoundError(f"Cannot determine module for file {filename} with extension {ext}!")

    return mod

##############################################################################
# ## IO Interface
##############################################################################
# FIXME: add support for Assets

# read write interface: specialized modules can assume the following
# read(source, slicing=None, **kwargs)
#  source is a valid source for the module as determined by the module's is_xxx function
# write(sink, data, slicing=None, *kwargs)
#  sink is a valid source for the module as determined by the module's is_xxx function
#  data is a Source class

def is_source(source_, exists=True):
    """
    Checks if `source_` is a valid Source.

    Arguments
    ---------
    source_ : object
        Source to check.
    exists : bool
        If True, check if source exists in case it has a location.

    Returns
    -------
    is_source : bool
       True if source is a valid source.
    """
    if isinstance(source_, pathlib.Path):
        source_ = str(source_)

    if isinstance(source_, src.Source):
        if exists:
            return source_.exists()
        else:
            return True
    elif isinstance(source_, str):
        try:
            mod = location_to_module(source_)
        except SourceModuleNotFoundError:
            return False
        if exists:
            return mod.Source(source_).exists()
        else:
            return True
    elif isinstance(source_, (np.memmap, np.ndarray, list, tuple)):
        return True
    else:
        return False


def as_source(source_, slicing=None, *args, **kwargs):
    """
    Convert source specification to a Source class.

    Arguments
    ---------
    source_ : object
        The source specification.

    Returns
    -------
    source : Source class
        The source class.
    """
    if isinstance(source_, pathlib.Path):
        source_ = str(source_)

    if not isinstance(source_, src.Source):
        mod = source_to_module(source_)
        source_ = mod.Source(source_, *args, **kwargs)
    if slicing is not None:
        source_ = slc.Slice(source=source_, slicing=slicing)
    return source_


def source(source_, slicing=None, *args, **kwargs):
    """
    Convert source specification to a Source class.

    Arguments
    ---------
    source_ : object
        The source specification.

    Returns
    -------
    source : Source class
        The source class.
    """
    return as_source(source_, slicing=slicing, *args, **kwargs)


def ndim(source_):
    """
    Returns number of dimensions of a source.

    Arguments
    ---------
    source_ : str, array or Source
        The source specification.

    Returns
    -------
    ndim : int
        The number of dimensions in the source.
    """
    source_ = as_source(source_)
    return source_.ndim


def shape(source_):
    """
    Returns shape of a source.

    Arguments
    ---------
    source_: str, array or Source
       The source specification.

    Returns
    -------
    shape : tuple of ints
       The shape of the source.
    """
    source_ = as_source(source_)
    return source_.shape


def size(source_):
    """
    Returns size of a source.

    Arguments
    ---------
    source_ : str, array or Source
        The source specification.

    Returns
    -------
    size : int
        The size of the source.
    """
    source_ = as_source(source_)
    return source_.size


def dtype(source_):
    """
    Returns dtype of a source.

    Arguments
    ---------
    source_ : str, array or Source
        The source specification.

    Returns
    -------
    dtype : dtype
        The data type of the source.
    """
    source_ = as_source(source_)
    return source_.dtype


def order(source_):
    """
    Returns order of a source.

    Arguments
    ---------
    source_ : str, array or Source
        The source specification.

    Returns
    -------
    order : 'C', 'F', or None
        The order of the source data items.
    """
    source_ = as_source(source_)
    return source_.order


def location(source_):
    """
    Returns the location of a source.

    Arguments
    ---------
    source_ : str, array or Source
        The source specification.

    Returns
    -------
    location : str or None
        The location of the source.
    """
    source_ = as_source(source_)
    return source_.location


def memory(source_):
    """
    Returns the memory type of `source_`.

    Arguments
    ---------
    source_ : str, array or Source
        The source specification.

    Returns
    -------
    memory : str or None
        The memory type of the source.
    """
    if isinstance(source_, pathlib.Path):
        source_ = str(source_)

    if sma.is_shared(source_):
        return 'shared'


def element_strides(source_):
    """
    Returns the strides of the data array of a source.

    Arguments
    ---------
    source_ : str, array, dtype or Source
        The source specification.

    Returns
    -------
    strides : tuple of int
        The strides of the source.
    """
    try:
        source_ = as_source(source_)
        strides = source_.element_strides
    except Exception as e:
        raise ValueError(f'Cannot determine the strides for the source!; {e}')

    return strides


def buffer(source_):
    """
    Returns an io buffer of the data array of a source for use with e.g. cython.

    Arguments
    ---------
    source_ : source specification
      The source specification.

    Returns
    -------
    buffer : array or memmap
      A buffer to read and write data.
    """
    try:
        source_ = as_source(source_)
        buffer_ = source_.as_buffer()
    except Exception as e:
        raise ValueError(f'Cannot get a io buffer for the source!; {e}')

    return buffer_


# TODO: arg memory= to specify which kind of array is created, better use device=
# TODO: arg processes= in order to use ParallelIO -> can combine with buffer=
def read(source_, *args, **kwargs):
    """
    Read data from a data source.

    Arguments
    ---------
    source_ : str, pathlib.Path, array, Source class
       The source to read the data from.

    Returns
    -------
    data : array
        The data of the source.
    """
    if isinstance(source_, pathlib.Path):
        source_ = str(source_)
    mod = source_to_module(source_)
    return mod.read(source_, *args, **kwargs)


def write(sink, data, *args, **kwargs):
    """
    Write data to a data source.

    Arguments
    ---------
    sink : str, pathlib.Path, array, Source class
        The source to write data to.
    data : array
        The data to write to the sink.
    slicing : slice specification or None
        Optional sub-slice to write data to.

    Returns
    -------
    sink : str, array or Source class
        The sink to which the data was written.
    """
    if isinstance(sink, pathlib.Path):
        sink = str(sink)
    mod = source_to_module(sink)
    return mod.write(sink, as_source(data), *args, **kwargs)


def create(source_, *args, **kwargs):
    """
    Create a data source on disk.

    Arguments
    ---------
    source_ : str, pathlib.Path, array, Source class
        The source to write data to.

    Returns
    -------
    sink : str, array or Source class
       The sink to which the data was written.
    """
    if isinstance(source_, pathlib.Path):
        source_ = str(source_)
    mod = source_to_module(source_)
    return mod.create(source_, *args, **kwargs)


def initialize(source_=None, shape=None, dtype=None, order=None, location=None,
               memory=None, like=None, hint=None, **kwargs):
    """
    Initialize a source with specified properties.

    Note
    ----
    The source is created on disk or in memory if it does not exist so processes
    can start writing into it.

    Arguments
    ---------
    source_ : str, array, Source class
        The source to write data to.
    shape : tuple or None
        The desired shape of the source.
        If None, inferred from existing file or from the like parameter.
        If not None and source has a valid shape shapes are tested to match.
    dtype : type, str or None
        The desired dtype of the source.
        If None, inferred from existing file or from the like parameter.
        If not None and source has a valid dtype the types are tested to match.
    order : 'C', 'F' or None
        The desired order of the source.
        If None, inferred from existing file or from the like parameter.
        If not None and source has a valid order the orders are tested to match.
    location : str or None
        The desired location of the source.
        If None, inferred from existing file or from the like parameter.
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
    source: Source class
        The initialized source.
    """
    if isinstance(source_, pathlib.Path):
        source_ = str(source_)
    if isinstance(source_, (str, te.Expression)):
        location = source_
        source_ = None

    if like is not None:
        shape, dtype, order = _from_like(like, shape, dtype, order)

    if source_ is None:
        if location is None:
            shape, dtype, order = _from_hint(hint, shape, dtype, order)
            if memory in ['shared', 'automatic']:
                return sma.create(shape=shape, dtype=dtype, order=order, **kwargs)
            else:
                return npy.create(shape=shape, dtype=dtype, order=order)
        else:
            try:
                source_ = as_source(location)
            except (FileNotFoundError, ValueError) as err:  # TODO: see if nore exceptions are needed
                if isinstance(err, ValueError):
                    if not str(err).startswith('Cannot create memmap without shape at location'):
                        raise err
                try:
                    shape, dtype, order = _from_hint(hint, shape, dtype, order)
                    mod = location_to_module(location)
                    return mod.create(location=location, shape=shape, dtype=dtype, order=order, **kwargs)
                except Exception as error:
                    raise ValueError(f'Cannot initialize source for location {location}; {error}')

    if isinstance(source_, np.ndarray):
        source_ = as_source(source_)

    # ######## Exception handling ##############
    if not isinstance(source_, src.Source):
        raise ValueError(f'Source specification {source_} not a valid location, array or Source class!')

    current_vars = locals()
    for attr in ('shape', 'dtype', 'order'):
        if current_vars.get(attr) is not None and current_vars[attr] != getattr(source_, attr, None):
            raise IncompatibleSource(source_, attr, current_vars)

    if location is not None and abspath(location) != abspath(source_.location):
        raise IncompatibleSource(source_, 'location', current_vars)
    if memory == 'shared' and not sma.is_shared(source_):
        raise ValueError(f'Incompatible memory type, the source {source_} is not shared!')

    return source_


def _from_like(like, shape, dtype, order):
    if like is not None:
        like = as_source(like)
        if shape is None:
            shape = like.shape
        if dtype is None:
            dtype = like.dtype
        if order is None:
            order = like.order
    return shape, dtype, order


def _from_hint(hint, shape, dtype, order):
    """Helper for initialize."""
    try:
        return _from_like(hint, shape, dtype, order)
    except Exception as err:
        warnings.warn(f'Cannot infer shape, dtype and order from hint {hint}, keeping defaults; {err}')
        return shape, dtype, order
  

def initialize_buffer(source_, shape=None, dtype=None, order=None, location=None, memory=None, like=None, **kwargs):
    """
    Initialize a buffer with specific properties.

    Arguments
    ---------
    source_ : str, array, Source class
        The source to write data to.
    shape : tuple or None
        The desired shape of the source.
        If None, inferred from existing file or from the like parameter.
        If not None and source has a valid shape shapes are tested to match.
    dtype : type, str or None
        The desired dtype of the source.
        If None, inferred from existing file or from the like parameter.
        If not None and source has a valid dtype the types are tested to match.
    order : 'C', 'F' or None
        The desired order of the source.
        If None, inferred from existing file or from the like parameter.
        If not None and source has a valid order the orders are tested to match.
    location : str or None
        The desired location of the source.
        If None, inferred from existing file or from the like parameter.
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
    The buffer is created if it does not exist.
    """
    source_ = initialize(source_, shape=shape, dtype=dtype, order=order, location=location, memory=memory, **kwargs)
    return source_.as_buffer()


###############################################################################
# ## Utils
###############################################################################

def file_list(expression=None, file_list=None, sort=True, verbose=False):
    """
    Returns the list of files that match the tag expression.

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


def get_info(d_type):
    """
    Get the numpy info object for a data type. (automatically determines if integer or float)

    Parameters
    ----------
    d_type: dtype
        The data type to get the info for.

    Returns
    -------
    info: numpy info object
        The info object for the data type.
    """
    try:
        return np.iinfo(d_type)
    except ValueError:
        return np.finfo(d_type)


def get_value(source_, value_type):  # REFACTOR: should be moved to io_utils or Source module
    """
    Get the minimal or maximal value of a source data type.

    Parameters
    ----------
    source_: str, array, dtype or Source
        The source specification.
    value_type: str
        The value type to get, either 'min' or 'max'.

    Returns
    -------
    value: number
        The value of the data type.
    """
    if isinstance(source_, pathlib.Path):
        source_ = str(source_)

    if value_type not in ['min', 'max']:
        raise ValueError(f'Unknown value type {value_type}, accepted arguments are "min" and "max"!')

    if isinstance(source_, (src.Source, np.ndarray)):
        source_ = source_.dtype

    if isinstance(source_, str):
        try:
            source_ = np.dtype(source_)
        except TypeError:
            pass

    if not isinstance(source_, (type, np.dtype)):
        source_ = dtype(source_)

    try:
        info = get_info(source_)
        return getattr(info, value_type)
    except ValueError as e:
        raise ValueError(f'Cannot determine the {value_type} value for the type {source_}!; {e}')


def min_value(source_):
    """
    Returns the minimal value of a source data type.

    Arguments
    ---------
    source_ : str, array, dtype or Source
        The source specification.

    Returns
    -------
    min_value : number
        The minimal value for the data type of the source
    """
    return get_value(source_, 'min')


def max_value(source_):
    """
    Returns the maximal value of a source data type.

    Arguments
    ---------
    source_ : str, array, dtype or Source
        The source specification.

    Returns
    -------
    max_value : number
       The maximal value for the data type of the source
    """
    max_value = get_value(source_, 'max')
    return max_value


def convert(source_, sink, processes=None, verbose=False, **kwargs):
    """
    Transforms a source into another format.

    Arguments
    ---------
    source_ : source specification
        The source or list of sources.
    sink : source specification
        The sink or list of sinks.

    Returns
    -------
    sink : sink specification
        The sink or list of sinks.
    """
    if isinstance(sink, pathlib.Path):
        sink = str(sink)
    source_ = as_source(source_)
    if verbose:
        print(f'converting {source_} -> {sink}')
    mod = source_to_module(source_)
    if hasattr(mod, 'convert'):
        return mod.convert(source_, sink, processes=processes, verbose=verbose, **kwargs)
    else:
        return write(sink, source_)


def convert_files(filenames, extension=None, path=None, processes=None, verbose=False, workspace=None, verify=False):
    """
    Transforms list of files to their sink format in parallel.

    Arguments
    ---------
    filenames : list of str
        The filenames to convert
    extension : str
        The new file format extension.
    path : str or None
        Optional path specification.
    processes : int, 'serial' or None
        The number of processes to use for parallel conversion.
    verbose : bool
        If True, print progress information.

    Returns
    -------
    filenames : list of str
        The new file names.
    """
    if extension.startswith('.'):  # FIXME: downstream code should handle extension with or without dot
        extension = extension[1:]
    if not isinstance(filenames, (tuple, list)):
        filenames = [filenames]
    if len(filenames) == 0:
        return []
    n_files = len(filenames)

    if path is not None:
        filenames = [fu.join(path, fu.split(f)[1]) for f in filenames]  # TODO: replace with pathlib
    sinks = [str(pathlib.Path(f).with_suffix('.'+extension)) for f in filenames]

    if verbose:
        timer = tmr.Timer()
        print(f'Converting {n_files} files to {extension}!')

    if not isinstance(processes, int) and processes != 'serial':
        processes = mp.cpu_count()

    # print(n_files, extension, filenames, sinks)
    _convert = functools.partial(_convert_files, n_files=n_files, extension=extension, verbose=verbose, verify=verify)

    if processes == 'serial':
        [_convert(source_, sink, i) for i, source_, sink in zip(range(n_files), filenames, sinks)]
    else:
        with CancelableProcessPoolExecutor(processes) as executor:
            results = executor.map(_convert, filenames, sinks, range(n_files))
            if workspace is not None:
                workspace.executor = executor
            results = list(results)  # to catch exceptions
        if workspace is not None:
            workspace.executor = None

    if verbose:
        timer.print_elapsed_time(f'Converting {n_files} files to {extension}')

    return sinks


@ptb.parallel_traceback
def _convert_files(source_, sink, fid, n_files, extension, verbose, verify=False):
    source_ = as_source(source_)
    if verbose:
        print(f'Converting file {fid}/{n_files} {source_} -> {sink}')
    mod = file_extension_to_module[extension]
    if mod is None:
        raise ValueError(f"Cannot determine module for extension {extension}!")
    mod.write(sink, source_)
    if verify:
        src_mean = source_.array.mean()
        sink_mean = mod.read(sink).mean()
        if not math.isclose(src_mean, sink_mean, rel_tol=1e-5):
            raise RuntimeError(f"Conversion of {source_} to {sink} failed, means differ")


###############################################################################
# ## Helpers
###############################################################################

_shape = shape
_dtype = dtype
_order = order
_location = location
_memory = memory


###############################################################################
# ## Tests
###############################################################################
def _test():
    import ClearMap.IO.IO as io

    print(io.abspath('.'))
    # reload(io)


# TODO:
#
# def copy(source, sink):
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
# def convert(source, sink, **args):
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
#      raise RuntimeError('convert: unknown source format!');
    
#
###############################################################################
# # Other
###############################################################################
#
# def writeTable(filename, table):
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

# Temp

# def isFileExpression(source):
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
# def isDataFile(source, exists = False):
#    """Checks if a file has a valid data file extension usable in *ClearMap*
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
# def isImageFile(source, exists = False):
#    """Checks if a file has a valid image file extension usable in *ClearMap*
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
# def isArrayFile(source, exists =False):
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
# def isDataSource(source, exists = False):
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
