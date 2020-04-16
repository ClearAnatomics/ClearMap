# -*- coding: utf-8 -*-
"""
Parallel tools for writing and reading large data arrays.

This module provides parallel reading and writing routines for very large files.
"""
__author__    = 'Christoph Kirst <ckirst@rockefeller.edu>'
__license__   = 'MIT License <http://www.opensource.org/licenses/mit-license.php>'
__copyright__ = 'Copyright (c) 2017 by Christoph Kirst, The Rockefeller University, New York City'


import numpy as np
import multiprocessing as mp

import pyximport;
pyximport.install(setup_args={"include_dirs":np.get_include()}, reload_support=True)

import ClearMap.IO.IO as io
import ClearMap.IO.MMP as mmp
import ClearMap.IO.SMA as sma
import ClearMap.IO.Slice as slc
import ClearMap.IO.FileUtils as fu
import ClearMap.Utils.Timer as tmr

import ClearMap.ParallelProcessing.DataProcessing.ArrayProcessing as ap

import ClearMap.ParallelProcessing.DataProcessing.ArrayIOCode as code


###############################################################################
### Defaults
###############################################################################

default_processes = ap.default_processes;
"""Default number of processes to use

Note
----
Half of the cpu count might be faster.
"""

default_blocks_per_process = ap.default_blocks_per_process;
"""Default number of blocks per process to split the data.

Note
----
10 blocks per processor is a good choice.
"""

default_cutoff_size = ap.default_cutoff_size;
"""Default size of array below which ordinary numpy is used.

Note
----
Ideally test this on your machine for different array sizes.
"""


###############################################################################
### Read and write
###############################################################################

def read(filename, sink = None, slicing = None, as_shared = None, blocks = None, processes = None, verbose = False, **kwargs):
  """Read a large array into memory in parallel.
  
  Arguments
  ---------
  filename : str
    The filename of array to load.
  slicing : slice, tuple, or None
    if not None this specifies the slice to read.
  as_shared : bool
    If True, read into shared memory
  blocks : int or None
    number of blocks to split array into for parallel processing
  processes : None or int
    number of processes, if None use number of cpus
  verbose : bool
    print info about the file to be loaded
    
  Returns
  -------
  array : array
    The data as an array in memory.
  """
  if processes is None:
    processes = default_processes;
  if blocks is None:
    blocks = processes * default_blocks_per_process;
  
  #source info
  source = mmp.Source(filename);
  if slicing is not None:
    source = slc.Slice(source=source, slicing=slicing);
      
  shape, dtype, order, offset = source.shape, source.dtype, source.order, source.offset;
  
  if order not in ['C', 'F']:
      raise NotImplementedError('Cannot read in parallel from non-contigous source!');  
      #TODO: implement parallel reader with strides !
  
  if verbose:
    timer = tmr.Timer();
    print('Reading data from source of shape = %r, dtype = %r, order = %r, offset = %r' %(shape, dtype, order, offset)); 
  
  
  #use initialze  form IO !!
  #prepare outputs
  if as_shared:
    data = sma.empty(shape, dtype = dtype, order = order);
  else:
    data = np.empty(shape, dtype = dtype, order = order);
  
  d = data.reshape(-1, order = 'A');
  if dtype == bool:
    d = d.view('uint8');  
  
  code.read(data = d, filename = filename, offset = offset, blocks = blocks, processes = processes);
  
  if verbose:
    timer.print_elapsed_time(head = 'Reading data from %s' % filename);
           
  return data;


def write(filename, data, slicing = None, blocks = None, processes = None, verbose = False):
  """Write a large array to disk in parallel.
  
  Arguments
  ---------
  filename : str
    Filename of array to load.
  data : array
    Array to save to disk.
  blocks : int or None
    Number of blocks to split array into for parallel processing.
  processes : None or int
    Number of processes, if None use number of cpus.
  verbose : bool
    Print info about the file to be loaded.
    
  Returns
  -------
  filename : str 
      The filename of the numpy array on disk.
  """
  if processes is None:
    processes = mp.cpu_count();
  if blocks is None:
    blocks = processes * default_blocks_per_process;
  
  #data
  data = io.as_source(data);  
  
  
  #prepare sink 
  is_file = fu.is_file(filename)
  if slicing is not None and not is_file:
    raise ValueError('Cannot write to a slice to a non-existing file %s!' % filename);

  if slicing is None:
    #create file on disk via memmap
    fortran_order = 'F' == data.order;
    memmap = np.lib.format.open_memmap(filename, mode = 'w+', shape = data.shape, 
                                       dtype = data.dtype, fortran_order = fortran_order);
    memmap.flush();
    del(memmap);
  
  sink = mmp.Source(location=filename);
  if slicing is not None:
    sink = slc.Slice(source=sink, slicing=slicing);
  shape, dtype, order, offset = sink.shape, sink.dtype, sink.order, sink.offset;
    
  if (data.order != order):
    raise RuntimeError('Order of arrays do not match %r!=%r' % (data.order, order));    
  if order not in ['C', 'F']:
    raise NotImplementedError('Cannot read in parallel from non-contigous source!');  
    #TODO: implement parallel reader with strides !
  
  if verbose:
    timer = tmr.Timer();
    print('Writing data to sink of shape = %r, dtype = %r, order = %r, offset = %r' %(shape, dtype, order, offset)); 
  
  d = data.reshape(-1, order = 'A');
  if d.dtype == bool:
    d = d.view('uint8');
  
  code.write(data = d, filename = filename, offset = offset, blocks = blocks, processes = processes);
  
  if verbose:
    timer.print_elapsed_time(head = 'Writing data to %s' % filename);
           
  return filename;


###############################################################################
### Tests
###############################################################################

def _test():
  
  import numpy as np
  from ClearMap.Utils.Timer import Timer;
  import ClearMap.ParallelProcessing.ParallelIO as pio
  reload(pio)
  
  #dat = np.random.rand(2000,2000,1000) > 0.5;
  #dat = np.random.rand(1000,1000,500) > 0.5;
  #dat = np.random.rand(1000,1000,500);   
  dat = np.random.rand(200,300,400);  
  
  filename = 'test.npy';
  timer = Timer();
  np.save(filename, dat)
  timer.print_elapsed_time('Numpy saving data of size %d' % dat.size);
  
  filename2 = 'test2.npy';
  timer = Timer();
  pio.write(filename2, dat, processes = 4, blocks = 4, verbose = False)
  timer.print_elapsed_time('ParallelIO writing data of size %d' % dat.size);
  
  
  timer = Timer();
  dat2 = np.load(filename)
  timer.print_elapsed_time('Numpy loading data of size %d' % dat.size);
  print(np.all(dat == dat2))

  timer = Timer();
  dat3 = pio.read(filename, verbose = True);
  timer.print_elapsed_time('ParallelIO reading data of size %d' % dat2.size);
  print(np.all(dat3 == dat))
  
  pio.io.fu.delete_file(filename);
  pio.io.fu.delete_file(filename2);
  
  #speed up for large files 2000x2000x1000 on 24 core workstation
  #pio: elapsed time: 0:00:04.867
  #numpy: elapsed time: 0:00:27.982
  
  
  
  
    

  