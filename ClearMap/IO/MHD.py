# -*- coding: utf-8 -*-
"""
MHD
===

IO interface to mhd / raw files.

Note
----
Using the function :func:`write_header_from_source` a mhd header for a numpy
binary array can be created to enable loading the numpy file into imagej.
"""
__author__    = 'Christoph Kirst <christoph.kirst.ck@gmail.com>'
__license__   = 'GPLv3 - GNU General Pulic License v3 (see LICENSE.txt)'
__copyright__ = 'Copyright © 2020 by Christoph Kirst'
__webpage__   = 'http://idisco.info'
__download__  = 'http://www.github.com/ChristophKirst/ClearMap2'

import os
import numpy as np


###############################################################################
### Source classe
###############################################################################

#TODO: implement sources !


###############################################################################
### Utils
###############################################################################

dtype_to_mtype = {np.dtype('int8')    : "MET_CHAR",
                  np.dtype('uint8')   : "MET_UCHAR",
                  np.dtype('int16')   : "MET_SHORT", 
                  np.dtype('uint16')  : "MET_USHORT",
                  np.dtype('int32')   : "MET_INT",
                  np.dtype('uint32')  : "MET_UINT",
                  np.dtype('int64')   : "MET_LONG",
                  np.dtype('uint64')  : "MET_ULONG",
                  np.dtype('float32') : "MET_FLOAT", 
                  np.dtype('float64') : "MET_DOUBLE"
                 }   

mtype_to_dtype = {v : k for v,k in dtype_to_mtype.items()};

tag_order = [
     'Comment',
     'ObjectType',
     'TransformType',
     'NDims',
     'BinaryData',
     'ElementByteOrderMSB',
     'BinaryDataByteOrderMSB',
     'Color',
     'Position',
     'Offset',
     'Orientation',
     'AnatomicalOrientation',
     'TransformMatrix',
     'CenterOfRotation',
     'CompressedData',
     'CompressedDataSize',
     'DimSize',
     'HeaderSize',
     'Modality',
     'SequenceID',
     'ElementMin',
     'ElementMax',
     'ElementNumberOfChannels',
     'ElementSize',
     'ElementType',
     'ElementSpacing',
     'ElementDataFile'
     ]


def write_header(filename, mhd_header):
  """Write raw header mhd file.
  
  Arguments
  ---------
  filename : str
    Filename of header.
  mhd_header : dict
   Dictionary of meta data.
  
  Returns
  -------
  filename : str
    The filename of the mhd header.
  """
  header = ''
  for tag in tag_order:
    if tag in mhd_header.keys():
      header += '%s = %s\n'%(tag, mhd_header[tag])
  f = open(filename,'w')
  f.write(header)
  f.close()
  
  return filename;


def header_from_source(source, header = None):
  """Create a mhd header file for a source file.
  
  Arguments
  ---------
  source : Source specification
    Source file or class to create a mhd header file for.
  filename : str or None
    Filename of the mhd file. If None, the source location with extension 'mhd' is used.
  header : dict or None
    Optional additional entries for the header file.
  
  Returns
  -------
  filename : str
    The filename of the mhd header.
  """
  
  import ClearMap.IO.IO as io

  source = io.as_source(source);
  
  if not isinstance(source, (io.mmp.Source,)):
   raise ValueError('The source %r is not valid for mhd header creation!' % (source));                     
   
  if source.order not in ['C', 'F']:
    raise ValueError('The source %r is not contiguous!');
                    
  if not source.dtype in dtype_to_mtype.keys():
    raise ValueError('Data type %d of source %r not valid for mhd file format!' % (source.dtype, source));
  mtype = dtype_to_mtype[source.dtype];
  
  #construct the header info                     
  mhd_header = {}
  
  #generic
  mhd_header['ObjectType'] = 'Image'
  mhd_header['BinaryData'] = 'True'
  mhd_header['BinaryDataByteOrderMSB'] = 'False'
  
  #shape
  mhd_header['NDims'] = '%d' % source.ndim;
          
  shape = source.shape;
  if source.order == 'C':
    shape = shape[::-1];
  mhd_header['DimSize'] = ' '.join(['%d' % s for s in shape])
  
  #data
  mhd_header['ElementType'] = mtype;          
  mhd_header['HeaderSize'] = '%d' % source.offset;            
  mhd_header['ElementDataFile'] = os.path.split(os.path.abspath(source.location))[-1];

  if isinstance(header, dict):
    mhd_header.update(header);
    
  return mhd_header;


def write_header_from_source(source, filename = None, header = None):
  """Create a mhd header file for a source file.
  
  Arguments
  ---------
  source : Source specification
    Source file or class to create a mhd header file for.
  filename : str or None
    Filename of the mhd file. If None, the source location with extension 'mhd' is used.
  header : dict or None
    Optional additional entries for the header file.
  
  Returns
  -------
  filename : str
    The filename of the mhd header.
  """
  import ClearMap.IO.IO as io
  source = io.as_source(source);
  
  mhd_header = header_from_source(source, header=header);
                              
  if filename is None:
    filename = source.location + '.mhd'         
              
  return write_header(filename, mhd_header);
    

#TODO: cleanup, swtich to sources
def write_raw(filename, data):
  """Write the data into a raw format file.

  Arguments
  ---------
  filename : str
    The file name as regular expression.
  data : array
    The data to write in to a raw file.
  
  Returns
  -------
  filename: str
    The file name of raw file
  """
  rawfile = open(filename,'wb');
  d = len(data.shape);
  if d <= 2:
      #data.tofile(rawfile);
      data.transpose([1,0]).tofile(rawfile);
  elif d == 3:
      #data.transpose([2,0,1]).tofile(rawfile);
      data.transpose([2,1,0]).tofile(rawfile);
  elif d== 4:
      #data.transpose([3,2,0,1]).tofile(rawfile);
      data.transpose([3,2,1,0]).tofile(rawfile);
  else:
      raise RuntimeError('writeRawData: image dimension %d not supported!' % d);
  
  rawfile.close();
  
  return filename;


def write(filename, source, header = None, **kwargs):
  """Write  data into to raw/mhd file pair

  Arguments
  ---------
  filename : str
    The file name of the raw/mhd file.
  source : source specification
    The source to write as mhd/raw file.
  
  Returns
  -------
  filename : str
    The filename of the mhd file.
  """ 
  import ClearMap.IO.IO as io
  fext = io.file_extension(filename);
  if fext == "raw":
    header_name = filename[:-3] + 'mhd';
    raw_name = filename;                         
  elif fext == 'mhd':
    header_name = filename;
    raw_name = filename[:-3] + 'raw';
  else:
    header_name = filename + '.mhd';
    raw_name = filename + '.raw';                         
  
  hdm_header = header_from_source(source, header=header);
                               
  write_header(header_name, hdm_header)
  write_raw(raw_name, source)
  
  return header_name;




###############################################################################
### Tests
###############################################################################

def _test():
  import numpy as np
  import ClearMap.IO.MHD as mhd
  
  data = np.array(255* np.random.rand(200,500,20), order='C');
  data = np.array(data, dtype = 'uint8');
  
  data[:] = 0;
  data[10:25,5:10,3:15] = 255;
  
  fname = 'test.npy'
  np.save(fname, data);

  h = mhd.header_from_source(fname, header = None)         
  
  hname = mhd.write_header_from_source(fname)
