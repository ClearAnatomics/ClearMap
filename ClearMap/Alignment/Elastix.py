# -*- coding: utf-8 -*-
"""
Elastix
=======

Interface to Elastix for alignment of volumetric data.

The elastix documentation can be found `here <http://elastix.isi.uu.nl/>`_.

In essence, a transformation :math:`T(x)` is sought so that for a fixed image 
:math:`F(x)` and a moving image :math:`M(x)`:

.. math::
    F(x) = M(T(x))

Once the map :math:`T` is estimated via elastix, transformix maps an image
:math:`I(x)` from the moving image frame to the fixed image frame, i.e.:

.. math::
    I(x) \\rightarrow I(T(x)) 

To register an image onto a reference image, the fixed image is typically 
choosed to be the image to be registered, while the moving image is the 
reference image. In this way an object identified in the data at position x
is mapped via transformix as:

.. math::
    x \\rightarrow T(x)

Summary
-------
    * elastix finds a transformation :math:`T: \\mathrm{fixed image} \\rightarrow \\mathrm{moving image}`
    * the fixed image is image to be registered
    * the moving image is typically the reference image
    * the result folder may contain an image (mhd file) that is :math:`T^{-1}(\\mathrm{moving})`,
      i.e. has the size of the fixed image
    * transformix applied to data gives :math:`T^{-1}(\\mathrm{data})` !
    * transformix applied to points gives :math:`T(\\mathrm{points})` !
    * point arrays are assumed to be in (x,y,z) coordinates consistent with (x,y,z) array represenation of images in ClearMap
    
Main routines are: :func:`align`, :func:`transform_data` and :func:`transform_points`.
    
See Also
--------
    `Elastix documentation <http://elastix.isi.uu.nl/>`_
    :mod:`~ClearMap.Alignment.Resampling`
"""
__author__    = 'Christoph Kirst <ckirst@rockefeller.edu>'
__license__   = 'MIT License <http://www.opensource.org/licenses/mit-license.php>'
__copyright__ = 'Copyright (c) 2019 by Christoph Kirst'


import os
import tempfile
import shutil
import re

import numpy as np

import multiprocessing as mp

import ClearMap.IO.IO as io
import ClearMap.Settings as settings


##############################################################################
### Initialization and Settings
##############################################################################

elastix_binary = None;
"""The elastix executable.

Notes
-----
  - setup in :func:`initialize_elastix`
"""

elastix_lib = None;
"""Path to the elastix library.

Notes
-----
  - setup in :func:`initialize_elastix`
"""

transformix_binary = None;
"""The transformix executable.

Notes
-----
  - setup in :func:`initialize_elastix`
"""
    
initialized = False;
"""Status of the elastixs binarys and paths.

Notes
-----
  - setup in :func:`initialize_elastix`
"""

    
def print_settings():
  """Prints the current elastix configuration.
  
  See also
  --------
  :const:`elastix_binary`, :const:`elastix_lib`, :const:`transformix_binary`, :const:`initialized`
  """
  
  if initialized:
    print("elastix_binary     = %s" % elastix_binary);
    print("elastix_lib        = %s" % elastix_lib);
    print("transformix_binary = %s" % transformix_binary);
  else:
    print("Elastix not initialized");


def set_elastix_library_path(path = None): 
  """Add elastix library path to the LD_LIBRARY_PATH variable in linux.
  
  Arguments
  ---------
  path : str or None
    Path to elastix root directory. 
    If None :const:`ClearMap.Settings.elastix_path` is used.
  """
   
  if path is None:
    path = settings.elastix_path;
  
  if 'LD_LIBRARY_PATH' in os.environ.keys():
    lp = os.environ['LD_LIBRARY_PATH'];
    if not path in lp.split(':'):
      os.environ['LD_LIBRARY_PATH'] = lp + ':' + path;
  else:
    os.environ['LD_LIBRARY_PATH'] = path


def initialize_elastix(path = None):
  """Initialize all paths and binaries of elastix

  Arguments
  ---------
    path (str or None): path to elastix root directory, if None 
    :const:`ClearMap.Settings.elastix_path` is used.
      
  See also
  --------
    :const:`elastix_binary`, :const:`elastix_lib`, :const:`transformix_binary`,
    :const:`initialized`, :func:`set_elastix_library_path`
  """
  global elastix_binary, elastix_lib, transformix_binary, initialized
  
  if path is None:
    path = settings.elastix_path;
    
  if path is None:
    raise RuntimeError("Cannot find elastix path!");
  
  #search for elastix binary
  elastixbin = os.path.join(path, 'bin/elastix');
  if os.path.exists(elastixbin):
    elastix_binary = elastixbin;
  else:
    raise RuntimeError("Cannot find elastix binary %s, set path in Settings.py accordingly!" % elastixbin);
  
  #search for transformix binarx
  transformixbin = os.path.join(path, 'bin/transformix');
  if os.path.exists(transformixbin):
    transformix_binary = transformixbin;
  else:
    raise RuntimeError("Cannot find transformix binary %s set path in Settings.py accordingly!" % transformixbin);
  
  #search for elastix libs
  elastix_lib = os.path.join(path, 'lib');
  if os.path.exists(elastix_lib):
    elastix_lib = elastix_lib;
  else:
    elastix_lib = os.path.join(path, 'bin');
    if os.path.exists(elastix_lib):
      elastix_lib = elastix_lib;
    else:
      raise RuntimeError("Cannot find elastix libs in %s  set path in Settings.py accordingly!" % elastix_lib);
  
  #set path
  set_elastix_library_path(elastix_lib);
      
  initialized = True;
  
  print("Elastix sucessfully initialized from path: %s" % path);
  
  return path;


initialize_elastix();


def check_elastix_initialized():
  """Checks if elastix is initialized
  
  Returns
  -------
  initialized : bool
    True if elastix paths are set.
  """
  if not initialized:
    raise RuntimeError("Elastix not initialized: run initialize_elastix(path) with proper path to elastix first");

  return True;


###############################################################################
### Basic interface routines
####################################txtfile####################################

def transform_file(result_directory):
  """Finds and returns the transformation parameter file.
  
  Arguments
  ---------
  result_directory : str
    Path to directory of elastix results.
      
  Returns
  -------
  filename : str
    File name of the first transformation parameter file.
      
  Notes
  -----
  In case of multiple transformation parameter files the top level file is 
  returned.     
  """    
  
  files = os.listdir(result_directory);
  files = [x for x in files if re.match('TransformParameters.\d.txt', x)];
  files.sort();
  
  if files == []:
    raise RuntimeError('Cannot find a valid transformation file in %r!' % result_directory);
  
  return os.path.join(result_directory, files[-1])


def transform_directory_and_file(transform_parameter_file = None, transform_directory = None):  
  """Determines transformation directory and file from either.
     
  Arguments
  ---------
  transform_parameter_file : str or None
    File name of the transformation parameter file.
  transform_directory : str or None
    Directory to the transformation parameter.
    
  Returns
  -------
  transform_parameter_file : str 
    File name of the transformation parameter file.
  transform_directory : str
    Directory to the transformation parameter.  
  
  Notes
  -----
  Only one of the two arguments need to be specified.
  """
  
  if transform_parameter_file == None:
    if transform_directory == None:
      raise ValueError('Neither the alignment directory nor the transformation parameter file is specified!'); 
    transform_parameter_dir  = transform_directory
    transform_parameter_file = transform_file(transform_parameter_dir);
  else:
    transform_parameter_dir  = os.path.split(transform_parameter_file);
    transform_parameter_dir  = transform_parameter_dir[0];
    transform_parameter_file = transform_parameter_file;
    
  return transform_parameter_dir, transform_parameter_file;


def set_path_transform_files(result_directory):
  """Replaces relative with absolute path in the parameter files in the result directory.
  
   Arguments
   ---------
   result_directory : str
     Path to directory of elastix results.

  Notes
  -----
  When elastix is not run in the directory of the transformation files
  the aboslute path needs to be given in each transformation file 
  to point to the subsequent transformation files. This is done via this 
  routine.
  """
  
  files = os.listdir(result_directory);
  files = [x for x in files if re.match('TransformParameters.\d.txt', x)];
  files.sort();
  
  if files == []:
    raise RuntimeError('Cannot find a valid transformation file in %r!' % result_directory);
  
  rec = re.compile("\(InitialTransformParametersFileName \"(?P<parname>.*)\"\)");
  
  for f in files:
    fh, tmpfn = tempfile.mkstemp();
    ff = os.path.join(result_directory, f);
    #print ff        
      
    with open(tmpfn, 'w') as newfile:
      with open(ff) as parfile:
        for line in parfile:
          #print line
          m = rec.match(line);
          if m != None:
            pn = m.group('parname');
            if pn != 'NoInitialTransform':
              pathn, filen = os.path.split(pn);
              filen = os.path.join(result_directory, filen);
              newfile.write(line.replace(pn, filen));
            else:
              newfile.write(line);
          else:
            newfile.write(line);
                          
    os.close(fh);
    os.remove(ff);
    shutil.move(tmpfn, ff);


def set_metric_parameter_file(parameter_file, metric):
  """Replaces the metric in the parameter file.
     
  Arguments
  ---------
  parameter_file : str
    The parameter file name.
  metric : str
    The metric to use.
      
  Notes
  -----
  Used to replace the metric when inverse transform is estimated.
  """
  fh, tmpfn = tempfile.mkstemp();     
  rec = re.compile("\(Metric \"(?P<parname>.*)\"\)");
  mset = False;
  
  with open(tmpfn, 'w') as newfile:
    with open(parameter_file) as parfile:
      for line in parfile:
        #print line
        m = rec.match(line);
        if m != None:
          pn = m.group('parname');
          newfile.write(line.replace(pn, metric));
          mset = True;
        else:
          newfile.write(line);
         
  if not mset:
    newfile.write("(Metric \"" + metric + "\")\n");
           
  os.close(fh);
  os.remove(parameter_file);
  shutil.move(tmpfn, parameter_file);


def result_data_file(result_directory):
  """Returns the mhd result file in a result directory.
  
  Arguments
  ---------
  result_directory : str
    Path to elastix result directory.
      
  Returns
  -------
  result_file : str
    The mhd file in the result directory.
  """
  files = os.listdir(result_directory);
  files = [x for x in files if re.match('.*.mhd', x)];
  files.sort();
  
  if files == []:
    raise RuntimeError('Cannot find a valid result data file in ' + result_directory);
  
  return os.path.join(result_directory, files[0])


def transform_file_size_and_spacing(transform_file):
  """Parse the image size and spacing from a transformation parameter file.

  Arguments
  ---------
  transform_file : str
    File name of the transformix parameter file.
      
  Returns
  -------
  size: tuple
    The image size.
  spacing : tuple
    The image spacing.
  """
  resi = re.compile("\(Size (?P<size>.*)\)");
  resp = re.compile("\(Spacing (?P<spacing>.*)\)");
  
  si = None;
  sp = None;
  with open(transform_file) as parfile:
    for line in parfile:
      m = resi.match(line)
      if m != None:
        pn = m.group('size');
        si = pn.split();
          
      m = resp.match(line);
      if m != None:
        pn = m.group('spacing');
        sp = pn.split();
  
      parfile.close();
  
  si = [float(x) for x in si];
  sp = [float(x) for x in sp];
  
  return si, sp

    
def set_transform_file_size_and_spacing(transform_file, size, spacing):
  """Replaces size and scale in the transformation parameter file
  
  Arguments
  ---------
  transform_file : str
    Transformation parameter file.
  size : tuple
    The new image size.
  spacing : tuple
    The new image spacing.
  """
  
  resi = re.compile("\(Size (?P<size>.*)\)");
  resp = re.compile("\(Spacing (?P<spacing>.*)\)");
  
  fh, tmpfn = tempfile.mkstemp();
  
  si = [int(x) for x in size];
  
  with open(transform_file) as parfile:        
    with open(tmpfn, 'w') as newfile:
      for line in parfile:
        m = resi.match(line)
        if m != None:
          newfile.write("(Size %d %d %d)" % si);
        else:
          m = resp.match(line)
          if m != None:
            newfile.write("(Spacing %d %d %d)" % spacing);
          else:
            newfile.write(line);
      
      newfile.close();               
      parfile.close();
      
      os.remove(transform_file);
      shutil.move(tmpfn, transform_file);
      

def rescale_size_and_spacing(size, spacing, scale):
  """Rescales the size and spacing
  
  Arguments
  ---------
  size : tuple
    The image size.
  spacing : tuple
    The image spacing.
  scale : tuple
    The scale factor.
  
  Returns
  -------
  size : tuple
    The scaled image size.
  spacing : tuple
    The scaled image spacing.
  """   

  si = [int(x * scale) for x in size];
  sp = spacing / scale;
  
  return si, sp


##############################################################################
### Elastix Runs
##############################################################################

def align(fixed_image, moving_image, affine_parameter_file, bspline_parameter_file = None, result_directory = None, processes = None):
  """Align images using elastix, estimates a transformation :math:`T:` fixed image :math:`\\rightarrow` moving image.
  
  Arguments
  ---------
  fixed_image : str
    Image source of the fixed image (typically the reference image).
  moving_image : str
    Image source of the moving image (typically the image to be registered).
  affine_parameter_file : str or None
    Elastix parameter file for the primary affine transformation.
  bspline_parameter_file : str or None
    Elastix parameter file for the secondary non-linear transformation.
  result_directory : str or None
    Elastic result directory.
  processes : int or None
    Number of threads to use.
      
  Returns
  -------
  result_directory : str
    Path to elastix result directory.
  """
  
  if processes is None:
    processes = mp.cpu_count()
  
  check_elastix_initialized();

  # result directory
  if result_directory == None:
    result_directory = tempfile.gettempdir();
  
  if not os.path.exists(result_directory):
    os.mkdir(result_directory);
  
  # run elastix
  if bspline_parameter_file is None:
    cmd = '%s -threads %d -m %s -f %s -p %s -out %s' % (elastix_binary, processes, moving_image, fixed_image, affine_parameter_file, result_directory);
  elif affine_parameter_file is None:
    cmd = '%s -threads %d -m %s -f %s -p %s -out %s' % (elastix_binary, processes, moving_image, fixed_image, bspline_parameter_file, result_directory);
  else:
    cmd = '%s -threads %d -m %s -f %s -p %s -p %s -out %s' % (elastix_binary, processes, moving_image, fixed_image, affine_parameter_file, bspline_parameter_file, result_directory);
  
  res = os.system(cmd);
  
  if res != 0:
    raise RuntimeError('align: failed executing: ' + cmd);
  
  return result_directory


def transform(source, sink = [], transform_parameter_file = None, transform_directory = None, result_directory = None):
  """Transform a raw data set to reference using the elastix alignment results.
  
  Arguments
  ---------
  source : str or array
    Image source to be transformed.
  sink : str, [] or None
    Image sink to save transformed image to. If [] return the default name 
    of the data file generated by transformix.
  transform_parameter_file : str or None
    Parameter file for the primary transformation. 
    If None, the file is determined from the transform_directory.
  transform_directory : str or None
    Result directory of elastix alignment. 
    If None the transform_parameter_file has to be given.
  result_directory : str or None
    The directorty for the transformix results.
      
  Returns
  -------
  transformed : array or st
    Array or file name of the transformed data.
      
  Note
  ----
  If the map determined by elastix is
  :math:`T: \\mathrm{fixed} \\rightarrow \\mathrm{moving}`, 
  transformix on data works as :math:`T^{-1}(\\mathrm{data})`.
  """
  check_elastix_initialized();  
  
  # image
  source = io.as_source(source);
  if isinstance(source, io.tif.Source):
    imgname = source.location;
    delete_image = None;
  else:
    imgname = os.path.join(tempfile.gettempdir(), 'elastix_input.tif');
    io.write(source, imgname);
    delete_image = imgname;

  # result directory
  delete_result_directory = None;
  if result_directory == None:
    resultdirname = os.path.join(tempfile.gettempdir(), 'elastix_output');
    delete_result_directory = resultdirname;
  else:
    resultdirname = result_directory;
     
  if not os.path.exists(resultdirname):
    os.makedirs(resultdirname);
  
  # tranformation parameter
  transform_parameter_dir, transform_parameter_file = transform_directory_and_file(transform_parameter_file = transform_parameter_file, transform_directory = transform_directory);
  
  set_path_transform_files(transform_parameter_dir);
 
  #transformix -in inputImage.ext -out outputDirectory -tp TransformParameters.txx
  cmd = '%s -in %s -out %s -tp %s' % (transformix_binary, imgname, resultdirname, transform_parameter_file);
  
  res = os.system(cmd);
  
  if res != 0:
    raise RuntimeError('transform_data: failed executing: ' + cmd);
  
  # read data and clean up
  if delete_image is not None:
      os.remove(delete_image);
  
  if sink == []:
    return result_data_file(resultdirname);
  elif sink is None:
    resultfile = result_data_file(resultdirname);
    result = io.read(resultfile);
  elif isinstance(sink, str):
    resultfile = result_data_file(resultdirname);
    result = io.convert(resultfile, sink);
  else:
    raise RuntimeError('transform_data: sink not valid!');
    
  if delete_result_directory is not None:
    shutil.rmtree(delete_result_directory);
  
  return result;


def deformation_field(sink = [], transform_parameter_file = None, transform_directory = None, result_directory = None):
  """Create the deformation field T(x) - x.
      
  Arguments
  ---------
  sink : str, [] or None
    Image sink to save the transformation field; if [] return the default name 
    of the data file generated by transformix.
  transform_parameter_file : str or None
    Parameter file for the primary transformation, if None, the file is 
    determined from the transform_directory.
  transform_directory : str or None
    Result directory of elastix alignment, if None the 
    transform_parameter_file has to be given.
  result_directory : str or None
    The directorty for the transformix results.
      
  Returns
  -------
  deformation_field : array or str
    Array or file name of the deformation field data.
      
  Note
  ----
  The map determined by elastix is 
  :math:`T \\mathrm{fixed} \\rightarrow \\mathrm{moving}`.
  """
  check_elastix_initialized();   
  
  # result directory
  delete_result_directory = None;
  if result_directory == None:
    resultdirname = os.path.join(tempfile.gettempdir(), 'elastix_output');
    delete_result_directory = resultdirname;
  else:
    resultdirname = result_directory;
      
  if not os.path.exists(resultdirname):
    os.makedirs(resultdirname);
     
  # setup transformation 
  transform_parameter_dir, transform_parameter_file = transform_directory_and_file(transform_parameter_file = transform_parameter_file, transform_directory = transform_directory); 
  set_path_transform_files(transform_parameter_dir);
 
  #transformix -in inputImage.ext -out outputDirectory -tp TransformParameters.txt
  cmd = '%s -def all -out %s -tp  %s' % (transformix_binary, resultdirname, transform_parameter_file)
  
  res = os.system(cmd);
  
  if res != 0:
    raise RuntimeError('deformation_field: failed executing: ' + cmd);
  
  # read result and clean up
  if sink == []:
    return result_data_file(resultdirname);
  elif sink is None:
    resultfile = result_data_file(resultdirname);
    result = io.read(resultfile);
  elif isinstance(sink, str):
    resultfile = result_data_file(resultdirname);
    result = io.convert(resultfile, sink);
  else:
    raise RuntimeError('deformation_field: sink not valid!');
      
  if delete_result_directory is not None:
    shutil.rmtree(delete_result_directory);
  
  return result;


def deformation_distance(deformation_field, sink = None, scale = None):
  """Compute the distance field from a deformation vector field.
  
  Arguments
  ---------
  deformation_field : str or array
    Source of the deformation field determined by :func:`deformation_field`.
  sink : str or None
    Image sink to save the deformation field to.
  scale : tuple or None
    Scale factor for each dimension, if None = (1,1,1).
      
  Returns
  -------
  deformation_distannce : array or st
    Array or file name of the deformation distance data.
  """
  
  deformation_field = io.read(deformation_field);
  
  df = np.square(deformation_field);
  if not scale is None:
      for i in range(3):
          df[:,:,:,i] = df[:,:,:,i] * (scale[i] * scale[i]);
  df = np.sqrt(np.sum(df, axis = 3));
  
  return io.write(sink, df);
    

###############################################################################
### Point transformations
###############################################################################

def write_points(filename, points, indices = False, binary = True):
  """Write points as elastix/transformix point file
  
  Arguments
  ---------
  filename : str
    File name of the elastix point file.
  points : array or str
    Source of the points.
  indices : bool
    Write as pixel indices or physical coordiantes.
  
  Returns
  -------
  filename: str
    File name of the elastix point file.
  """
  
  points = io.read(points);
  
  if binary:
    with open(filename, 'wb') as pointfile:
      if indices:
        np.array(1, dtype = np.int64).tofile(pointfile)
      else:
        np.array(0, dtype = np.int64).tofile(pointfile)
        
      num_points = np.array(len(points), dtype = np.int64);
      num_points.tofile(pointfile);

      points = np.asarray(points, dtype = np.double);
      points.tofile(pointfile);

      pointfile.close();        
      
  else:
    with open(filename, 'w') as pointfile:
      if indices:
        pointfile.write('index\n')
      else:
        pointfile.write('point\n')
    
      pointfile.write(str(points.shape[0]) + '\n');
      np.savetxt(pointfile, points, delimiter = ' ', newline = '\n', fmt = '%.5e')
      pointfile.close();
  
  return filename;


def read_points(filename, indices = False, binary = True):
  """Parses the output points from the output file of transformix
  
  Arguments
  ---------
  filename : str
    File name of the transformix output file.
  indices : bool
    If True return pixel indices otherwise float coordinates.
      
  Returns
  --------
  points : array
    The point coordinates.
  """
  
  if binary:
    with open(filename) as f:
      index = np.fromfile(f, dtype=np.int64, count = 1)[0];
      #print(index)
      if index == 0:
        indices = False;
      else:
        indices = True;
      
      num_points = np.fromfile(f, dtype=np.int64, count = 1)[0];
      #print(num_points)
      if num_points == 0:
        return np.zeros((0,3));
      
      points = np.fromfile(f, dtype = np.double);
      #print points.shape
      points = np.reshape(points, (num_points,3));
      
      f.close();
      
    return points;
  
  else: # text file
  
    with open(filename) as f:
      lines = f.readlines()
      f.close();
    
    num_points = len(lines);
    
    if num_points == 0:
      return np.zeros((0,3));
    
    points = np.zeros((num_points, 3));
    k = 0;
    for line in lines:
      ls = line.split();
      if indices:
        for i in range(0,3):
          points[k,i] = float(ls[i+22]);
      else:
        for i in range(0,3):
          points[k,i] = float(ls[i+30]);
      
      k += 1;
    
    return points;


def transform_points(source, sink = None, transform_parameter_file = None, transform_directory = None, indices = False, result_directory = None, temp_file = None, binary = True):
  """Transform coordinates math:`x` via elastix estimated transformation to :math:`T(x)`.

  Arguments
  ---------
  source : str
    Source of the points.
  sink : str or None
    Sink for transformed points.
  transform_parameter_file : str or None
    Parameter file for the primary transformation. 
    If None, the file is determined from the transform_directory.
  transform_directory : str or None
    Result directory of elastix alignment.
    If None the transform_parameter_file has to be given.
  indices : bool
    If True use points as pixel coordinates otherwise spatial coordinates.
  result_directory : str or None
    Elastic result directory.
  temp_file : str or None
    Optional file name for the elastix point file.
      
  Returns
  -------
  points : array or st
    Array or file name of transformed points.
      
  Note
  ----
  The transformation is from the fixed image coorindates to the moving 
  image coordiantes.
  """   
  check_elastix_initialized();    

  # input point file
  if temp_file == None:
    if binary:
      temp_file = os.path.join(tempfile.gettempdir(), 'elastix_input.bin');
    else:
      temp_file = os.path.join(tempfile.gettempdir(), 'elastix_input.txt');
  
  delete_point_file = None;
  if isinstance(source, str):
    if len(source) > 3 and source[-3:] in ['txt', 'bin']:
      if source[-3:] == 'txt':
        binary = False; 
      if source[-3] == 'bin':
        binary = True;
      pointfile = source;
    else:
      points = io.read(source);
      pointfile = temp_file;
      delete_point_file = temp_file;
      write_points(pointfile, points, indices = indices, binary = binary);
  elif isinstance(source, np.ndarray):
    pointfile = temp_file;
    delete_point_file = temp_file;
    write_points(pointfile, source, indices = indices, binary = binary);
  else:
    raise RuntimeError('transform_points: source not string or array!');
  #print(pointfile)
  
  # result directory
  if result_directory == None:
    outdirname = os.path.join(tempfile.gettempdir(), 'elastix_output');
    delete_result_directory = outdirname;
  else:
    outdirname = result_directory;
    delete_result_directory = None;
      
  if not os.path.exists(outdirname):
    os.makedirs(outdirname);
  
  #transform
  transform_parameter_dir, transform_parameter_file = transform_directory_and_file(transform_parameter_file = transform_parameter_file, transform_directory = transform_directory);
  set_path_transform_files(transform_parameter_dir);
  
  #run transformix   
  cmd = '%s -def %s -out %s -tp %s' % (transformix_binary, pointfile, outdirname, transform_parameter_file);
  print(cmd)
  
  res = os.system(cmd);
  
  if res != 0:
    raise RuntimeError('failed executing ' + cmd);
  
  # read data and clean up
  if delete_point_file is not None:
    os.remove(delete_point_file);
  
  #read data / file 
  if sink == []: # return sink as file name
    if binary:
      return os.path.join(outdirname, 'outputpoints.bin')
    else:
      return os.path.join(outdirname, 'outputpoints.txt')
  
  else:
    if binary:
      transpoints = read_points(os.path.join(outdirname, 'outputpoints.bin'), indices = indices, binary = True);
    else:
      transpoints = read_points(os.path.join(outdirname, 'outputpoints.txt'), indices = indices, binary = False); 
    
    if delete_result_directory is not None:
      shutil.rmtree(delete_result_directory);
  
  return io.write(sink, transpoints);

        
        
def inverse_transform(fixed_image, affine_parameter_file, bspline_parameter_file = None, transform_parameter_file = None, transform_directory = None, result_directory = None, processes = None):
  """Estimate inverse tranformation :math:`T^{-1}:` moving image :math:`\\rightarrow` fixed image.
  
  Arguments
  ---------
  fixed_image : str
    Image source of the fixed image (typically the reference image)
  affine_parameter_file : str
    The paramter file for the original affine transformation.
  bspline_parameter_file : str
    The paramter file for the original b-spline transformation.
  transform_directory : str
    Elastic result directory of the original transform.
  result_directory : str or None
    Elastic result directory of the inverse transform.
      
  Returns
  -------
  result_directory : str
    Path to elastix result directory.
  """
  
  check_elastix_initialized();
  
  # result directory
  if result_directory == None:
      result_directory = tempfile.getgettempdir()();
  
  if not os.path.exists(result_directory):
      os.mkdir(result_directory);
  
  # transformation files
  transform_parameter_dir, transform_parameter_file = transform_directory_and_file(transform_parameter_file = transform_parameter_file, transform_directory = transform_directory);    
  set_path_transform_files(transform_parameter_dir);
  
  #set metric of the parameter files
  if bspline_parameter_file is not None:
    _, bsplinefile = os.path.split(bspline_parameter_file);
    bsplinefile    = os.path.join(result_directory, bsplinefile);
    shutil.copyfile(bspline_parameter_file, bsplinefile);
    set_metric_parameter_file(bsplinefile, 'DisplacementMagnitudePenalty');
  else:
    bsplinefile = None;
    
  if affine_parameter_file is not None:
    _, affinefile = os.path.split(affine_parameter_file);
    affinefile    = os.path.join(result_directory, affinefile);
    shutil.copyfile(affine_parameter_file, affinefile);
    set_metric_parameter_file(affinefile, 'DisplacementMagnitudePenalty');
  else:
    affinefile = None;
  
  # run elastix
  if bsplinefile is None:
    cmd = '%s -threads %d -m %s -f %s -t0 %s -p %s -out %s' % (elastix_binary, processes, fixed_image, fixed_image, transform_parameter_file, affinefile,  result_directory);
  elif affinefile is None:
    cmd = '%s -threads %d -m %s -f %s -t0 %s -p %s -out %s' % (elastix_binary, processes, fixed_image, fixed_image, transform_parameter_file, bsplinefile, result_directory);
  else:
    cmd = '%s -threads %d -m %s -f %s -t0 %s -p %s -p %s -out %s' % (elastix_binary, processes, fixed_image, fixed_image, transform_parameter_file, affinefile, bsplinefile, result_directory);    
  
  res = os.system(cmd);
  
  if res != 0:
    raise RuntimeError('inverse_transform: failed executing: ' + cmd);
  
  return result_directory


###############################################################################
### Tests
###############################################################################


def _test():
  import os
  import numpy as np
  import ClearMap.Settings as settings
  import ClearMap.Alignment.Elastix as elx
  #reload(elx)
  
  path = os.path.join(settings.test_data_path, 'Elastix');
    
  result_directory = os.path.join(path, 'elastix_template_to_ref')  
    
  elx.transform_file(result_directory)
  
  shape = np.array([432, 512, 229])
  points = np.random.rand(30,3) * 0.25 * shape + 0.5 * shape;
  points = np.round(points);
  points = np.array(points, dtype = int)

  test_txt = elx.transform_points(points, transform_directory=result_directory, binary=False, indices=False);
  
  test = elx.transform_points(points, transform_directory=result_directory, binary=True, indices=False);

  print(np.allclose(test_txt, test))
