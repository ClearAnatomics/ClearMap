# -*- coding: utf-8 -*-
"""
FileUtils
=========

This module provides utilities for file management used by various IO modules.

See also
--------
:mod:`ClearMap.IO`.
"""
__author__    = 'Christoph Kirst <christoph.kirst.ck@gmail.com>'
__license__   = 'GPLv3 - GNU General Pulic License v3 (see LICENSE.txt)'
__copyright__ = 'Copyright © 2020 by Christoph Kirst'
__webpage__   = 'http://idisco.info'
__download__  = 'http://www.github.com/ChristophKirst/ClearMap2'

import os
import shutil

__all__ = ['is_file', 'is_directory', 'file_extension', 'join', 'split', \
           'abspath', 'create_directory', 'delete_directory', \
           'copy_file', 'delete_file']

##############################################################################
### Basic file queries
##############################################################################

def is_file(filename):
  """Checks if a file exists.
  
  Arguments
  ---------
  filename : str
    The file name to check if it exists.
      
  Returns
  -------
  is_file : bool
    True if filename exists on disk and is not a directory.   
  """
  if not isinstance(filename, str):
    return False;

  if os.path.isdir(filename):
    return False;

  return os.path.exists(filename);

     
def is_directory(dirname):
  """Checks if a directory exsits.
  
  Arguments
  ---------
  dirname : str
    The directory name.
      
  Returns
  -------
  is_directory : bool
    True if source is a real file. 
  """
  if not isinstance(dirname, str):
    return False;

  return os.path.isdir(dirname);


##############################################################################
### File name manipulation
##############################################################################

def file_extension(filename):
  """Returns the file extension of a file
  
  Arguments
  ---------
  filename : str
    The file name.
    
  Returns
  -------
  extension : str
    The file extension or None if it does not exists.
  """
  if not isinstance(filename, str):
    return None;
  
  fext = filename.split('.');
  if len(fext) < 2:
      return None;
  else:
      return fext[-1];


def join(path, filename):
  """Joins a path to a file name.
  
  Arguments
  ---------
  path : str
    The path to append a file name to.
  filename : str
    The file name.
    
  Returns
  -------
  filename : str
    The full file name.
  """  
  #correct to allow joining '/foo' with '/bar' to /foo/bar (os gives /bar!)
  if len(filename) > 0 and filename[0] == '/':
    filename = filename[1:];
  
  return os.path.join(path, filename);


def split(filename):
  """Splits a file name into it's path and name.
  
  Arguments
  ---------
  filename : str
    The file name.
    
  Returns
  -------
  path : str
    The path of the file.
  filename : str
    The file name.
  """
  return os.path.split(filename);


def abspath(filename):
  """Returns the filename using the full path specification.
  
  Arguments
  ---------
  filename : str
    The file name.
    
  Returns
  -------
  filename : str
    The full file name.
  """ 
  return os.path.abspath(filename);


##############################################################################
### File manipulation
##############################################################################

def create_directory(filename, split = True):
  """Creates the directory of the file name if it does not exists.
   
  Arguments
  ---------
  filename : str
    The name to create the directory from.
  split : bool
    If True, split the filename first.
      
  Returns
  -------
  directory : str
    The directory name.
  """      
  if split:
    path, name = os.path.split(filename);
  else:
    path = filename;    
  
  if not is_directory(path):
    os.makedirs(path);
  
  return path;


def delete_directory(filename, split = False):
  """Deletes a directory of the filename if it exists.
   
  Arguments
  ---------
  filename : str
    The name to create the directory from.
  split : bool
    If True, split the filename first.
      
  Returns
  -------
  directory : str
    The directory name.
  """      
  if split:
    path, name = os.path.split(filename);
  else:
    path = filename;    
  
  if is_directory(path):
    shutil.rmtree(path);


def delete_file(filename):
  """Deletes a file.
   
  Arguments
  ---------
  filename : str
    Filename to delete.
  """
  if is_file(filename):
    os.remove(filename);

    
def copy_file(source, sink):
  """Copy a file.
  
  Arguments
  ---------
  source : str
    Filename of the file to copy.
  sink : str
    File or directory name to copy the file to.
  
  Returns
  -------
  sink : str
    The name of the copied file.
  """ 
  if is_directory(sink):
    path, name = os.path.split(source);
    sink = os.path.join(sink, name);
  shutil.copy(source, sink);
  return sink;




###############################################################################
### Tests
###############################################################################

def uncompress(filename, extension = 'zip', check = True, verbose = True):
  """Unzips a file if only the compressed zip file exists.
  
  Arguments
  ---------
  filename : str
    The filename to search for.
  extension : str
    The extension for the compressed file.
  check : bool
    If True, check if the uncomressed file already exists.
  verbose : bool
    Print progrss info.
  
  Returns
  -------
  filename : str or None
     The uncompressed filename or None if failed.
  """
  try:
    if not os.path.exists(filename) or not check:
      zip_filename = filename + "." + extension;
      if os.path.exists(zip_filename):
        if verbose:
          print('Uncompressing source: %r' % zip_filename);
        import zipfile
        with zipfile.ZipFile(zip_filename, 'r') as zipf:
          zipf.extractall(path=os.path.split(zip_filename)[0]);
      else:
        if verbose:
          print('Cannot find compressed source: %r' % zip_filename)
    return filename;
  except:
    return None;
    

###############################################################################
### Tests
###############################################################################

def test():
  import ClearMap.IO.FileUtils as fu
  reload(fu)
  
  filename = fu.__file__;
  path, name = fu.os.path.split(filename)
  
  fu.is_file(filename), fu.is_directory(filename);
  fu.file_extension(filename);
  
  
  
  
  