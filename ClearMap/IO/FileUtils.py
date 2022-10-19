# -*- coding: utf-8 -*-
"""
FileUtils
=========

This module provides utilities for file management used by various IO modules.

See also
--------
:mod:`ClearMap.IO`.
"""
__author__ = 'Christoph Kirst <christoph.kirst.ck@gmail.com>'
__license__ = 'GPLv3 - GNU General Pulic License v3 (see LICENSE.txt)'
__copyright__ = 'Copyright © 2020 by Christoph Kirst'
__webpage__ = 'https://idisco.info'
__download__ = 'https://www.github.com/ChristophKirst/ClearMap2'

import os
import shutil
import importlib

__all__ = ['is_file', 'is_directory', 'file_extension', 'join', 'split',
           'abspath', 'create_directory', 'delete_directory',
           'copy_file', 'delete_file']

##############################################################################
# ## Basic file queries
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
        return False

    if os.path.isdir(filename):
        return False

    return os.path.exists(filename)

     
def is_directory(dirname):
    """
    Checks if a directory exists.

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
        return False

    return os.path.isdir(dirname)


##############################################################################
# ## File name manipulation
##############################################################################

def file_extension(filename):
    """
    Returns the file extension of a file

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
        return

    fext = filename.split('.')  # FIXME: use os.path.splitext(...)[-1][1:]  # To remove leading dot
    if len(fext) < 2:
        return
    else:
        return fext[-1]


def join(path, filename):
    """
    Joins a path to a file name.

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
    # TODO: correct to allow joining '/foo' with '/bar' to /foo/bar (os gives /bar!)
    if len(filename) > 0 and filename[0] == '/':
        filename = filename[1:]

    return os.path.join(path, filename)


def split(filename):
    """
    Splits a file name into it's path and name.

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
    return os.path.split(filename)


def abspath(filename):
    """
    Returns the filename using the full path specification.

    Arguments
    ---------
    filename : str
        The file name.

    Returns
    -------
    filename : str
        The full file name.
    """
    return os.path.abspath(filename)


##############################################################################
# ## File manipulation
##############################################################################

def create_directory(filename, split=True):
    """
    Creates the directory of the file name if it does not exists.

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
        path, name = os.path.split(filename)
    else:
        path = filename

    if not is_directory(path):
        os.makedirs(path)

    return path


def delete_directory(filename, split=False):
    """
    Deletes a directory of the filename if it exists.

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
        path, name = os.path.split(filename)
    else:
        path = filename

    if is_directory(path):
        shutil.rmtree(path)


def delete_file(filename):
    """
    Deletes a file.

    Arguments
    ---------
    filename : str
        Filename to delete.
    """
    if is_file(filename):
        os.remove(filename)

    
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
        path, name = os.path.split(source)
        sink = os.path.join(sink, name)
    shutil.copy(source, sink)
    return sink


def uncompress(file_path, extension='zip', check=True, verbose=False):
    """
    Unzips a file only if 1) the file does not exist (check), 2) the compressed file exists.

    Arguments
    ---------
    file_path : str
        The file path to search for.
    extension : str
        The extension for the compressed file.
    check : bool
        If True, check if the decompressed file already exists.
    verbose : bool
        Print progress info.

    Returns
    -------
    filename : str or None
        The uncompressed filename or None if failed.
    """
    if not os.path.exists(file_path) or not check:
        if extension == 'auto':
            for algo in ('zip', 'bz2', 'gzip', 'lzma'):
                f_path_w_ext = f'{file_path}.{algo}'
                if os.path.exists(f_path_w_ext):
                    extension = algo
                    break
            else:
                raise ValueError(f'Could not find compressed source for {file_path}')

        compressed_path = f'{file_path}.{extension}'
        if os.path.exists(compressed_path):
            if verbose:
                print(f'Decompressing source: {compressed_path}')
            if extension == 'zip':
                import zipfile
                try:
                    with zipfile.ZipFile(compressed_path, 'r') as zipf:
                        if os.path.splitext(file_path)[-1] in ('.tif', '.nrrd'):
                            zipf.extract(os.path.basename(file_path), path=os.path.dirname(compressed_path))
                        else:
                            zipf.extractall(path=os.path.dirname(compressed_path))
                    if not os.path.exists(file_path):
                        raise FileNotFoundError
                except Exception as err:  # FIXME: TOO broad
                    print(err)
                    return
            elif extension in ('bz2', 'gzip', 'lzma'):
                mod = importlib.import_module(extension)
                with open(file_path, 'wb') as out, \
                        open(compressed_path, 'rb') as compressed_file:
                    out.write(mod.decompress(compressed_file.read()))
            else:
                raise NotImplementedError(f'Unrecognized compression extension {extension}')
        else:
            print(f'Cannot find compressed source: {compressed_path}')
    return file_path
    

###############################################################################
# ## Tests
###############################################################################

def test():
    import ClearMap.IO.FileUtils as fu
    reload(fu)

    filename = fu.__file__
    path, name = fu.os.path.split(filename)

    fu.is_file(filename), fu.is_directory(filename)
    fu.file_extension(filename)
