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
__download__ = 'https://github.com/ClearAnatomics/ClearMap'

import os
import shutil
import importlib

__all__ = ['is_file', 'is_directory', 'file_extension', 'join', 'split',
           'abspath', 'create_directory', 'delete_directory',
           'copy_file', 'delete_file']

from pathlib import Path

from ClearMap.IO import IO as clearmap_io


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
    Unzips a file only if:
        1) the file does not exist (check)
        2) the compressed file exists.


    Arguments
    ---------
    file_path : str or pathlib.Path
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
    compression_algorithms = ('zip', 'bz2', 'gzip', 'lzma')
    file_path = Path(file_path)
    if file_path.exists() and file_path.suffix in compression_algorithms:
        return uncompress(file_path.with_suffix(''), extension=file_path.suffix, check=check, verbose=verbose)
    if not file_path.exists() or not check:
        if extension == 'auto':
            for algo in compression_algorithms:
                f_path_w_ext = file_path.with_suffix(f'{file_path.suffix}.{algo}')
                if f_path_w_ext.exists():
                    extension = algo
                    break
            else:
                raise ValueError(f'Could not find compressed source for {file_path}')

        compressed_path = file_path.with_suffix(f'{file_path.suffix}.{algo}')
        if compressed_path.exists():
            if verbose:
                print(f'Decompressing source: {compressed_path}')
            if extension == 'zip':
                import zipfile
                try:
                    with zipfile.ZipFile(compressed_path, 'r') as zipf:
                        if file_path.suffix in ('.tif', '.nrrd'):
                            zipf.extract(file_path.name, path=compressed_path.parent)
                        else:
                            zipf.extractall(path=compressed_path.parent)
                    if not file_path.exists():
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


def compress(file_path, extension='zip', check=True, verbose=False):
    """
    Compresses a file only if:
        1) the file exists (check)
        2) the compressed file does not exist.

    Arguments
    ---------
    file_path : str or pathlib.Path
        The file path to search for.
    extension : str
        The extension for the compressed file.
    check : bool
        If True, check if the compressed file already exists.
    verbose : bool
        Print progress info.

    Returns
    -------
    filename : Path or None
        The compressed filename or None if failed.
    """
    file_path = Path(file_path)
    if file_path.exists() and check:
        if extension == 'auto':
            extension = 'zip'
        compressed_path = file_path.with_suffix(f'{file_path.suffix}.{extension}')
        if not compressed_path.exists():
            if verbose:
                print(f'Compressing source: {file_path}')
            if extension == 'zip':
                import zipfile
                with zipfile.ZipFile(compressed_path, 'w', compression=zipfile.ZIP_DEFLATED) as zipf:
                    if file_path.suffix in ('.tif', '.nrrd'):
                        zipf.write(file_path, arcname=file_path.name)
                    else:
                        for root, dirs, files in file_path.walk():
                            for file in files:
                                zipf.write(root / file, arcname=root / file)
                if not compressed_path.exists():
                    raise FileNotFoundError(f'Could not create compressed file {compressed_path} from {file_path}')
            elif extension in ('bz2', 'gzip', 'lzma'):
                mod = importlib.import_module(extension)
                with open(file_path, 'rb') as in_file, open(compressed_path, 'wb') as compressed_file:
                    compressed_file.write(mod.compress(in_file.read()))
            else:
                raise NotImplementedError(f'Unrecognized compression extension {extension}')
        else:
            print(f'Compressed file {compressed_path} exists. Skipping compression.')
    return compressed_path


def checksum(file_path, algorithm='md5', verbose=False):
    """
    Calculates the checksum of a file.

    Arguments
    ---------
    file_path : str or pathlib.Path
        The file path to search for.
    algorithm : str
        The algorithm to use for the checksum.
    verbose : bool
        Print progress info.

    Returns
    -------
    checksum : str or None
        The checksum or None if failed.
    """
    file_path = Path(file_path)
    if file_path.exists():
        if verbose:
            print(f'Calculating checksum for {file_path}')
        import hashlib
        hash = hashlib.new(algorithm)  # TODO: check because could be faster to call the function directly
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash.update(chunk)
        file_checksum = hash.hexdigest()
    else:
        raise FileNotFoundError(f'Cannot find file {file_path}')
    return file_checksum


def find_existing_extension(path, extensions):  # FIXME: see if could use Glob
    """
    Return the first extension in the list of `extensions` that creates
    and existing file with `path`.

    Parameters
    ----------
    path
    extensions

    Returns
    -------
    str or None
    """
    for extension in extensions:
        if path.with_suffix(extension).exists():
            return extension


def check_extensions(extensions):
    """
    Check if the extensions are valid (i.e. correctly constructed and
    supported by ClearMap IO).

    Parameters
    ----------
    extensions: list[str]
        The list of extensions to check.

    Raises
    ------
    ValueError
    """
    for ext in extensions:
        if not ext.startswith('.'):
            raise ValueError(f'Extension "{ext}" should start with a dot.')
        if not is_clearmap_source_extension(ext):
            raise ValueError(f'Unknown extension "{ext}". '
                             f'Supported extensions are "{clearmap_io.file_extension_to_module.keys()}".')


def is_clearmap_source_extension(extension):
    """
    Check if the extension is supported by ClearMap IO.

    Parameters
    ----------
    extension: str
        The extension to check.

    Returns
    -------
    bool
    """
    return extension.lstrip('.') in clearmap_io.file_extension_to_module.keys()
    

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
