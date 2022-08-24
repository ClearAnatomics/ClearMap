# -*- coding: utf-8 -*-
"""
Workspace
=========

The Workspace module keeps track of the data files of a project.
Using this module will simplify access to data and results using coherent
filenames accross experiments and samples.

One can think of a Workspace as a transparent data structure for ClearMap.

Note
----
Additional standard filenames can be added in the `ftype_to_filename dict`.
"""
__author__ = 'Christoph Kirst <christoph.kirst.ck@gmail.com>'
__license__ = 'GPLv3 - GNU General Pulic License v3 (see LICENSE.txt)'
__copyright__ = 'Copyright © 2020 by Christoph Kirst'
__webpage__ = 'https://idisco.info'
__download__ = 'https://www.github.com/ChristophKirst/ClearMap2'

# TODO: integrate with DataJoint or NWB data formats

import os

from collections import OrderedDict

import numpy as np

import ClearMap.IO.IO as clearmap_io
from ClearMap.Utils.TagExpression import Expression
import ClearMap.ParallelProcessing.DataProcessing.ArrayProcessing as array_processing
import ClearMap.Visualization.Plot3d as q_plot3d


###############################################################################
# ## Filenames
###############################################################################

default_file_type_to_name = OrderedDict(
    raw                       = "/Raw/raw_<X,2>_<Y,2>.npy",
    autofluorescence          = "/Autofluorescence/auto_<X,2>_<Y,2>.npy",
    stitched                  = "stitched.npy", 
    layout                    = "layout.lyt",
    background                = "background.npy",
    resampled                 = "resampled.tif",
    resampled_to_auto         = 'elastix_resampled_to_auto',
    auto_to_reference         = 'elastix_auto_to_reference', 
    )

file_type_synonyms = dict(
    r  = "raw",
    a  = "autofluorescence",
    st = "stitched",
    l  = "layout",
    bg = "background",
    rs = "resampled"
    )
  
default_file_type_to_name_tube_map = default_file_type_to_name.copy()
default_file_type_to_name_tube_map.update(
    arteries                  = "/Raw/arteries_<X,2>_Y,2>.npy", 
    binary                    = "binary.npy",
    binary_status             = "binary_status.npy",
    skeleton                  = 'skeleton.npy',
    graph                     = "graph.gt",
    density                   = "density.tif"
    )
   
file_type_synonyms.update(
    b  = "binary",
    bs = "binary_status",
    g  = "graph",
    sk = "skeleton"
    )


default_file_type_to_name_cell_map = default_file_type_to_name.copy()
default_file_type_to_name_cell_map.update(
    raw                       = "/Raw/Z<Z,4>.tif",
    autofluorescence          = "/Autofluorescence/Z<Z,4>.tif",
    cells                     = 'cells.npy',
    density                   = "density.tif",
    )

file_type_synonyms.update(c="cells")

default_file_type_to_name_both = {**default_file_type_to_name_cell_map, **default_file_type_to_name_tube_map}
    

default_workspaces = OrderedDict(
    CellMap = default_file_type_to_name_cell_map,
    TubeMap = default_file_type_to_name_tube_map,
    Both = default_file_type_to_name_both
    )
    

def filename(ftype, file_type_to_name=None, directory=None, expression=None, values=None, prefix=None, postfix=None,
             extension=None, debug=None):
    """
    Returns the standard file name to use for a result file.

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
        Optional string for debug files in which the string is added as postfix.
    If True, 'debug' is added.

    Returns
    -------
    filename : str
        The standard file name of the requested file type.
    """
    if file_type_to_name is None:
        file_type_to_name = default_file_type_to_name

    if ftype in file_type_synonyms.keys():
        ftype = file_type_synonyms[ftype]

    if ftype == 'expression' or expression is not None:
        f_name = Expression(expression).string(values=values)
        # Note: expressions are used for raw data only atm -> no prefix, debug
        prefix = None
        debug = None
    else:
        f_name = file_type_to_name.get(ftype)
    f_name_expression = Expression(f_name)
    if f_name_expression.tags:
        f_name = f_name_expression.string(values=values)
        # Note: expressions are used for raw data only atm -> no prefix, debug
        prefix = None
        debug = None

    if f_name is None:
        raise ValueError(f'Cannot find name for type {ftype}!')

    if prefix:
        if isinstance(prefix, list):
            prefix = '_'.join(prefix)
        f_name = f'{prefix}_{f_name}'

    if postfix:
        if isinstance(postfix, list):
            postfix = '_'.join(postfix)
        base, ext = os.path.splitext(f_name)
        f_name = f'{base}_{postfix}{ext}'

    if debug:
        if not isinstance(debug, str):
            debug = 'debug'
        f_name = f'{debug}_{f_name}'

    if extension:
        extension = extension if extension.startswith('.') else f'.{extension}'
        f_name = f'{os.path.splitext(f_name)[0]}{extension}'

    if directory:
        f_name = clearmap_io.join(directory, f_name)

    return f_name


###############################################################################
# ## Workspace
###############################################################################

class Workspace(object):
    """Class to organize files."""
  
    def __init__(self, wtype=None, prefix=None, file_type_to_name=None, directory=None, debug=None, **kwargs):
        self._wtype = wtype
        self._prefix = prefix
        self.directory = directory
        self._file_type_to_name = default_workspaces.get(wtype, default_file_type_to_name).copy()
        if file_type_to_name is not None:
            self._file_type_to_name.update(file_type_to_name)
        self._file_type_to_name.update(**kwargs)
        self._debug = debug
    
    @property
    def wtype(self):
        return self._wtype
  
    @wtype.setter
    def wtype(self, value):
        self.update(default_workspaces.get(value, default_file_type_to_name))
        self._wtype = value
    
    @property
    def prefix(self):
        return self._prefix
  
    @prefix.setter
    def prefix(self, value):
        self._prefix = value

    @property
    def directory(self):
        return self._directory
  
    @directory.setter
    def directory(self, value):
        if value and len(value) > 0 and value[-1] == os.path.sep:
            value = value[:-1]
        self._directory = value
    
    @property
    def file_type_to_name(self):
        return self._file_type_to_name
  
    @file_type_to_name.setter
    def file_type_to_name(self, value):
        self._file_type_to_name = value
    
    def update(self, *args, **kwargs):
        self._file_type_to_name.update(*args, **kwargs)
  
    @property
    def debug(self):
        return self._debug
  
    @debug.setter
    def debug(self, value):
        if value is True:  # So that value can be a custom string
            value = 'debug'
        if value is False:
            value = None
        self._debug = value
  
    def create_debug(self, ftype, slicing, debug=None, **kwargs):
        if debug is None:
            debug = self.debug
        if debug is None:
            debug = 'debug'
        self.debug = None
        source = clearmap_io.as_source(self.filename(ftype, **kwargs))
        self.debug = debug
        return clearmap_io.write(self.filename(ftype, **kwargs), np.asarray(source[slicing], order='F'))
  
    def plot(self, ftype, **kwargs):
        return q_plot3d.plot(self.filename(ftype, **kwargs))
  
    def load(self, filename):
        """Loads the configuration from disk"""
        d = np.load(filename)[0]
        self.__dict__.update(d)

    def save(self, filename):
        """Saves the configuration to disk"""
        # prevent np to add .npy to a .workspace file
        with open(filename, "wb") as fid:
            np.save(fid, [self.__dict__])
  
    def filename(self, ftype, file_type_to_name=None, directory=None, expression=None, values=None, prefix=None,
                 extension=None, debug=None, **kwargs):
        if directory is None:
            directory = self.directory
        if prefix is None:
            prefix = self.prefix
        if file_type_to_name is None:
            file_type_to_name = self.file_type_to_name
        if debug is None:
            debug = self.debug
        return filename(ftype, file_type_to_name=file_type_to_name,
                        directory=directory, expression=expression,
                        values=values, prefix=prefix, extension=extension,
                        debug=debug, **kwargs)
  
    def expression(self, *args, **kwargs):
        return Expression(self.filename(*args, **kwargs))

    def extension(self, ftype, file_type_to_name=None, directory=None, expression=None, values=None, prefix=None,
                  extension=None, debug=None, **kwargs):
        filename = self.filename(ftype=ftype,  file_type_to_name=file_type_to_name,
                                 directory=directory, expression=expression,
                                 values=values, prefix=prefix, extension=extension,
                                 debug=debug, **kwargs)
        return clearmap_io.extension(filename)
  
    def file_list(self, ftype, file_type_to_name=None, directory=None, expression=None, values=None, prefix=None,
                  extension=None, debug=None, **kwargs):
        filename = self.filename(ftype=ftype,  file_type_to_name=file_type_to_name,
                                 directory=directory, expression=expression,
                                 values=values, prefix=prefix, extension=extension,
                                 debug=debug, **kwargs)
        return clearmap_io.file_list(filename)
  
    def create(self, ftype, dtype=None, shape=None, order=None, file_type_to_name=None, directory=None,
               expression=None, values=None, prefix=None, extension=None, debug=None, **kwargs):
        filename = self.filename(ftype=ftype,  file_type_to_name=file_type_to_name,
                                 directory=directory, expression=expression,
                                 values=values, prefix=prefix,
                                 debug=debug, **kwargs)
        clearmap_io.create(filename, shape=shape, dtype=dtype, order=order)
        return filename
  
    def source(self, *args, **kwargs):
        f_name = self.filename(*args, **kwargs)
        return clearmap_io.as_source(f_name)
  
    def read(self, *args, **kwargs):
        return array_processing.read(self.filename(*args, **kwargs))
 
    # def write(self, *args, **kwargs):
    #   return ap.write(self.filename(*args, **kwargs));
  
    def __str__(self):
        s = "Workspace"  # self.__class__.__name__
        if self.wtype is not None:
            s += f'[{self.wtype}]'
        if self.prefix is not None:
            s += f'({self.prefix})'
        if self.directory is not None:
            s += f'{{{self.directory}}}'
        if self.debug is not None:
            s += f'[{self.debug}]'
        return s
  
    def __repr__(self):
        return self.__str__()
  
    def info(self, tile_axes=None, check_extensions=True):  # REFACTOR:
        out = f'{self}\n'

        padding = np.max([len(k) for k in self.file_type_to_name])

        for f_type, f_names in self.file_type_to_name.items():
            if Expression(f_names).tags:
                if check_extensions:
                    files = self.file_list(f_type, extension='*')
                    extensions = np.unique([os.path.splitext(f)[-1] for f in files])
                    # extensions = [clearmap_io.file_extension(f) for f in files]
                    # extensions = np.unique(extensions)
                else:
                    extensions = [self.extension(f_type)]

                if len(extensions) == 0:
                    out += f'{f_type : padding}: no file\n'
                else:
                    tmp_f_type = f_type  # used for first extension only
                    for extension in extensions:
                        expression = Expression(self.filename(f_type, extension=extension))
                        tag_names = expression.tag_names()
                        if tile_axes is None:
                            tile_axes_ = tag_names
                        else:
                            tile_axes_ = tile_axes
                        for n in tile_axes_:
                            if n not in tag_names:
                                raise ValueError(f'The expression does not have the named pattern {n}')
                        for n in tag_names:
                            if n not in tile_axes_:
                                raise ValueError(f'The expression has the named pattern {n} that is not in tile_axes={tile_axes_}')

                        # construct tiling
                        files = clearmap_io.file_list(expression)
                        if files:
                            tile_positions = [expression.values(f) for f in files]
                            tile_positions = [tuple(tv[n] for n in tile_axes_) for tv in tile_positions]
                            tile_lower = tuple(np.min(tile_positions, axis=0))
                            tile_upper = tuple(np.max(tile_positions, axis=0))
                            tag_names = tuple(tag_names)

                            if tmp_f_type is not None:
                                out += f'{tmp_f_type : >{padding}}: '
                                tmp_f_type = None
                            else:
                                out += f'{"" : >{padding}}  '
                            out += ('%s {%d files, %r: %r -> %r}' % (expression.string()[len(self.directory)+1:], len(files), tag_names, tile_lower, tile_upper)) + '\n'
            else:
                f_name = self.filename(f_type)
                files = []
                if clearmap_io.is_file(f_name):
                    files += [f_name]
                f_name = self.filename(f_type, postfix='*')
                files += clearmap_io.file_list(f_name)
                if len(files) > 0:
                    files = [f[len(self.directory)+1:] for f in files]

                    out += f'{f_type : >{padding}}: {files[0]}\n'  # REFACTOR: use header
                    for f in files[1:]:
                        out += f'{"" : >{padding}}  {f}\n'
                else:
                    out += f'{f_type : >{padding}}: no file\n'

        print(out)  # TODO: add print option or return s
