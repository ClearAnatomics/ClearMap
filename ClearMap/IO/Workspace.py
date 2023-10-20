# -*- coding: utf-8 -*-
"""
Workspace
=========

The Workspace module keeps track of the data files of a project.
Using this module will simplify access to data and results using coherent
filenames across experiments and samples.

One can think of a Workspace as a transparent data structure for ClearMap.

Note
----
Additional standard filenames can be added in the `ftype_to_filename dict`.
"""
__author__ = 'Christoph Kirst <christoph.kirst.ck@gmail.com>'
__license__ = 'GPLv3 - GNU General Public License v3 (see LICENSE.txt)'
__copyright__ = 'Copyright © 2023 by Christoph Kirst'
__webpage__ = 'https://idisco.info'
__download__ = 'https://www.github.com/ChristophKirst/ClearMap2'

# TODO: provide modularity -> e.g. modules for various files, adding/removing channels, test data slicing etc.

import os

from collections import OrderedDict

import numpy as np

import ClearMap.IO.IO as clearmap_io
from ClearMap.Utils.TagExpression import Expression
# import ClearMap.ParallelProcessing.DataProcessing.ArrayProcessing as array_processing
import ClearMap.Visualization.Plot3d as q_plot_3d

########################################################################################################################
# Workspaces
########################################################################################################################

workspace_types = [None, 'CellMap', 'TubeMap', 'AxonMap', 'Both']  # FIXME: these shouldn't be mutually exclusive

########################################################################################################################
# File and directory names
########################################################################################################################

default_directory_type = 'analysis'

default_directory_type_to_name = OrderedDict(
    data=None,
    analysis=None,
)

default_file_type_to_name = OrderedDict(
    raw                       = "Raw/raw_<X,2>_<Y,2>.npy",
    autofluorescence          = "Autofluorescence/auto_<X,2>_<Y,2>.npy",
    stitched                  = "stitched.npy", 
    layout                    = "layout.lyt",
    background                = "background.npy",
    resampled                 = "resampled.tif",
    autofluorescence_resampled="resampled_autofluorescence.tif",
    resampled_to_autofluorescence='elastix_resampled_to_auto',
    resampled_aligned_to_autofluorescence="elastix_resampled_to_auto/result.0.mhd",
    autofluorescence_to_reference='elastix_auto_to_reference',
    autofluorescence_aligned_to_reference="elastix_auto_to_reference/result.1.mhd",
    )


default_file_type_to_directory_type = OrderedDict(
    raw="data",
    autofluorescence="data",
    resampled="data",
    autofluorescence_resampled="data",
    resampled_to_autofluorescence="data",
    resampled_aligned_to_autofluorescence="data",
    autofluorescence_to_reference="data",
    autofluorescence_aligned_to_reference="data",
)

default_file_type_synonyms = dict(
    r  = "raw",
    a  = "autofluorescence",
    st = "stitched",
    l  = "layout",
    bg = "background",
    rs = "resampled",
    auto="autofluorescence",
    resampled_to_auto='resampled_to_autofluorescence',
    resampled_aligned_to_auto='resampled_aligned_to_autofluorescence',
    auto_to_reference='autofluorescence_to_reference',
    auto_aligned_to_reference='autofluorescence_aligned_to_reference',
    resampled_auto='autofluorescence_resampled',
    auto_resampled='autofluorescence_resampled'
)
  
# Workspace specialization
default_file_type_to_name_from_workspace = OrderedDict()
default_file_type_to_directory_type_from_workspace = OrderedDict()
default_file_type_synonyms_from_workspace = OrderedDict()

default_file_type_to_name_from_workspace[None] = default_file_type_to_name.copy()
default_file_type_to_directory_type_from_workspace[None] = default_file_type_to_directory_type.copy()
default_file_type_synonyms_from_workspace[None] = default_file_type_synonyms.copy()

default_file_type_to_name_from_workspace['CellMap'] = default_file_type_to_name.copy()
default_file_type_to_name_from_workspace['CellMap'].update(
    raw="/Raw/Z<Z,4>.tif",
    autofluorescence="/Autofluorescence/Z<Z,4>.tif",
    cells="cells.npy",
    density="density.tif",
)

default_file_type_to_directory_type_from_workspace['CellMap'] = default_file_type_to_directory_type.copy()
default_file_type_to_name_from_workspace['CellMap'].update(
)

default_file_type_synonyms_from_workspace['CellMap'] = default_file_type_synonyms.copy()
default_file_type_synonyms_from_workspace['CellMap'].update(
    c="cells"
)

default_file_type_to_name_from_workspace['TubeMap'] = default_file_type_to_name.copy()
default_file_type_to_name_from_workspace['TubeMap'].update(
    arteries                  = "Raw/arteries_<X,2>_Y,2>.npy",
    binary                    = "binary.npy",
    binary_status             = "binary_status.npy",
    skeleton                  = 'skeleton.npy',
    graph                     = "graph.gt",
    density                   = "density.tif"
)
default_file_type_to_name_from_workspace['Both'] = {**default_file_type_synonyms_from_workspace['CellMap'],
                                                    **default_file_type_to_name_from_workspace['TubeMap']}
   
default_file_type_to_directory_type_from_workspace['TubeMap'] = default_file_type_to_directory_type.copy()
default_file_type_to_directory_type_from_workspace['TubeMap'].update(
    arteries="data",
)

default_file_type_synonyms_from_workspace['TubeMap'] = default_file_type_synonyms.copy()
default_file_type_synonyms_from_workspace['TubeMap'].update(
    b  = "binary",
    bs = "binary_status",
    g  = "graph",
    sk = "skeleton"
)

default_file_type_to_name_from_workspace['AxonMap'] = default_file_type_to_name.copy()
default_file_type_to_name_from_workspace['AxonMap'].update(
    axons='axons.npy',
    orientation="orientation.npy",
    preprocessed="analyzed/preprocessed.npy",
    lightsheet="analyzed/lightsheet.npy",
    equalized="analyzed/equalized.npy",
    tubeness="analyzed/tubeness.npy",
    hessian="analyzed/hessian.npy",
    features="analyzed/features.npy",
    fod="analyzed/fod.npy"
)

default_file_type_to_directory_type_from_workspace['AxonMap'] = default_file_type_to_directory_type.copy()
default_file_type_to_directory_type_from_workspace['AxonMap'].update(
    )

default_file_type_synonyms_from_workspace['AxonMap'] = default_file_type_synonyms.copy()
default_file_type_synonyms_from_workspace['AxonMap'].update(
)

    
########################################################################################################################
# Determine file and directory names
########################################################################################################################

def file_type(f_type=None, ws_type=None, file_type_synonyms=None):
    file_type_synonyms_ = default_file_type_synonyms_from_workspace[ws_type].copy()
    if file_type_synonyms is not None:
        file_type_synonyms_.update(file_type_synonyms)
    f_type = file_type_synonyms_.get(f_type, f_type)
    return f_type


def directory_type(f_type=None, ws_type=None,
                   file_type_to_directory_type=None,
                   file_type_synonyms=None, return_f_type=False):
    f_type = file_type(f_type=f_type, ws_type=ws_type,
                      file_type_synonyms=file_type_synonyms)

    file_type_to_directory_type_ = default_file_type_to_directory_type_from_workspace[ws_type].copy()
    if file_type_to_directory_type is not None:
        file_type_to_directory_type_.update(file_type_to_directory_type)

    if f_type is not None:
        if f_type in file_type_to_directory_type:
            dirtype = file_type_to_directory_type[f_type]
        else:
            dirtype = default_directory_type
    else:
        dirtype = None

    if return_f_type:
        return dirtype, f_type
    return dirtype


def directory_name(f_type=None, dirtype=None, ws_type=None, directory=None, default_directory=None,
                   file_type_to_directory_type=None,
                   directory_type_to_name=None,
                   file_type_synonyms=None, return_f_type=False, return_dirtype=False):
    """Returns the directory name to use for a file in the work space.

    Arguments
    ---------
    f_type : str or None
      The type of the file for which the file name is requested. If None, use the
    dirtype : str or None
      The directory type, overwrites the directory type inferred from the file type if not None.
    ws_type : str or None
      The type of workspace to use. If None, use the default workspace.
    directory : str or None
      A specific directory to use.
    default_directory : str or None
      A default directory to fall back to if not None.
    file_type_to_directory_type : dict or None
      The file type to determine the directory type for. If None, the default is used.
    directory_type_to_name : dict or None
      The mapping between th directory type and the actual directory.
    file_type_synonyms : dict or None
      Additional file type synonyms or abbreviation to use for the file types.
    return_f_type : bool
      If True return also the file type
    return_dirtype : bool
      If True return also the directory type.

    Returns
    -------
    directory : str
      The standard directory name of the requested file type.
    """
    if dirtype is None:
        dirtype, f_type = directory_type(f_type=f_type, ws_type=ws_type,
                                        file_type_to_directory_type=file_type_to_directory_type,
                                        file_type_synonyms=file_type_synonyms, return_f_type=True)

    if directory is None:
        if dirtype is None:
            raise ValueError('cannot determine directory')
        else:
            directory_type_to_name_ = default_directory_type_to_name.copy()
            if directory_type_to_name is not None:
                directory_type_to_name_.update(directory_type_to_name)
            directory = directory_type_to_name_.get(dirtype, directory)

    if directory is None and default_directory is not None:
        directory = default_directory

    if directory is not None and len(directory) > 0 and directory[-1] == os.path.sep:
        directory = directory[:-1]

    result = (directory,)
    if return_f_type:
        result += (f_type,)
    if return_dirtype:
        result += (dirtype,)
    if len(result) == 1:
        result = result[0]
    return result
    

def file_name(f_type, dirtype=None, ws_type=None, directory=None, default_directory=None,
              expression=None, values=None, prefix=None, postfix=None, extension=None, debug=None,
              file_type_to_name=None, file_type_synonyms=None,
              file_type_to_directory_type=None, directory_type_to_name=None, **kwargs):
    """
    Returns the standard file name to use for a result file.

    Arguments
    ---------
    f_type : str or None
        The type of the file for which the file name is requested.
    dirtype : str or None
        The directory type to use for the requested file name.
    ws_type : str or None
        The type of workspace to use. If None, use the default workspace.
    directory : str or None
        A specific directory for the file.
    default_directory : str or None
        A default directory to fall back to if not None.
    expression : str or None
        A tag expression to use if f_type is 'expression'.
    values : dict or None
        The values to use in case a tag expression is given.
    prefix : str or None
        Optional prefix to the file if not None.
    postfix : str or list of str or None
        Optional postfix to the file if not None.
    extension : str or None
        Optional extension to replace existing one.
    debug : str, bool or None
        Optional string for testing in which the string is added as postfix.
        If True, 'debug' is added.
    file_type_to_name : dict or None
        The file types to name mappings. If None, the default is used.
    file_type_synonyms : dict or None
      Additional file type synonyms or abbreviation to use for the file types.
    file_type_to_directory_type : dict or None
      The file type to determine the directory type for. If None, the default is used.
    directory_type_to_name : dict or None
      The mapping between th directory type and the actual directory.

    Returns
    -------
    filename : str
        The file name of the requested file specifications.
    """
    directory, f_type = directory_name(f_type=f_type, dirtype=dirtype, ws_type=ws_type,
                                       directory=directory, default_directory=default_directory,
                                       file_type_to_directory_type=file_type_to_directory_type,
                                       directory_type_to_name=directory_type_to_name,
                                       return_f_type=True, **kwargs)

    file_type_to_name_ = default_file_type_to_name_from_workspace[ws_type].copy()
    if file_type_to_name is not None:
        file_type_to_name_.update(file_type_to_name)

    if f_type in file_type_synonyms.keys():
        f_type = file_type_synonyms[f_type]

    if f_type == 'expression' or expression is not None:
        f_name = Expression(expression).string(values=values)
        # Note: expressions are used for raw data only atm -> no prefix, debug
        # prefix = None
        # debug = None
    else:
        f_name = file_type_to_name_.get(f_type)

    f_name_expression = Expression(f_name)
    if f_name_expression.tags:
        f_name = f_name_expression.string(values=values)
        # Note: expressions are used for raw data only atm -> no prefix, debug
        # prefix = None
        # debug = None

    if f_name is None:
        raise ValueError(f'Cannot find name for type {f_type}!')

    if prefix:
        if isinstance(prefix, list):
            prefix = '_'.join(prefix)
        path, file_ = os.path.split(f_name)
        f_name = os.path.join(path, f'{prefix}_{file_}')

    if postfix:
        if isinstance(postfix, list):
            postfix = '_'.join(postfix)
        base, ext = os.path.splitext(f_name)
        f_name = f'{base}_{postfix}{ext}'

    if debug:
        if not isinstance(debug, str):
            debug = 'debug'
        path, file_ = os.path.split(f_name)
        f_name = os.path.join(path, f'{debug}_{file_}')

    if extension:
        extension = extension if extension.startswith('.') else f'.{extension}'
        f_name = f'{os.path.splitext(f_name)[0]}{extension}'

    if directory:
        f_name = clearmap_io.join(directory, f_name)

    return f_name


###############################################################################
# Workspace
###############################################################################

class Workspace(object):
    """Class to organize files."""
  
    def __init__(self, ws_type=None, directory=None, prefix=None, postfix=None, debug=None,
                 file_type_to_name=None, file_type_synonyms=None,
                 file_type_to_directory_type=None, directory_type_to_name=None,
                 **kwargs):

        self._wstype = ws_type  # RO
        self.prefix = prefix
        self.postfix = postfix
        self._debug = debug

        if directory:
            directory = os.path.normpath(directory)
        self.directory = directory

        self._file_type_to_name = default_file_type_to_name_from_workspace.get(ws_type, default_file_type_to_name).copy()
        if file_type_to_name is not None:
            self._file_type_to_name.update(file_type_to_name)
        self._file_type_to_name.update(**kwargs)
    
        self._file_type_synonyms = default_file_type_synonyms_from_workspace.get(ws_type,
                                                                                 default_file_type_synonyms).copy()
        if file_type_synonyms is not None:
            self._file_type_to_name.update(file_type_synonyms)

        self._file_type_to_directory_type = default_file_type_to_directory_type_from_workspace.get(
            ws_type, default_file_type_to_directory_type).copy()
        if file_type_to_directory_type is not None:
            self._file_type_to_directory_type.update(file_type_to_directory_type)

        self._directory_type_to_name = default_directory_type_to_name.copy()
        if directory_type_to_name is not None:
            self._directory_type_to_name.update(directory_type_to_name)

    @property
    def wstype(self):
        return self._wstype
  
    # @wstype.setter
    # def wstype(self, value):  # Remove setter  -> RO
    #     self.update(default_workspaces.get(value, default_file_type_to_name))
    #     self._wstype = value
    
    # @property
    # def prefix(self):
    #     return self._prefix
    #
    # @prefix.setter
    # def prefix(self, value):
    #     self._prefix = value

    # @property
    # def postfix(self):
    #     return self._postfix
    #
    # @postfix.setter
    # def postfix(self, value):
    #     self._postfix = value

    @property
    def directory(self):
        return self._directory
  
    @directory.setter
    def directory(self, path):
        # if value and len(value) > 0 and value[-1] == os.path.sep:
        #     value = value[:-1]
        self._directory = path.rstrip(os.sep)

    def load(self, file_path):
        """Loads the workspace configuration from disk"""
        d = np.load(file_path)[0]
        self.__dict__.update(d)

    def save(self, file_path):
        """Saves the workspace configuration to disk"""
        # prevent np to add .npy to a .workspace file
        with open(file_path, "wb") as fid:
            np.save(fid, [self.__dict__])
    
    # @property
    # def file_type_to_name(self):
    #     return self._file_type_to_name
    #
    # @file_type_to_name.setter
    # def file_type_to_name(self, value):
    #     self._file_type_to_name = value
    
    @property
    def file_type_to_directory_type(self):
        return self._file_type_to_directory_type

    @file_type_to_directory_type.setter
    def file_type_to_directory_type(self, value):
        self._file_type_to_directory_type = value

    @property
    def file_type_synonyms(self):
        return self._file_type_synonyms

    @file_type_synonyms.setter
    def file_type_synonyms(self, value):
        self._file_type_synonyms = value

    @property
    def directory_type_to_name(self):
        return self._directory_type_to_name

    @directory_type_to_name.setter
    def directory_type_to_name(self, value):
        self._directory_type_to_name = value

    def update(self, *args, **kwargs):
        # FIXME: add check that
        self._file_type_to_name.update(*args, **kwargs)
  
    @property
    def debug(self):
        return self._debug
  
    @debug.setter
    def debug(self, value):
        if value is True:  # So that value can be a custom string
            value = 'debug'
        self._debug = value if value else None
  
    def create_debug(self, f_type, slicing, debug=None, **kwargs):
        if debug is None:
            debug = self.debug if self.debug is not None else 'debug'
        self.debug = None
        # FIXME: why not force debug=None here instead of unset set
        source = clearmap_io.as_source(self.file_name(f_type, **kwargs))
        self.debug = debug
        return clearmap_io.write(self.file_name(f_type, **kwargs), np.asarray(source[slicing], order='F'))
  
    def file_name(self, f_type=None, dirtype=None, ws_type=None, directory=None,
                  expression=None, values=None, prefix=None, postfix=None, extension=None, debug=None,
                  file_type_to_name=None, file_type_synonyms=None,
                  file_type_to_directory_type=None, directory_type_to_name=None,
                  **kwargs):

        return file_name(f_type=f_type, dirtype=dirtype, ws_type=ws_type or self.wstype,
                         directory=directory, default_directory=self.directory,
                         expression=expression, values=values,
                         prefix=prefix or self.prefix, postfix=postfix or self.prefix,
                         extension=extension, debug=debug or self.debug,
                         file_type_to_name=file_type_to_name or self._file_type_to_name,
                         file_type_synonyms=file_type_synonyms or self.file_type_synonyms,
                         file_type_to_directory_type=file_type_to_directory_type or self.file_type_to_directory_type,
                         directory_type_to_name=directory_type_to_name or self.directory_type_to_name,
                         **kwargs)

    def filename(self, *args, **kwargs):
        return self.file_name(*args, **kwargs)

    def exists(self, f_type, file_type_to_name=None, directory=None, expression=None, values=None, prefix=None,
               extension=None, debug=None, **kwargs):
        return os.path.exists(self.file_name(f_type, file_type_to_name=file_type_to_name, directory=directory,
                                            expression=expression, values=values, prefix=prefix, extension=extension,
                                            debug=debug, **kwargs))

    def all_tiles_exist(self, f_type, file_type_to_name=None, directory=None, expression=None, values=None,
                        prefix=None, extension=None, debug=None, **kwargs):
        files = self.file_list(f_type, file_type_to_name=file_type_to_name, directory=directory, expression=expression,
                               values=values, prefix=prefix, extension=extension, debug=debug, **kwargs)
        return len(files) == self.mosaic_shape(f_type).prod()

    def directory_type(self, f_type=None, ws_type=None,
                       file_type_to_directory_type=None, file_type_synonyms=None, return_f_type=False):
        return directory_type(f_type=f_type, ws_type=ws_type or self.wstype,
                              file_type_to_directory_type=file_type_to_directory_type or self.file_type_to_directory_type,
                              file_type_synonyms=file_type_synonyms or self.file_type_synonyms,
                              return_f_type=return_f_type)

    def directory_name(self, f_type=None, dirtype=None, ws_type=None, directory=None, default_directory=None,
                       file_type_to_directory_type=None,
                       directory_type_to_name=None,
                       file_type_synonyms=None, return_f_type=False, return_dirtype=False):
        return directory_name(f_type=f_type, dirtype=dirtype, ws_type=ws_type or self.wstype,
                              directory=directory, default_directory=default_directory or self.directory,
                              file_type_to_directory_type=file_type_to_directory_type or self.file_type_to_directory_type,
                              directory_type_to_name=directory_type_to_name or self.directory_type_to_name,
                              file_type_synonyms=file_type_synonyms or self.file_type_synonyms,
                              return_f_type=return_f_type, return_dirtype=return_dirtype)

    def expression(self, *args, **kwargs):
        return Expression(self.file_name(*args, **kwargs))

    def mosaic_shape(self, f_type):
        exp = Expression(self.file_name(f_type))
        positions = self.get_positions(f_type)
        tile_axes_ = exp.tag_names()
        indices = [tuple(tv[n] for n in tile_axes_) for tv in positions]
        # noinspection PyArgumentList
        mosaic_shape = np.array(indices).max(axis=0) + 1  # Because 0 indexing
        return mosaic_shape

    def get_positions(self, f_type):
        exp = Expression(self.file_name(f_type))
        files = self.file_list(f_type)
        positions = [exp.values(f) for f in files]
        return positions

    def extension(self, *args, **kwargs):
        return clearmap_io.file_extension(self.file_name(*args, **kwargs))
  
    def file_list(self, *args, **kwargs):
        return clearmap_io.file_list(self.file_name(*args, **kwargs))
  
    def create(self, f_type=None, dtype=None, shape=None, order=None, **kwargs):
        filename = self.file_name(f_type=f_type, **kwargs)
        clearmap_io.create(filename, shape=shape, dtype=dtype, order=order)
        return filename
  
    def source(self, *args, **kwargs):
        f_name = self.file_name(*args, **kwargs)
        return clearmap_io.as_source(f_name)
  
    def read(self, *args, **kwargs):
        return clearmap_io.read(self.file_name(*args, **kwargs))
        # return array_processing.read(self.file_name(*args, **kwargs))

    def plot(self, f_type, **kwargs):
        return q_plot_3d.plot(self.file_name(f_type, **kwargs))
 
    def write(self, *args, **kwargs):
        if 'data' in kwargs.keys():
            data = kwargs.pop('data')
        else:
            data = args[-1]
            args = args[:-1]
        return clearmap_io.write(self.filename(*args, **kwargs), data)
        # return array_processing.write(self.filename(*args, **kwargs), data)

    def __format_pattern_line(self, files, expression, tag_names, tile_axes_):
        tile_positions = [expression.values(f) for f in files]
        tile_positions = [tuple(tv[n] for n in tile_axes_) for tv in tile_positions]
        tile_lower = tuple(np.min(tile_positions, axis=0))
        tile_upper = tuple(np.max(tile_positions, axis=0))
        tag_names = tuple(tag_names)
        relative_file_pattern = os.path.relpath(expression.string(), start=self.directory)
        pattern_line = f'{relative_file_pattern} {{{len(files)} files, {tag_names}: {tile_lower} -> {tile_upper}}}\n'
        return pattern_line
  
    def __str__(self):
        s = "Workspace"  # self.__class__.__name__
        if self.wstype is not None:
            s += f'[{self.wstype}]'
        if self.prefix is not None:
            s += f'({self.prefix})'
        if self.directory is not None:
            s += f'{{{self.directory}}}'
        if self.debug is not None:
            s += f'[{self.debug}]'
        return s
  
    def __repr__(self):
        return self.__str__()
  
    def info(self, tile_axes=None, check_extensions=True, check_directory=True):  # REFACTOR:
        out = f'{self}\n'

        out += 'directories:\n'
        len_dirtype = np.max([len(dirtype) for dirtype in self.directory_type_to_name])

        for dirtype in self.directory_type_to_name:
            dirname = self.directory_name(dirtype=dirtype)
            out += f'  [{dirtype : >{len_dirtype}}]: {dirname}\n'

        out += '\nfiles:\n'

        len_f_type = np.max([len(f_type) for f_type in self._file_type_to_name])
        header = f'  [{{:{len_dirtype}}}] {{:{len_f_type}}}'

        for f_type, f_names in self._file_type_to_name.items():
            dirname, dirtype = self.directory_name(f_type=f_type, return_dirtype=True)
            len_dirname = len(dirname) + 1 if dirname is not None else 0  # FIXME: unused
            if Expression(f_names).tags:
                if check_extensions:
                    extensions = list(np.unique([os.path.splitext(f)[-1] for f in
                                                 self.file_list(f_type, extension='*')]))
                else:
                    extensions = [self.extension(f_type)]

                if not extensions:
                    out += f'{header.format(dirtype, f_type)}: no file\n'
                else:
                    for i, extension in enumerate(extensions):
                        expression = Expression(self.file_name(f_type, extension=extension))
                        tag_names = expression.tag_names()
                        tile_axes_ = tile_axes if tile_axes is not None else tag_names
                        for n in tile_axes_:
                            if n not in tag_names:
                                raise ValueError(f'The expression does not have the named pattern {n}')
                        for n in tag_names:
                            if n not in tile_axes_:
                                raise ValueError(f'The expression has the named pattern {n} '
                                                 f'that is not in tile_axes={tile_axes_}')

                        # construct tiling
                        files = clearmap_io.file_list(expression)
                        if files:
                            if i == 0:
                                out += f'{header.format(dirtype, f_type)}: '
                            else:
                                out += f'{header.format("", "")}  '
                            pattern_line = self.__format_pattern_line(files, expression, tag_names, tile_axes_)
                            out += pattern_line
            else:
                f_name = self.file_name(f_type)

                if check_directory and clearmap_io.is_directory(f_name):
                    out += f'{header.format(dirtype, f_type)}: directory\n'
                else:
                    files = []
                    if clearmap_io.is_file(f_name):
                        files += [f_name]
                    f_name = self.file_name(f_type, postfix='*')
                    files += clearmap_io.file_list(f_name)
                    if files:
                        files = [os.path.relpath(f, start=self.directory) for f in files]

                        out += f'{header.format(dirtype, f_type)}: {files[0]}\n'
                        for f in files[1:]:
                            out += f'{header.format("", "")}  {f}\n'
                    else:
                        out += f'{header.format(dirtype, f_type)}: no file\n'

        print(out)  # TODO: add print option or return s
