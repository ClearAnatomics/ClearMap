# -*- coding: utf-8 -*-
"""
IO (deprecated)
===============

Backwards-compatibility shim for the pre-3.1 ``ClearMap.IO.IO`` module.

Every public name of the old module still resolves, but each access emits a
``DeprecationWarning`` naming its replacement:

* data access: :mod:`ClearMap.IO.io_ops` (read, write, open_ro, edit, create, ...)
* dispatch: :mod:`ClearMap.IO.dispatch` (as_source, source_to_module, ...)
* source properties: :mod:`ClearMap.IO.source_geometry` (shape, dtype, order, ...)
* initialisation: :mod:`ClearMap.IO.source_initialization`
* dtype limits: :mod:`ClearMap.IO.dtypes`
* file paths: :mod:`ClearMap.IO.FileUtils`
* format modules: :mod:`ClearMap.IO.source.backends` (the old ``io.mmp``, ``io.tif``, ...)

To find remaining uses in ClearMap itself, run the tests with
``-W error::DeprecationWarning:ClearMap``.
"""
__author__ = 'Christoph Kirst <christoph.kirst.ck@gmail.com>'
__license__ = 'GPLv3 - GNU General Public License v3 (see LICENSE.txt)'
__copyright__ = 'Copyright © 2020 by Christoph Kirst'
__webpage__ = 'https://idisco.info'
__download__ = 'https://github.com/ClearAnatomics/ClearMap'

import importlib
import warnings

_BACKENDS = 'ClearMap.IO.source.backends'

# name -> (module, attribute); attribute None means the name was a module alias
_MOVED = {
    # data access
    'read': ('ClearMap.IO.io_ops', 'read'),
    'write': ('ClearMap.IO.io_ops', 'write'),
    'open_ro': ('ClearMap.IO.io_ops', 'open_ro'),
    'edit': ('ClearMap.IO.io_ops', 'edit'),
    'create': ('ClearMap.IO.io_ops', 'create'),
    'is_source': ('ClearMap.IO.io_ops', 'is_source'),
    'buffer': ('ClearMap.IO.io_ops', 'buffer'),
    # dispatch
    'as_source': ('ClearMap.IO.dispatch', 'as_source'),
    'source': ('ClearMap.IO.dispatch', 'as_source'),
    'source_to_module': ('ClearMap.IO.dispatch', 'source_to_module'),
    'location_to_module': ('ClearMap.IO.dispatch', 'location_to_module'),
    'filename_to_module': ('ClearMap.IO.dispatch', 'filename_to_module'),
    # source properties
    'ndim': ('ClearMap.IO.source_geometry', 'ndim'),
    'shape': ('ClearMap.IO.source_geometry', 'shape'),
    'size': ('ClearMap.IO.source_geometry', 'size'),
    'dtype': ('ClearMap.IO.source_geometry', 'dtype'),
    'order': ('ClearMap.IO.source_geometry', 'order'),
    'element_strides': ('ClearMap.IO.source_geometry', 'element_strides'),
    'location': ('ClearMap.IO.source_geometry', 'location'),  # FIXME: Why is location in there ??
    'memory': (f'{_BACKENDS}.sma_backend', 'memory'),
    # initialisation
    'initialize': ('ClearMap.IO.source_initialization', 'initialize'),
    'initialize_buffer': ('ClearMap.IO.source_initialization', 'initialize_buffer'),
    # dtype limits
    'get_info': ('ClearMap.IO.dtypes', 'get_info'),
    'get_value': ('ClearMap.IO.dtypes', 'get_value'),
    'min_value': ('ClearMap.IO.dtypes', 'min_value'),
    'max_value': ('ClearMap.IO.dtypes', 'max_value'),
    # conversion and file lists
    'convert': ('ClearMap.IO.conversion', 'convert'),
    'convert_files': ('ClearMap.IO.conversion', 'convert_files'),
    'file_list': (f'{_BACKENDS}.file_list_backend', 'file_list'),
    # file paths (re-exported from FileUtils by the old module)
    **{name: ('ClearMap.IO.FileUtils', name) for name in (
        'is_file', 'is_directory', 'file_extension', 'join', 'split', 'abspath',
        'create_directory', 'delete_directory', 'copy_file', 'link_file', 'delete_file')},
    # exceptions the old module imported
    'IncompatibleSource': ('ClearMap.Utils.exceptions', 'IncompatibleSource'),
    'SourceModuleNotFoundError': ('ClearMap.Utils.exceptions', 'SourceModuleNotFoundError'),
    'CancelableProcessPoolExecutor': ('ClearMap.Utils.utilities', 'CancelableProcessPoolExecutor'),
    # module aliases the old module leaked (e.g. io.mmp.create, io.slc.unpack_slicing, io.mp.cpu_count)
    'src': ('ClearMap.IO.source.Source', None),
    'slc': ('ClearMap.IO.source.Slice', None),
    'npy': (f'{_BACKENDS}.npy_backend', None),
    'mmp': (f'{_BACKENDS}.mmp_backend', None),
    'sma': (f'{_BACKENDS}.sma_backend', None),
    'tif': (f'{_BACKENDS}.tif_backend', None),
    'nrrd': (f'{_BACKENDS}.nrrd_backend', None),
    'mhd': (f'{_BACKENDS}.mhd_backend', None),
    'csv': (f'{_BACKENDS}.csv_backend', None),
    'gt': (f'{_BACKENDS}.gt_backend', None),
    'fl': (f'{_BACKENDS}.file_list_backend', None),
    'fu': ('ClearMap.IO.FileUtils', None),
    'te': ('ClearMap.Utils.tag_expression', None),
    'tmr': ('ClearMap.Utils.Timer', None),
    'ptb': ('ClearMap.ParallelProcessing.ParallelTraceback', None),
    'mp': ('multiprocessing', None),
    'np': ('numpy', None),
    'pd': ('pandas', None),
}


def _available_backends():
    from ClearMap.IO.source.protocol import Backend
    return [backend for backend in Backend if backend.is_available]


# Old module-level tables, rebuilt from the backend registry on access
_COMPUTED = {
    'source_modules': (lambda: [b.module for b in _available_backends()],
                       'iterate ClearMap.IO.source.protocol.Backend (backend.module)'),
    'file_extension_to_module': (lambda: {ext: b.module for b in _available_backends() for ext in b.extensions},
                                 'ClearMap.IO.source.backends.registry.BY_EXTENSION'),
    'gt_loaded': (lambda: any(b.key == 'gt' for b in _available_backends()),
                  'ClearMap.IO.source.protocol.Backend.GT.is_available'),
}


def __getattr__(name):
    if name in _MOVED:
        module_name, attribute = _MOVED[name]
        replacement = f'import {module_name}' if attribute is None else f'{module_name}.{attribute}'
        warnings.warn(f'ClearMap.IO.IO.{name} is deprecated; use {replacement}.', DeprecationWarning, stacklevel=2)
        try:
            module = importlib.import_module(module_name)
        except ImportError as err:  # e.g. io.gt without graph-tool, as in the old module
            raise AttributeError(f'ClearMap.IO.IO.{name} is unavailable: {err}') from err
        return module if attribute is None else getattr(module, attribute)
    if name in _COMPUTED:
        compute, replacement = _COMPUTED[name]
        warnings.warn(f'ClearMap.IO.IO.{name} is deprecated; use {replacement}.', DeprecationWarning, stacklevel=2)
        return compute()
    if name == 'AssetBase':
        raise AttributeError('ClearMap.IO.IO.AssetBase was removed; use ClearMap.IO.workspace_asset.Asset.')
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')

def __dir__():
    return sorted(list(globals()) + list(_MOVED) + list(_COMPUTED))
