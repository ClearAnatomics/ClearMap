"""
Asset
=====

This module implements the Asset class that is used to represent the files and folders of a workspace.

The Asset class can represent Sources in which case it wraps the IO functions to read and write the data.
It is however not limited to Sources and can represent any file or folder of the workspace.
It also includes the following features:

* awareness of the content type (e.g. nuclei, veins, arteries, cells, axons, ...) and hence the algorithms applicable.
* awareness of the resource type (e.g. data, results, graphs, ...) and hence the sub folder to store the asset in.
* Transparent compression and decompression
* alternate extensions.
* checksums
* alternate file names (shortcuts)

In the future, it will also include:

* status (debug, ...)
* list of parent assets with the parametrised function to generate them

.. todo::
    * Add support for status (debug, ...) link workspace status to asset status
    * Add asset origin: chained list of parent assets (with the parametrised function to generate them)

.. done::
    * Add support for multiple extensions (glob ?)
    * Check for graphs and other data types
    * Handle identities elegantly
"""
__author__ = 'Charly Rousseau <charly.rousseau@icm-institute.org>'
__license__ = 'GPLv3 - GNU General Public License v3 (see LICENSE.txt)'
__webpage__ = 'https://idisco.info'
__download__ = 'https://github.com/ClearAnatomics/ClearMap'

import os
import warnings
from copy import deepcopy
from functools import cached_property
from pathlib import Path

import natsort
import numpy as np

from ClearMap.Analysis.Graphs.GraphGt import Graph
from ClearMap.IO import IO as clearmap_io
from ClearMap.IO import FileUtils as file_utils
from ClearMap.IO.assets_constants import CONTENT_TYPE_TO_PIPELINE
from ClearMap.IO.assets_specs import TypeSpec, ChannelSpec, StateManager
from ClearMap.Utils.tag_expression import Expression
from ClearMap.Visualization.Qt import Plot3d as q_plot_3d
from ClearMap.Utils.exceptions import ClearMapAssetError, AssetNotFoundError


class Asset:
    """
    An asset is a file or a folder that is part of a workspace.

    Attributes
    ----------
    base_directory : str | Path
        The base directory of the asset. This is the root directory of the workspace.
    type_spec : TypeSpec
        The type specification of the asset. This contains all the information for a step in the pipeline
        regardless of the channel.
    channel_spec : ChannelSpec
        The channel specification of the asset. This contains all the information for a channel.
    expression : str or Expression
        An optional expression that can be used to generate the file name.
        If provided, it will override the default file name.
        If the expression contains x and/or y, the asset is considered tiled.
    sample_id : str
        An optional sample id that can be used to identify the sample.
        If provided, it will be added as a prefix to the file name.
    subdirectory : str
        An optional subdirectory to store the asset in.
        This would override the value computed from the resource type.
    version : int or str
        The version of the asset. If provided, it will be added as a suffix to the file name.
        If supplied as string, it should be convertible to an integer.
    _checksum : str
        The checksum of the asset. If not provided, it will be computed from the file.
    extensions_to_directories : dict
        A dictionary mapping extensions to directories. This is used to store assets with multiple extensions
        in different directories. The key is the extension and the value is the directory.
    """

    def __new__(cls, *args, **kwargs):  # Used to optionally return an ExpressionAsset
        instance = super(Asset, cls).__new__(cls)
        instance.__init__(*args, **kwargs)
        if instance.is_tiled:
            instance = super(Asset, cls).__new__(TiledAsset)
            instance.__init__(*args, **kwargs)
        elif instance.is_expression:  # TODO: maybe separate ExpressionAsset and TiledAsset
            instance = super(Asset, cls).__new__(ExpressionAsset)
            instance.__init__(*args, **kwargs)
        return instance

    def __init__(self, base_directory,
                 type_spec: TypeSpec,
                 channel_spec: ChannelSpec,
                 expression: str | Expression | None = None,
                 sample_id: str = '',
                 subdirectory: Path | str = '',
                 version: int | str | None = None,
                 checksum: str = '',
                 status_manager: StateManager | None = None,
                 extensions_to_directories: dict | None = None):
        """
        An asset is a file or a folder that is part of a workspace.

        Arguments
        ---------
        base_directory : str | Path
            The base directory of the asset. This is the root directory of the workspace.
            If the type_spec, the expression, the subdirectory or extensions_to_directories define absolute paths,
            this is ignored.
        type_spec : TypeSpec
            The type specification of the asset. This contains all the information for a step in the pipeline
            regardless of the channel.
        channel_spec : ChannelSpec
            The channel specification of the asset. This contains all the information for a channel.
        expression : str or Expression, optional
            An optional expression that can be used to generate the file name.  # TODO: see if can use type_spec name
        sample_id : str
            An optional sample id that can be used to identify the sample.
            If provided, it will be added as a prefix to the file name.
        subdirectory : str
            An optional subdirectory to store the asset in. This would override the value computed from
            the resource type.
            .. warning:: The preferred way to store assets is to use the resource type.
        version : int or str, optional
            The version of the asset. If provided, it will be added as a suffix to the file name.
            If supplied as string, it should be convertible to an integer.
        checksum : str
            The checksum of the asset. If not provided, it will be computed from the file.
        extensions_to_directories : dict, optional
            A dictionary mapping extensions to directories. This is used to store assets with multiple extensions
            in different directories. The key is the extension and the value is the directory.
            .. warning:: This is mostly intended for the raw step of tiled assets.
        """
        if not isinstance(base_directory, (str, Path)):
            raise ValueError(f'base_directory must be a string or a path, got "{type(base_directory)}".')
        self.base_directory = Path(base_directory)  # REFACTOR: from reference to workspace or assetcollection

        self.sample_id = sample_id  # REFACTOR: from reference to workspace or assetcollection
        self.type_spec = type_spec
        self.channel_spec = channel_spec
        self.subdirectory = Path(subdirectory) if subdirectory else None
        self.expression = expression

        self.version = int(version) if isinstance(version, str) else version  # (e.g. 1, 2, ...)
        self._checksum = checksum
        self.extensions_to_directories = extensions_to_directories or {}
        self.status_manager = status_manager
        self.use_sub_dir = False  # Modified on first probe
        # TODO: add list of parent assets with the parametrised function to generate them

    def __str__(self):
        return str(self.path)

    def __repr__(self):
        header = getattr(self, 'header', '{} {}')
        hdr_str = header.format(self.type_spec.resource_type, self.type_spec.name)
        if self.exists:
            if self.is_folder:
                return f'{hdr_str}: directory\n'
            elif self.is_expression:
                return f'{hdr_str}: {self.format_expression()}\n'
            else:  # regular file(s)
                paths_str = '\n'.join([f'{" "*(len(hdr_str) + 2)}{p}' for p in self.all_existing_paths()])
                return f'{hdr_str}: {paths_str}\n'
        else:
            return f'{hdr_str}: no file\n'

    @property
    def expression(self):
        return self._expression or (self.type_spec.basename if self.type_spec.is_expression() else None)

    @expression.setter
    def expression(self, value):
        if isinstance(value, str):
            value = Expression(value)
        self._expression = value

    @property
    def status(self):
        status = self.status_manager.status if self.status_manager else None
        if status is True:
            status = 'debug'  # Must be a string to build the path
        return status

    def variant(self, sample_id=None, extension=None, version=None, expression=None):
        """
        Returns a variant of the asset with the given sample_id, extension and version.

        Parameters
        ----------
        sample_id : str
            The sample id to use.
        extension : str
            The extension to use.
        version : int
            The version to use.

        Returns
        -------
        Asset
            The variant of the asset.
        """
        if expression is not None:
            if extension:
                if isinstance(expression, Expression):
                    expression = expression.string()
                expression = Expression(f'{expression}{extension}')
            if not isinstance(expression, Expression):
                expression = Expression(expression)
        else:
            if extension:
                expression = Expression(str(self.with_extension(extension))) if self.is_expression else None
            else:
                expression = self.expression
        type_spec = deepcopy(self.type_spec)
        if not self.is_expression and extension:
            type_spec.extensions = list(dict.fromkeys([extension] + self.type_spec.extensions))
        # WARNING: long list, should use keyword arguments
        return Asset(self.base_directory, type_spec, self.channel_spec, expression,
                     sample_id or self.sample_id, self.subdirectory, version or self.version,
                     self._checksum, self.status_manager, self.extensions_to_directories)

    @property
    def name_parts(self):
        if self.type_spec.is_expression():
            return [self.type_spec.basename]
        parts = (self.status, self.sample_id,
                 self.channel_spec.name if self.channel_spec else None,
                 self.type_spec.name)  # WARNING: if only basename, no asset_sub_type
        return [part for part in parts if part]

    @property
    def base_name(self):  # TODO: private ?
        """
        The base name of the asset. This is the name of the file without the extension.
        This is essentially the concatenation of the specified name parts.

        Returns
        -------
        str
            The base name of the asset.
        """
        if isinstance(self.type_spec.basename, str) and self.type_spec.basename.startswith('/'):  # If absolute path, use basename
            name = self.type_spec.basename
        elif isinstance(self.type_spec.basename, Expression):
            if any(tag.ttype == 'S' for tag in self.type_spec.basename.tags):
                raise ClearMapAssetError(f'Expression {self.type_spec.basename} cannot contain S tags. '
                                         f'Found {[tag.name for tag in self.type_spec.basename.tags if tag.ttype == "S"]}'
                                         f'Expression should be parameterized first.')
            else:
                name = self.type_spec.basename.string()
        elif self.type_spec.name == 'raw':  # Use literal for raw because it does not follow ClearMap normalization
            name = self.expression.string()
        else:  # Otherwise, build the name
            name = ''
            for part in self.name_parts:
                if isinstance(part, (list, tuple)):  # channel could be a tuple for compound assets
                    part = '-'.join(part)
                if Expression(part).tags:
                    part = Expression(part)  # FIXME: substitute None width tags
                    values = {}
                    for tag in part.tag_names():
                        if hasattr(self, tag):
                            values[tag] = getattr(self, tag)
                    part = part.string(values=values)
                name += f'{part}_'
            if self.version is not None:
                name += f'v{self.version}_'
            name = name[:-1]  # Remove trailing _

        for ext in self.type_spec.extensions:  # Strip extension if present
            if name.endswith(ext):
                name = name[:-len(ext)]
                break

        return name

    @property
    def existing_extension(self):
        """
        Returns the first existing extension from the list of extensions.

        Returns
        -------
        str | None
            The first existing extension from the list of extensions.
            None if not found on disk.
        """
        # WARNING: do not use path here as it would be circular
        if self.is_folder:
            return None
        # Use tmp_path with an extension to allow with_suffix
        tmp_path = Path(f'{self.base_name}{self.type_spec.default_extension}')
        if not self.base_name.startswith('/'):
            tmp_path = self.directory / tmp_path
        existing_extension = file_utils.find_existing_extension(tmp_path, self.type_spec.extensions)
        return existing_extension

    @property
    def best_extension(self):
        """
        Returns the best extension for the asset.
        If the asset exists on disk with any known extension,
        it returns the first found extension in the order specified in the list of extensions.
        Otherwise, it returns the default extension.

        Returns
        -------
        str
            The best extension for the asset.
        """
        if self.is_expression:  # WARNING: do not use path here as it would be circular. Maybe better option
            return self.type_spec.default_extension
        else:
            return self.existing_extension or self.type_spec.default_extension

    @property
    def directory(self):
        sub_dir = Path(self.type_spec.directory)
        if sub_dir.is_absolute():
            return sub_dir
        else:
            if self.use_sub_dir:
                return self.base_directory / sub_dir
            else:
                base_name = self.base_name + self.type_spec.default_extension  # default_extension to avoid recursion
                if (self.base_directory / base_name).exists():  # compatibility with 3.0.0
                    return self.base_directory
                else:
                    self.use_sub_dir = True
                    return self.base_directory / sub_dir

    @property
    def file_name(self):
        if self.is_expression:
            f_name = self.expression.string()
            for ext in self.type_spec.extensions:
                if f_name.endswith(ext):
                    return f_name
            else:
                return f_name + self.best_extension
        else:
            return self.base_name + self.best_extension

    @property
    def path(self):
        """
        Returns the path of the asset with the best extension (whether it exists or not).

        Returns
        -------
        pathlib.Path
            The path with the best extension.
        """
        file_name = self.file_name  # minimize dynamic calls
        if str(file_name).startswith('/'):
            return Path(file_name)
        else:
            return self.directory / file_name

    @property
    def existing_path(self):
        """
        Returns the path of the asset with the first existing extension.

        Returns
        -------
        pathlib.Path
            The path with the first existing extension.
            If missing, try to decompress the asset.
            If nothing is found, raise a FileNotFound error.
        """
        path = None
        if not self.exists and not self.is_expression:
            if self.compressed_path.exists():
                self.decompress()
            else:
                sample_id = self.sample_id
                try:
                    self.sample_id = ''
                    if self.exists:
                        path = self.path
                        warnings.warn(f'Asset "{self.type_spec.name}" exists but without the sample id.')
                finally:
                    self.sample_id = sample_id
                if not path:
                    raise AssetNotFoundError(f'Asset "{self.type_spec.name}" does not exist. '
                                             f'Search location: "{self.path}"')
        else:
            path = self.path
        return path

    def all_existing_paths(self, sort=False):
        """
        Returns all existing paths of the asset.

        Parameters
        ----------
        sort : bool
            Whether to sort the paths.

        Returns
        -------
        list of pathlib.Path
            The paths with the first existing extension.
            If missing, try to decompress the asset.
            If nothing is found, raise a FileNotFound error.
        """
        paths = []
        for ext in self.type_spec.extensions:
            path = self.with_extension(ext)
            expression = Expression(path)
            if expression.n_tags():
                paths.extend(expression.glob(sort=sort))
            if path.exists():
                paths.append(path)
        if sort:
            paths = natsort.natsorted(paths)
        return [Path(p) for p in paths]  # Ensure we don't have a mix of strings and paths

    # FIXME: path_with_extension (not variant)
    def with_extension(self, extension):  # TODO: make explicit whether existing or not
        """
        Returns the path of the asset overwriting the default extension.

        Parameters
        ----------
        extension: str
            The extension to use.

        Returns
        -------
        pathlib.Path
            The path with the given extension whether it exists or not.
        """
        if not self.type_spec.extensions:
            raise ValueError(f'Asset {self.type_spec} has no extensions. It should be a directory.')
        if self.extensions_to_directories:
            path = Path(self.extensions_to_directories[extension]) / self.file_name.withext(extension)
            return path.with_suffix(extension)
        else:
            return self.path.with_suffix(extension)  # TODO: check if this works for expressions

    @property
    def exists(self):
        """
        Returns whether the asset exists on disk.

        Returns
        -------
        bool
            True if the asset exists, False otherwise.
        """
        return bool(file_utils.find_existing_extension(self.path, self.type_spec.extensions))

    def delete(self, missing_ok=False):
        if missing_ok and not self.exists:
            return
        os.remove(self.existing_path)

    @property
    def is_expression(self):
        return self.expression and self.expression.n_tags() > 0

    @property
    def is_tiled(self):
        if not self.is_expression:
            return False
        tags = [t.lower() for t in self.expression.tag_names()]
        return 'x' in tags or 'y' in tags

    @property
    def is_regular_file(self):
        return not self.is_folder and not self.is_expression

    @property
    def is_folder(self):
        return self.type_spec.is_folder

    @property
    def is_existing_source(self):  # FIXME: could exist compressed
        """
        Returns whether the asset is a source and exists on disk.

        .. warning:: This is not for e.g. tables.

        Returns
        -------
        bool
        """
        return self.exists and file_utils.is_clearmap_source_extension(self.existing_extension)

    @property
    def checksum(self):
        """
        Returns the checksum of the asset. This is computed if not already done.

        Returns
        -------
        str
            The checksum of the asset.
        """
        if not self._checksum:
            self._checksum = file_utils.checksum(self.existing_path, self.type_spec.checksum_algorithm)
        return self._checksum

    @property
    def size(self):
        return self.path.stat().st_size

    def read(self, *args, **kwargs):
        return clearmap_io.read(self.existing_path, *args, **kwargs)

    def write(self, data, *args, **kwargs):
        if isinstance(data, Graph):
            data.save(self.path)
        else:
            clearmap_io.write(self.path, data, *args, **kwargs)

    def create(self, *args, **kwargs):
        clearmap_io.create(self.path, *args, **kwargs)

    def as_source(self, slicing=None, *args, **kwargs):
        return clearmap_io.source(self.existing_path, slicing=slicing, *args, **kwargs)

    def shape(self):
        if not self.is_expression:
            return clearmap_io.shape(self.existing_path)

    def dtype(self):
        if self.is_existing_source and not self.is_expression:
            return clearmap_io.dtype(self.existing_path)

    def convert(self, new_extension, processes=None, verbose=False, **kwargs):
        if self.is_existing_source:
            clearmap_io.convert(self.existing_path, self.with_extension(new_extension),
                                processes=processes, verbose=verbose, **kwargs)

    def compress(self, algorithm=None):
        algorithm = algorithm or self.type_spec.compression_algorithms[0]
        file_utils.compress(self.existing_path, algorithm)

    def decompress(self, check=True):
        return file_utils.uncompress(self.compressed_path, check=check)

    @property
    def compressed_path(self):
        """
        Returns the path of the compressed asset.

        .. warning:: The compression extension is added to the original extension.

        Returns
        -------
        pathlib.Path
        """
        return self.with_extension(f'{self.path.suffix}.{self.type_spec.compression_algorithms[0]}')

    @property
    def pipeline(self):
        pipeline = CONTENT_TYPE_TO_PIPELINE.get(self.channel_spec.content_type)
        if pipeline is None:
            raise ValueError(f'Unknown data content type {self.channel_spec.content_type}. '
                             f'Cannot determine pipeline')
        return pipeline

    def plot(self, **kwargs):
        if self.is_existing_source:
            q_plot_3d.plot(self.existing_path, **kwargs)
        else:
            raise ValueError(f'Asset {self} is not a source. Cannot plot it.')

    def create_debug(self, slicing, status=None):
        if status is None:
            status = self.status_manager.status if self.status_manager.status is not None else 'debug'
        self.status_manager.status = False
        source = self.as_source()
        self.status_manager.status = status
        return self.write(np.asarray(source[slicing], order='F'))


class ExpressionAsset(Asset):
    """
    An asset that is composed of a list of files
    (e.g. tiled or stacked).
    """

    @property
    def is_parametrized(self):
        return not any([tag.ttype == 'S' for tag in self.expression.tags])

    def specify(self, values, extension=None):
        """
        Returns a new asset after specifying the values of the expression.

        Parameters
        ----------
        values : dict
            The values to specify.

        Example
        -------
        # >>> print(asset.type_spec.base_name)
        # '<moving_channel,S>_to_<fixed_channel,S>/result.<N,1>'
        # >>> parametrized_asset = asset.specify({'moving_channel': 'cfos', 'fixed_channel': 'autofluorescence'})
        # >>> print(parametrized_asset.type_spec.base_name)
        # 'cfos_to_autofluorescence/result.<N,1>'

        Returns
        -------
        Asset
            The specified asset.
        """
        expression = Expression(self.expression.string(values))
        extension = extension or self.best_extension
        if not expression.string().endswith(extension):
            expression = Expression(f'{expression.string()}{extension}')

        if any([tag.ttype == 'S' for tag in expression.tags]):
            raise ClearMapAssetError(f'Parametrised Expression {expression} should not contain S tags,'
                                     f' found {[tag.name for tag in expression.tags if tag.ttype == "S"]}')

        type_spec = deepcopy(self.type_spec)
        type_spec.basename = expression

        return Asset(self.base_directory, type_spec, self.channel_spec, expression, self.sample_id,
                     self.subdirectory, self.version, self._checksum, self.status_manager,
                     self.extensions_to_directories)

    @cached_property
    def file_list(self):
        """
        Returns the FileList source object associated with this pattern.

        Returns
        -------

        """
        return clearmap_io.file_list(self.path)  # We need path to have the folder

    def read(self, *args, **kwargs):
        raise NotImplementedError('Cannot read a tiled asset. This needs to be implemented in the FileList module')

    def write(self, data, *args, **kwargs):
        raise NotImplementedError('Cannot write a tiled asset. This needs to be implemented in the FileList module')

    def create(self, *args, **kwargs):
        raise NotImplementedError('Cannot create a tiled asset. This needs to be implemented in the FileList module')

    def shape(self):
        if self.is_tiled:
            raise ValueError(f'Asset {self} is tiled. Cannot determine shape without stitching.')
        else:
            tile_shape = clearmap_io.shape(self.file_list[0])
            return len(self.file_list), tile_shape[0], tile_shape[1]

    def convert(self, new_extension, processes=None, verbose=False, **kwargs):
        if self.is_existing_source:
            clearmap_io.convert_files(self.file_list, extension=new_extension,
                                      processes=processes, verbose=verbose, verify=True, **kwargs)

    @property
    def base_name(self):
        return Path(self.expression.string()).stem
    
    @property
    def existing_extension(self):
        return file_utils.find_existing_extension(self.file_list[0],
                                                  self.type_spec.extensions)

    def delete(self, missing_ok=False):
        if missing_ok and not self.exists:
            return
        for f in self.file_list:
            os.remove(f)

    @property  # FIXME: make function to avoid confusion with pathlib.Path.exists() (parenthesis and no parenthesis)
    def exists(self):
        """
        Whether all tiles exist on disk.

        Returns
        -------
        bool
            True if all tiles exist, False otherwise.
        """
        return self.all_tiles_exist

    @property
    def all_tiles_exist(self):  # TODO: try with all known extensions # REFACTOR: rename more generic (all images)
        """
        Whether all tiles exist on disk

        Returns
        -------
        bool
            True if all tiles exist, False otherwise.
        """
        # noinspection PyTypeChecker
        if not self.file_list:
            warnings.warn(f'No tiles found for channel {self.base_name} with expression {self.expression}')
            return False
        return len(self.file_list) == self.n_tiles

    @property
    def n_tiles(self):  # REFACTOR: rename more generic (n_images)
        """
        The number of tiles.

        Returns
        -------
        int
            The number of tiles.
        """
        n_tiles = len(self.file_list)
        return n_tiles if n_tiles > 0 else -1  # FIXME: should be theoretical number of tiles, not actual to compare with len(self.file_list)

    @property
    def tile_shape(self):  # FIXME: wont't work for stacked
        """
        The shape of single tiles.

        Returns
        -------
        tuple(int)
            The shape of the tiles.
        """
        return clearmap_io.shape(self.file_list[0])

    @property
    def tag_names(self):
        """
        The tag names of the tiles.

        Returns
        -------
        list(str)
            The tag names of the tiles.
        """
        return self.expression.tag_names()

    def format_expression(self):
        """
        Format the expression for display.

        The output is a string with the tag names and values of the expression.
        Of the type:
        [    data] raw           : Raw/raw_<X,2>_<Y,2>.npy {4 files, ('X', 'Y'): (0, 0) -> (1, 1)}

        Returns
        -------
        str
            The formatted expression.
        """
        # find all existing expressions using self.extensions_to_directories
        header = getattr(self, 'header', '{} {}')
        hdr_str = header.format(self.type_spec.resource_type, self.type_spec.name)
        tag_names = ', '.join(self.tag_names)
        tile_lower = tuple(np.min(self.positions, axis=0))
        tile_upper = tuple(np.max(self.positions, axis=0))
        pattern_line = (f'{hdr_str}: {self.expression.string()} {{{self.n_tiles} files,'
                        f' {tag_names}: {tile_lower} -> {tile_upper}}}')
        if self.extensions_to_directories:
            for ext, directory in self.extensions_to_directories.items():
                tmp_expression = Expression(Path(self.expression).with_suffix(ext))
                tmp_asset = Asset(directory, self.type_spec, self.channel_spec,
                                  tmp_expression, self.sample_id, self.subdirectory, self.version,
                                  self._checksum, self.status_manager)
                tmp_output = tmp_asset.format_expression()
                tmp_output = tmp_output.split(':')[1].strip()  # strip the header from tmp_asset output
                pattern_line += f' {tmp_output}'
        return pattern_line


class TiledAsset(ExpressionAsset):
    @property
    def n_tiles(self):  # FIXME: fix for stacked or result sequences
        """
        The number of tiles.

        Returns
        -------
        int
            The number of tiles.
        """
        return self.tile_grid_shape.prod()

    @property
    def tile_grid_shape(self):  # FIXME: wont't work for stacked or result sequences
        """
        The shape of the tile layout grid.

        Returns
        -------
        tuple(int)
            The shape of the tile grid.
        """
        # FIXME: limit to x and y
        indices = [tuple(tv[n] for n in self.tag_names) for tv in self.positions]
        if not indices:
            raise ValueError(f'No indices found for asset {self}. with expression {self.expression}')
        return np.array(indices).max(axis=0) + 1  # +1 because indexing from 0

    @property
    def positions(self):  # FIXME: wont't work for stacked
        """
        The positions of the tiles.

        Returns
        -------
        list(tuple)
            The positions of the tiles
        """
        return [self.expression.values(f) for f in self.file_list]



class AssetCollection:  # FIXME: fix how assets are retrieved
    """
    A collection of assets. This groups assets from a given sample that share the same channel
    (and therefore the same data content type).

    Attributes
    ----------
    base_directory: str | List[str]
        The base directory of the workspace.
    sample_id: str
        The sample id (e.g. the name of the sample, unique for an experiment).
    channel_spec: ChannelSpec
        The channel specification of the assets in the collection.
    """
    def __init__(self, base_directory, sample_id, channel_spec=None):
        """
        Initializes the asset collection.

        Parameters
        ----------
        base_directory: str
            The base directory of the workspace.
        sample_id: str
            The sample id (e.g. the name of the sample, unique for an experiment).
        channel_spec: ChannelSpec or dict, optional
            The channel specification of the assets in the collection.
            If dict, it should have the keys 'channel_name' and 'content_type'.
        """
        self.base_directory = base_directory
        self.sample_id = sample_id
        if isinstance(channel_spec, dict):
            channel_spec = ChannelSpec(**channel_spec)
        self.channel_spec = channel_spec
        self.assets = {}

    def __getitem__(self, item):
        try:
            return self.assets[item]
        except KeyError:
            raise KeyError(f'Asset "{item}" not found in collection with "{self.sample_id=}"'
                           f'{self.base_directory=}, {self.channel_spec=}')

    def get(self, item, default=None):
        try:
            return self.assets[item]
        except KeyError:
            return default

    def __setitem__(self, key, value):
        if not isinstance(value, Asset):
            raise ClearMapAssetError(f'Value must be an Asset. Got "{type(value)}" instead.')
        self.assets[key] = value

    def __contains__(self, item):
        return item in self.assets

    def __iter__(self):
        return iter(self.assets)

    def keys(self):
        return self.assets.keys()

    def values(self):
        return self.assets.values()

    def items(self):
        return self.assets.items()

    def update(self, assets):
        """
        Updates the collection with the given assets.

        Parameters
        ----------
        assets: iterable or dict of Asset or dict
        """
        if all([isinstance(asset, Asset) for asset in assets]):
            assets = {asset.name: asset for asset in assets}
        elif all([isinstance(asset, dict) for asset in assets.values()]):
            assets = {asset['name']: Asset(**asset) for asset in assets}
        elif all([isinstance(asset, Asset) for asset in assets.values()]):
            pass
        self.assets.update(assets)  # TODO: check if **assets

    def add_asset(self, type_spec: TypeSpec | None = None,
                  asset: Asset | None = None,
                  status_manager: StateManager | None = None,
                  **kwargs):  # FIXME: check what happens if asset exists in other form
        """
        Adds an asset to the collection.

        Parameters
        ----------
        type_spec : TypeSpec
            The type specification of the asset.
        asset : Asset
            The asset to add.
        status_manager : StateManager
            The status manager for the asset.
        kwargs : dict
            Additional parameters to pass to the Asset constructor.

        Returns
        -------

        """
        if type_spec and asset is None:
            asset = Asset(self.base_directory, type_spec,
                          channel_spec=self.channel_spec,
                          sample_id=self.sample_id,
                          status_manager=status_manager,
                          **kwargs)
        else:
            if asset is None:
                raise ClearMapAssetError('Either type_spec or asset must be provided.')
            else:
                if not isinstance(asset, Asset):
                    raise ClearMapAssetError(f'Asset must be an Asset. Got "{type(asset)}" instead.')
                # if asset.base_directory != self.base_directory:   # This could be a warning as it may be intended
                #     raise ClearMapAssetError(f'Asset base directory "{asset.base_directory}" '
                #                              f'does not match collection base directory "{self.base_directory}"')
                if asset.sample_id != self.sample_id:
                    raise ClearMapAssetError(f'Asset sample_id "{asset.sample_id}" '
                                             f'does not match collection sample_id "{self.sample_id}"')
                if asset.channel_spec != self.channel_spec:
                    raise ClearMapAssetError(f'Asset channel "{asset.channel_spec}" '
                                             f'does not match collection channel "{self.channel_spec}"')
        self.assets[asset.type_spec.name] = asset
        if asset.type_spec.sub_types:  # REFACTOR: this is ugly and should be dynamic from the asset
            for sub_type_name, sub_type in asset.type_spec.sub_types.items():  # Create subtypes
                asset_sub_type = Asset(asset.directory, sub_type, asset.channel_spec,
                                       expression=asset.expression, sample_id=self.sample_id,
                                       status_manager=asset.status_manager)
                self.assets[f'{asset.type_spec.name}_{sub_type_name}'] = asset_sub_type
