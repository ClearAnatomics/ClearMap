"""
Asset
=====

This module implements the Asset class that is used to represent the files and folders of a workspace.

The Asset class can represent Sources in which case it wraps the IO functions to read and write the data.
It is however not limited to Sources and can represent any file or folder of the workspace.
It also includes the following features:
    * awareness of the content type (e.g. nuclei, veins, arteries, cells, axons, ...) and hence the algorithms applicable.
    * awareness of the resource type (e.g. data, results, graphs, ...) and hence the sub folder to store the asset in.
    * alternate file names (shortcuts)
    * Transparent compression and decompression
    * alternate extensions.
    * checksums

In the future, it will also include:
    * status (debug, ...)
    * list of parent assets with the parametrised function to generate them

.. todo::
    Handle identities elegantly
    Add support for status (debug, ...) link workspace status to asset status
    Add support for alternate names (shortcut)
    Add list of parent assets with the parametrised function to generate them

.. done::
    Add support for multiple extensions (glob ?)
    Check for graphs and other data types
"""
__author__ = 'crousseau'
__license__ = 'GPLv3 - GNU General Pulic License v3 (see LICENSE.txt)'
__webpage__ = 'https://idisco.info'
__download__ = 'https://www.github.com/ChristophKirst/ClearMap2'


import os
from functools import cached_property
from pathlib import Path

import numpy as np

from ClearMap.IO import IO as clearmap_io
from ClearMap.IO import FileUtils as file_utils
from ClearMap.Visualization.Qt import Plot3d as q_plot_3d
from ClearMap.Utils.TagExpression import Expression


CONTENT_TYPE_TO_PIPELINE = {
    'reference': 'registration',
    'nuclei': 'CellMap',  # TODO: list ?
    'vessels': 'TubeMap',
    'veins': 'TubeMap',
    'arteries': 'TubeMap',
    # 'cells': 'CellMap',
    'axons': 'AxonMap',
}

DATA_CONTENT_TYPES = CONTENT_TYPE_TO_PIPELINE.keys()

RESOURCE_TYPE_DIR = {
    'data': 'data',
    'results': 'results',
    'graphs': 'graphs',
    'autofluorescence_to_reference': 'elastix_auto_to_reference',
    'resampled_to_autofluorescence': 'elastix_resampled_to_auto',
}

COMPRESSION_ALGORITHMS = ('gzip', 'bz2', 'zip', 'lzma')
CHECKSUM_ALGORITHMS = ('md5', 'sha1', 'sha256', 'sha512')

file_type_to_name = {

}
identities = {}
identities.update(
    stitched                  = "stitched.npy",
    layout                    = "layout.lyt",
    background                = "background.npy",
    resampled                 = "resampled.tif",
    autofluorescence_resampled="resampled_autofluorescence.tif",
    resampled_to_autofluorescence='elastix_resampled_to_auto',
    resampled_aligned_to_autofluorescence="elastix_resampled_to_auto/result.0.mhd",
    autofluorescence_to_reference='elastix_auto_to_reference',
    autofluorescence_aligned_to_reference="elastix_auto_to_reference/result.1.mhd",
    cells = "cells.npy",
    density = "density.tif",
    arteries = "Raw/arteries_<X,2>_Y,2>.npy",
    binary = "binary.npy",
    binary_status = "binary_status.npy",
    skeleton = 'skeleton.npy',
    graph = "graph.gt",
)
synonyms = []


def validate_arg(arg_name, value, valid_values):
    if value not in valid_values:
        raise ValueError(f'Unknown {arg_name} "{value}". '
                         f'Supported values are "{valid_values}".')
    return value


def find_existing_extension(path, extensions):
    for extension in extensions:
        if Path(path).with_suffix(extension).exists():
            return extension


def check_extensions(extensions):
    if extensions is not None:
        for ext in extensions:
            if not ext.startswith('.'):
                raise ValueError(f'Extension "{ext}" should start with a dot.')
            if ext[1:] not in clearmap_io.file_extension_to_module.keys():
                raise ValueError(f'Unknown extension "{ext[1:]}". '
                                 f'Supported extensions are "{clearmap_io.file_extension_to_module.keys()}".')


class Asset:
    def __init__(self, asset_type, base_directory, extensions, sub_type=None, sample_id='', channel=None,
                 data_content_type='', resource_type='', subdirectory='', version=None,
                 checksum_algorithm='md5', checksum='', compression_algorithm='gzip'):
        """
        An asset is a file or a folder that is part of a workspace.

        Arguments
        ---------
        asset_type : str
            The name of the asset. This is the name of the step, not necessarily the name of the file.
            It is used to compute the file name.
        base_directory : str
            The base directory of the asset. This is the root directory of the workspace.
        extensions : list[str] or None
            The list of extensions that can be substituted to get the actual file name.
            If None, the asset is a folder.
        sample_id : str
            An optional sample id that can be used to identify the sample.
            If provided, it will be added as a prefix to the file name.
        data_content_type : str
            The type of data contained in the asset. One of (nuclei, veins, arteries, cells, axons, ...).
        resource_type : str
            The type of resource from (data, results ...).
            This is used to determine the sub folder to store the asset in.
        subdirectory : str
            An optional subdirectory to store the asset in. This would override the value computed from
            the resource type.
            .. warning:: The preferred way to store assets is to use the resource type.
        checksum_algorithm : str
            The algorithm used to compute the checksum of the asset. One of (md5, sha1, sha256, sha512).
        checksum : str
            The checksum of the asset. If not provided, it will be computed from the file.
        compression_algorithm : str
            The algorithm used to compress the asset. One of (gzip, bz2, zip, lzma).
        """
        self.name = asset_type
        self.sub_name = sub_type  # (e.g. stitched, resampled, reduced, annotated, ...)
        self.base_directory = base_directory
        self.sample_id = sample_id
        self.channel = channel
        self.version = version  # (e.g. 1, 2, ...)
        check_extensions(extensions)
        self.extensions = extensions

        self.expression = expression

        self.data_content_type = validate_arg('data content type', data_content_type, DATA_CONTENT_TYPES)
        self.resource_type = validate_arg('resource type', resource_type, RESOURCE_TYPE_DIR.keys())
        self.subdirectory = subdirectory

        self._checksum = checksum
        self.checksum_algorithm = validate_arg('checksum algorithm', checksum_algorithm, CHECKSUM_ALGORITHMS)
        self.compression_algorithm = validate_arg('compression algorithm',
                                                  compression_algorithm, COMPRESSION_ALGORITHMS)
        self.status = None  # FIXME: use status
        # TODO: add list of parent assets with the parametrised function to generate them

    @property
    def file_name(self):
        if self.expression:
            return self.expression
        else:
            return self.base_name + self.best_extension

    @property
    def type_name(self):
        return file_type_to_name.get(self.name, self.name)

    @property
    def name_parts(self):
        parts = [part for part in (self.status, self.sample_id, self.type_name, self.sub_name) if part is not None]
        if self.channel is not None:
            parts.insert(1, self.channel)  # FIXME: channel could be a list for compound assets
        return parts

    @property
    def base_name(self):  # TODO: check if keep
        base_name = ''
        for part in self.name_parts:
            if part:
                if isinstance(part, list):
                    part = '_'.join(part)
                base_name += f'{part}_'
        if self.version is not None:
            base_name += f'v{self.version}_'
        base_name = base_name[:-1]  # Remove trailing _
        return base_name

    @property
    def directory(self):
        if self.subdirectory is not None:
            sub_dir = self.subdirectory
        else:
            sub_dir = RESOURCE_TYPE_DIR[self.resource_type]
        return os.path.join(self.base_directory, sub_dir)

    @property
    def path(self):
        return os.path.join(self.directory, self.file_name)

        # if f_type == 'expression' or expression is not None:  # FIXME:
        #     f_name = Expression(expression).string(values=values)

    @property
    def existing_extension(self):
        tmp_path = os.path.join(self.directory, f'{self.base_name}{self.extensions[0]}')
        existing_extension = find_existing_extension(tmp_path, self.extensions)
        return existing_extension

    @property
    def best_extension(self):
        return self.existing_extension or self.extensions[0]

    def get_path_with_extension(self, extension):
        """
        Returns the path of the asset overwriting the default extension.

        Parameters
        ----------
        extension

        Returns
        -------

        """
        if not self.extensions:
            raise ValueError(f'Asset {self.name} has no extensions. It should be a directory.')
        return self.path.replace(self.extensions[0], extension)  # TODO: use PathLib

    @property
    def existing_path(self):
        if not self.exists:
            if os.path.exists(self.compressed_path):
                self.decompress()
                return self.compressed_path
            else:
                raise FileNotFoundError(f'Asset "{self.name}" does not exist. Search location: "{self.path}"')
        else:
            return self.path

    @property
    def exists(self):
        return bool(find_existing_extension(self.path, self.extensions))

    @property
    def is_expression(self):
        if self.expression:
            return self.expression.ntags() > 0

    @property
    def is_existing_source(self):
        return self.exists and self.existing_extension.lstrip('.') in clearmap_io.file_extension_to_module.keys()

    @property
    def checksum(self):
        if not self._checksum:
            self._checksum = file_utils.checksum(self.path, self.checksum_algorithm)
        return self._checksum

    def read(self):
        return clearmap_io.read(self.path)

    def write(self, data):
        clearmap_io.write(self.path, data)

    def create(self):
        clearmap_io.create(self.path)

    def compress(self):
        file_utils.compress(self.path, self.compression_algorithm)

    def decompress(self):
        return file_utils.uncompress(self.compressed_path, self.path)

    @property
    def compressed_path(self):
        return f'{os.path.splitext(self.path)[0]}.{self.compression_algorithm}'  # TODO: use PathLib

    @property
    def pipeline(self):
        pipeline = None
        # TODO: use dict
        if self.data_content_type is not None:
            pipeline = CONTENT_TYPE_TO_PIPELINE.get(self.data_content_type)
            if pipeline is None:
                raise ValueError(f'Unknown data content type {self.data_content_type}. Cannot determine pipeline')
        return pipeline

    def plot(self):
        if self.is_existing_source:
            q_plot_3d.plot(self.existing_path)

    def convert(self, new_extension):
        if self.is_existing_source:
            clearmap_io.convert(self.path, new_extension)

    def __str__(self):
        return self.path


class TiledAsset(Asset):
    """
    An asset that is tiled.
    """
    def __init__(self, pattern, *args, **kwargs):
        self.pattern = pattern
        super().__init__(*args, **kwargs)

    @cached_property
    def file_list(self):
        """
        Returns the FileList source object associated with this pattern.

        Returns
        -------

        """
        return clearmap_io.file_list(self.pattern)  # TODO: check self.file_name or self.pattern

    @property
    def exists(self):
        return self.all_tiles_exist

    @property
    def all_tiles_exist(self):
        # noinspection PyTypeChecker
        return len(self.file_list) == self.n_tiles

    @property
    def n_tiles(self):
        return self.tile_shape().prod()

    @property
    def tile_shape(self):
        indices = [tuple(tv[n] for n in self.tag_names) for tv in self.positions]
        return np.array(indices).max(axis=0) + 1  # +1 because indexing from 0

    @property
    def positions(self):
        return [self.expression.values(f) for f in self.file_list]

    @property
    def tag_names(self):
        return self.expression.tag_names()


class AssetCollection:
    def __int__(self, base_directory, sample_id, data_content_type, channel=None):
        self.base_directory = base_directory
        self.sample_id = sample_id
        self.data_content_type = data_content_type
        self.channel = channel  # FIXME: channel could be a list for compound assets

        self.assets = {}

    def __getitem__(self, item):
        return self.assets[item]

    def __setitem__(self, key, value):
        if not isinstance(value, Asset):
            raise ValueError(f'Value must be an Asset. Got "{type(value)}" instead.')
        self.assets[key] = value

    def __contains__(self, item):
        return item in self.assets

    def __iter__(self):
        return iter(self.assets)

    def update(self, assets):
        if all([isinstance(asset, Asset) for asset in assets]):
            assets = {asset.name: asset for asset in assets}
        elif all([isinstance(asset, dict) for asset in assets.values()]):
            assets = {asset['name']: Asset(**asset) for asset in assets}
        elif all([isinstance(asset, Asset) for asset in assets.values()]):
            pass
        self.assets.update(assets)  # TODO: check if **assets

    def add(self, asset_type, extensions, resource_type, asset_sub_type=None, **kwargs):
        """
        Adds an asset to the collection.

        Parameters
        ----------
        asset_type : str
            The name of the asset. This is the name of the step, not necessarily the name of the file.
            It is used to compute the file name.
        extensions : list[str] or None
            The list of extensions that can be substituted to get the actual file name.
            If None, the asset is a folder.
        resource_type : str
            The type of resource from (data, results ...).
        asset_sub_type : str
            The sub type of the asset. This is used to compute the file name.

        All the other parameters are passed to the Asset constructor.

        Returns
        -------

        """
        asset = Asset(asset_type, self.base_directory, extensions, sub_type=asset_sub_type,
                      sample_id=self.sample_id, channel=self.channel, data_content_type=self.data_content_type,
                      resource_type=resource_type, **kwargs)

        self.assets[(asset_type, asset_sub_type, kwargs.get('version'))] = asset
