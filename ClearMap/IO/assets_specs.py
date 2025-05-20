"""
This module contains the specifications for the assets in the pipeline.
The specifications are used to create the assets and to
determine the file names and locations.
The specs passed to the asset are
`TypeSpec` objects and `ChannelSpec` objects.

`TypeSpec`
    contain the information about the type of asset
    (i.e. the step in the pipeline, e.g. stitched, layout, cells, density, binary...)

`ChannelSpec`
    contain the information about the channel
    (i.e. the labeling of the asset, e.g. cfos, dapi, autofluorescence, gfp...
    and the type of labelling e.g. nuclei, cells, vessels, veins, arteries, axons...).
"""
import ctypes
import multiprocessing
import warnings
from pathlib import Path

from ClearMap.IO.assets_constants import (EXTENSIONS, COMPRESSION_ALGORITHMS, CHECKSUM_ALGORITHMS,
                                          RESOURCE_TYPE_TO_FOLDER, DATA_CONTENT_TYPES)
from ClearMap.Utils.tag_expression import Expression, TAG_START
from ClearMap.Utils.utilities import validate_arg


class TypeSpec:
    """
    A specification for a type of asset. Not a concrete asset.
    This has all the information for a step in the pipeline regardless of the channel.
    """
    def __init__(self, resource_type: str | None = None, type_name: str | None = None,
                 sub_types: list[str] | dict | None = None, basename: str = '', file_format_category: str | None = None,
                 relevant_pipelines: list[str] | None = None, compression_algorithms: list[str] | None = None,
                 checksum_algorithm: str | None = None, extensions: list[str] | None = None):
        """
        A specification for a step in the pipeline. Not a concrete asset.

        Parameters
        ----------
        resource_type: str | None
            The type of resource. E.g. 'data', 'results', 'graphs'...
            This is used to determine the directory where the file should be stored.
            If empty, the file is stored in the root directory.
            If None, same as type_name. (For folder assets)
        file_format_category: str | None
            The file_type used to map to the list of possible extensions
            for the file. E.g. 'image', 'table', 'graph'...
            If None, the asset is a folder.
            **If extensions is not None, this is ignored.**
        relevant_pipelines: List[str] | None
            The list of pipelines that are relevant for this asset.
        type_name: str
            The name of the type.
            E.g. 'stitched', 'layout', 'background', 'resampled', 'cells', 'density', 'binary'...
            This is the name of the step, not necessarily the name of the file.
            It is used to compute the file name.
        sub_types: List[str] | Dict[str, SubTypeSpec] | None
            The subtypes of the asset. These are the substeps in the pipeline.
            E.g. `cleaned` in graph_cleaned.gt or `raw` for cells_raw.npy.
        basename: str
            The base name of the file. This is the name of the file without the extension and substep.
            It is linked to type_name but can be different. Leave empty if it is the same.
        compression_algorithms: List[str]
            The list of possible compression algorithms for the file (ordered by preference).
        checksum_algorithm: str
            The checksum algorithm used to check the integrity of the file.
        extensions: List[str] | None
            The list of possible extensions for the file. This should be determined by the file_type.
            This can be used to override the default extensions.
        """
        self._name = None
        if Expression(str(basename)).tags:
            basename = Expression(str(basename))
        self.basename = basename or type_name
        self.name = type_name
        if resource_type and resource_type not in RESOURCE_TYPE_TO_FOLDER:
            raise ValueError(f'Unknown resource type: {resource_type}')
        self.resource_type = resource_type
        self.relevant_pipelines = relevant_pipelines or []
        if isinstance(sub_types, (list, tuple)):
            sub_types = {k: None for k in sub_types}
        else:
            sub_types = sub_types or {}
        self.sub_types = sub_types

        self._file_format_category = file_format_category
        if extensions:
            self.extensions = extensions
        else:
            self.extensions = EXTENSIONS[file_format_category] if file_format_category else None
        compression_algorithms = compression_algorithms or COMPRESSION_ALGORITHMS
        self.compression_algorithms = compression_algorithms
        if checksum_algorithm is None:
            checksum_algorithm = CHECKSUM_ALGORITHMS[0]
        self.checksum_algorithm = validate_arg('checksum algorithm', checksum_algorithm, CHECKSUM_ALGORITHMS)

    def __str__(self):
        return f'{self.resource_type} - {self.basename}'

    def __repr__(self):
        return f'{self.__class__.__name__}({str(self)})'

    @property
    def basename(self):
        return self._basename

    @basename.setter
    def basename(self, value):
        if isinstance(value, str) and Expression(value).tags:
            value = Expression(value)
        self._basename = value

    def is_expression(self):
        return Expression.is_expression(self._basename)  # FIXME: why not self.basename (prop)

    def get_sub_type(self, sub_type_name, extensions=None, file_format_category=None, expression=None):
        sub_type = self.sub_types.get(sub_type_name)
        if sub_type is None:
            sub_type = self.add_sub_type(sub_type_name, extensions, file_format_category, expression)
        return sub_type

    def add_sub_type(self, sub_type_name, extensions=None, file_format_category=None, expression=None):
        """
        Adds or updates an asset subtype

        Parameters
        ----------
        sub_type_name
        extensions
        file_format_category
        expression

        Returns
        -------

        """
        if extensions is None and file_format_category is not None:
            extensions = EXTENSIONS[file_format_category]
        sub_type = SubTypeSpec(resource_type=self.resource_type,
                               relevant_pipelines=self.relevant_pipelines,
                               type_name=f'{self.name}_{sub_type_name}',
                               basename=expression or self.basename,
                               compression_algorithms=self.compression_algorithms,
                               checksum_algorithm=self.checksum_algorithm,
                               extensions=extensions or self.extensions)
        self.sub_types[sub_type_name] = sub_type
        return sub_type

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, value):
        self._name = value
        if not self.basename and value != 'raw':  # Skip raw because it has no default basename
            self.basename = value

    @property
    def is_folder(self):
        return self.extensions is None

    @property
    def directory(self):
        """
        The directory where the asset should be stored.
        Returns the directory based on the resource type. If empty, returns the root directory.

        Returns
        -------
        str
            The directory where the asset should be stored.
        """
        if self.resource_type == '':  # Experiment root dir if empty
            return ''
        else:
            return RESOURCE_TYPE_TO_FOLDER[self.resource_type or self.name]  # If None, defaults to self.name

    @property
    def default_extension(self):
        """
        The default extension for this asset (based on type and sub_type).

        Returns
        -------
        str
            The default extension for this asset.
        """
        return self.extensions[0] if self.extensions else ''


class SubTypeSpec(TypeSpec):
    """
    A specification for a subtype of an asset. Not a concrete asset.
    This has all the information for a substep in the pipeline regardless of the channel.
    """
    def __init__(self, **kwargs):
        """
        A specification for a substep in the pipeline. Not a concrete asset.

        Parameters
        ----------
        kwargs
            The same parameters as TypeSpec.
        """
        super().__init__(**kwargs)

    @property
    def main_type(self):
        return self.name.split('_')[0]


class ChannelSpec:
    """
    A specification for a channel. Not a concrete asset.
    This has all the information for a channel regardless of the type of asset.

    Attributes
    ----------
    name: str | List[str]
        The name of the channel. (Typically the name of the labeling, e.g.
        cfos, dapi, autofluorescence, gfp ...)
        If a list, the channel is a composite of the channels in the list.
    content_type: str
        The type of content in the channel.
        E.g. 'nuclei', 'cells', 'vessels'...
    number: int
        The number id of the channel.
    """
    channel_names = []

    def __init__(self, channel_name: str | list[str], content_type: str, channel_number: int | None = None):
        """
        A specification for a channel. Not a concrete asset.

        Parameters
        ----------
        channel_name: str | List[str]
            The name of the channel. (Typically the name of the labeling, e.g.
            cfos, dapi, autofluorescence, gfp ...)
            If a list, the channel is a composite of the channels in the list.
        content_type: str
            The type of content in the channel.
            E.g. 'nuclei', 'cells', 'vessels'...
        channel_number: int
            The number of the channel.
        """
        if channel_name not in ChannelSpec.channel_names:
            ChannelSpec.channel_names.append(channel_name)
        self.name = channel_name
        self.content_type = validate_arg('data content type', content_type, DATA_CONTENT_TYPES)
        self.number = len(ChannelSpec.channel_names) - 1 if channel_number is None else channel_number

    @classmethod
    def channel_number_to_name(cls, channel_number):
        return cls.channel_names[channel_number]


class StateManager:
    """
    A context manager to handle the state (e.g. debug mode).
    It is multiprocess safe using shared memory for the status.

    Attributes
    ----------
    _status: str
        The current context keyword.
    known_contexts: list[str]
        The list of known context keywords.
    """
    def __init__(self):
        self._status = multiprocessing.Array(ctypes.c_char, 20, lock=True)   # Shared memory for multiprocessing
        self.known_contexts = []

    def __call__(self, status=None):
        if status:
            if not isinstance(status, str):
                status = 'debug'
            self.status = status
        return self

    def __enter__(self):
        if not self.status:
            self.status = 'debug'
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.status = ''

    @property
    def status(self):
        with self._status.get_lock():
            val = self._status.value.decode('utf-8')
            if val.lower() in ('', 'none', 'null'):
                return None
            elif val.lower() == 'true':
                return True
            elif val.lower() == 'false':
                return False
            else:
                return val

    @status.setter
    def status(self, value):
        with self._status.get_lock():
            # if value not in self.known_contexts:
            #     raise ValueError(f'Unknown context: {value}')
            if isinstance(value, bool):
                value = str(value).lower()
            if value is None:
                value = ''
            if len(value) > 20:
                warnings.warn('Context keyword is too long, max 20 characters.')
                value = value[:20]
            self._status.value = value.encode('utf-8')
