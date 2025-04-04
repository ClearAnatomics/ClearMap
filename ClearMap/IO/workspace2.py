"""
This module is meant to replace the workspace module in ClearMap2.
The main difference is the support for an unlimited number of channels.
This is achieved by having a channel centered approach instead of a processing step centered approach.

The file types are now context dependant. There is no more "raw" or "stitched" files but the user defines
the channel prefix and the type (step) of the file so that there can be multiple files of the same type
depending on the channel.

The `file_type_to_name` dictionaries have thus been extensively modified to reflect this change.

The workspace is now based on a collection of Asset objects that handle the data themselves.
The workspace is now also fully stateful.

The "debug" mode is now a context manager that supports different contexts and is not restricted to debug mode.
"""
import os
import warnings
from copy import deepcopy
from pathlib import Path

import numpy as np

from ClearMap.IO.assets_constants import CONTENT_TYPE_TO_PIPELINE, CHANNELS_ASSETS_TYPES_CONFIG, RESOURCE_TYPE_TO_FOLDER
from ClearMap.IO.assets_specs import ChannelSpec, TypeSpec, StateManager
from ClearMap.IO.workspace_asset import Asset, AssetCollection
from ClearMap.Utils.exceptions import AssetNotFoundError, ClearMapWorkspaceError, ClearMapAssetError, \
    MissingChannelError
from ClearMap.Utils.utilities import substitute_deprecated_arg, handle_deprecated_args


def create_assets_types_config(type_spec_dict):
    assets_types_config = {}
    for name, spec in type_spec_dict.items():
        instance = TypeSpec(resource_type=spec.get('resource_type'),
                            type_name=spec.get('type_name', name),
                            sub_types=spec.get('sub_types'),
                            basename=spec.get('basename', ''),
                            file_format_category=spec.get('file_format_category'),
                            relevant_pipelines=spec.get('relevant_pipelines'),
                            extensions=spec.get('extensions'))
        assets_types_config[name] = instance

    subtypes = {}
    for name, spec in assets_types_config.items():
        for subtype in spec.sub_types.keys():  # Create subtypes
            if f'{name}_{subtype}' not in assets_types_config:
                subtypes[f'{name}_{subtype}'] = spec.get_sub_type(subtype)
    return {**assets_types_config, **subtypes}


CHANNEL_ASSETS_TYPES = create_assets_types_config(CHANNELS_ASSETS_TYPES_CONFIG)



class Workspace2:  # REFACTOR: subclass dict
    """
    A workspace manages asset collections for multiple channels.

    The assets can then be accessed through the get method.

    Attributes
    ----------

    directory: str | List[str]
        The base directory where all the assets are stored.
    asset_collections: dict
        A dictionary of AssetCollection objects.
        The keys are the channel ids. None for global assets
        (e.g. the logs).
    sample_id: str
        The id of the sample. This must be unique for each sample
        within an experiment.
    default_channel: int
        The default channel to use when no channel is specified.
    current_channel: int
        The channel currently in use.
    status_manager: ClearMap.IO.assets_specs.StateManager
        A context manager to handle the workspace state (e.g. debug mode).
    """
    def __init__(self, directory, default_channel=None, sample_id=None):
        self.directory = directory
        self.sample_id = sample_id
        self.asset_collections = {
            None: AssetCollection(self.directory, self.sample_id, None)  # Global assets
        }
        if default_channel is not None:
            warnings.warn('The default_channel argument is potentially dangerous.'
                          'Please use with caution. For a safer way, use the channel argument'
                          ' in the `get` method instead.')

            self.default_channel = default_channel
            self.current_channel = default_channel

        self.status_manager = StateManager()

    def __str__(self):
        out = self.__class__.__name__
        for v, brackets in ((self.sample_id, '()'),
                            (self.directory, '{}'),
                            (self.debug, '[]')):
            if v is not None:
                out += f'{brackets[0]}{v}{brackets[1]}'
        return out

    @property
    def debug(self):
        return self.status_manager.status

    @debug.setter
    def debug(self, value):
        self.status_manager.status = value

    def create_debug(self, asset_type, slicing, channel='auto', debug=None, **kwargs):
        """
        Create a debug file for the given asset type. This is typically a subset of the data
        (as defined by the slicing parameter) that is quickly accessible for debugging purposes.

        .. note::
        It is recommended to use self.get(asset_type, channel).create_debug(slicing, status) instead of this method.

        Parameters
        ----------
        asset_type
        slicing
        debug
        kwargs

        Returns
        -------

        """
        self.get(asset_type, channel=channel, **kwargs).create_debug(slicing, debug)

    @property
    def channels(self):
        return list(self.asset_collections.keys())

    def add_raw_data(self, file_path, channel_id=None,
                     data_content_type=None, sample_id=None):
        """
        Add a raw dataset to the workspace. This implies creating the corresponding assets
        for the given channel and data content type.

        Parameters
        ----------
        file_path: str
            The path to the raw data file.
        channel_id: str | None
            The channel id to use for the asset.
        data_content_type: str | None
            The content type of the data. This is used to determine the relevant pipelines.
        sample_id: str | None
            The sample id to use for the asset. If None, use the workspace sample_id.
        """
        # FIXME: hard coded channel name (maybe attribute in the workspace)
        sample_id = self.sample_id if sample_id is None else sample_id
        # FIXME: data_content_type is None if config was autogenerated
        if channel_id and data_content_type:
            channel_spec = ChannelSpec(channel_id, data_content_type)
        else:
            raise ClearMapWorkspaceError('A raw dataset cannot be added without a'
                                         ' channel_id and a data_content_type.')
        if channel_id in self.asset_collections.keys():
            raise ClearMapWorkspaceError(f'Channel {channel_id} already exists in the workspace.'
                                         f'Use update_raw_data explicitly instead.')
        self.asset_collections[channel_id] = (
            AssetCollection(self.directory, sample_id, channel_spec))
        raw_asset = Asset(self.directory, deepcopy(CHANNEL_ASSETS_TYPES['raw']),
                          channel_spec,
                          expression=file_path, sample_id=sample_id,
                          status_manager=self.status_manager)
        self.add_asset(raw_asset)

        pipelines = [CONTENT_TYPE_TO_PIPELINE[data_content_type]]
        # if raw_asset.is_expression:

        # FIXME: this shouldn't be the case for just stacking
        pipelines.append('stitching')  # WARNING: We need the "stitched" asset even for file conversion

        for name, spec in CHANNEL_ASSETS_TYPES.items():
            if any(p in pipelines for p in spec.relevant_pipelines):
                self.create_asset(spec, channel_spec, sample_id=sample_id)

    def add_channel(self, channel_spec, sample_id=''):
        self.asset_collections[channel_spec.name] = AssetCollection(self.directory, sample_id, channel_spec)

    def add_pipeline(self, pipeline_name, channel_id=None, **kwargs):
        """
        Add a pipeline to the workspace. This implies creating the corresponding assets
        for the given channel and pipeline.

        Parameters
        ----------
        pipeline_name: str
            The name of the pipeline to add.
        channel_id: str | None
            The channel id to use for the asset.
        """
        if channel_id not in self.asset_collections:
            raise MissingChannelError(f'Channel "{channel_id}" does not exist in the workspace.'
                                      f'Use add_raw_data to create a new channel.')
        if pipeline_name not in CONTENT_TYPE_TO_PIPELINE.values():
            raise ClearMapWorkspaceError(f'Pipeline {pipeline_name} does not exist in the workspace.')

        if 'sample_id' in kwargs:
            sample_id = kwargs.pop('sample_id')
        else:
            sample_id = self.get('raw', channel_id).sample_id
        channel_spec = self.asset_collections[channel_id].channel_spec
        for name, spec in CHANNEL_ASSETS_TYPES.items():
            if pipeline_name in spec.relevant_pipelines:
                self.create_asset(spec, channel_spec, sample_id=sample_id)

    def add_asset(self, asset):
        """
        Add an asset to the workspace. If the asset is channel specific, it is added to the
        corresponding channel collection (which is created if it does not exist).
        If the asset is global, it is added to the global collection.

        Parameters
        ----------
        asset: Asset
            The asset to add.
        """
        channel = asset.channel_spec.name if asset.channel_spec is not None else None
        if channel not in self.asset_collections:
            self.asset_collections[channel] = AssetCollection(self.directory, self.sample_id, asset.channel_spec)
        self.asset_collections[channel].add_asset(asset=asset)

    def create_asset(self, type_spec, channel_spec=None, sample_id=None):
        """
        Create an asset of the given type and channel.

        Parameters
        ----------
        type_spec: TypeSpec
            The type specification of the asset to create.
        channel_spec: ChannelSpec | None
            The channel to create the asset for.
        sample_id : str | None
            The sample id to create the asset for. If None, use the workspace sample_id.
            If the workspace sample_id is None, the assets will not use the sample_id
            as prefix in the file name.

        Returns
        -------
        asset: Asset
        """
        sample_id = self.sample_id if sample_id is None else sample_id
        asset = Asset(self.directory, type_spec, channel_spec, sample_id=sample_id,
                      status_manager=self.status_manager)
        self.add_asset(asset)
        return asset

    def get(self, asset_type, channel='current',
            asset_sub_type=None, sample_id=None,
            extension=None, version=None,
            status=None, debug=None,
            prefix=None, postfix=None,
            default='closest'):
        """
        Get the asset of the given type. If no exact match is found, try to find the closest match.

        Parameters
        ----------
        asset_type: str | TypeSpec
            The type of the asset. (e.g. 'raw', 'stitched', 'layout' ...)
        channel: str | None
            The channel to search in. If None, search for a global asset.
        asset_sub_type: str | None
            The subtype of the asset. (e.g. 'aligned', 'aligned_axis', 'placed' ...)
        sample_id: str | None
            The prefix of the asset name. (typically the sample id)
        extension: str | None
            The extension of the asset. This is typically set by the asset type.
        version: str | int | None
            The version of the asset.
        status: str | bool | None
            The status of the asset. If None, use the workspace status.
        debug: str | bool | None
            The debug mode of the asset. If None, use the workspace debug mode.
            .. deprecated:: 3.0.0
                Use the status argument instead.
        prefix: str | None
            The prefix of the asset name. (typically the sample id)
            .. deprecated:: 3.0.0
                Use the sample_id argument instead.
        postfix: str | None
            The postfix of the asset name. (typically the asset subtype)
            .. deprecated:: 3.0.0
                Use the asset_sub_type argument instead.

        Returns
        -------
        Asset object (or subclass)
        """

        if channel == 'current':
            channel = self.current_channel
        elif channel == 'default':
            channel = self.default_channel

        # Handle deprecated arguments  # FIXME: use decorator
        if debug is not None:
            status = substitute_deprecated_arg(debug, status, 'debug', 'status')
        if prefix is not None:
            sample_id = substitute_deprecated_arg(prefix, sample_id, 'prefix', 'sample_id')
        if postfix is not None:
            asset_sub_type = substitute_deprecated_arg(postfix, asset_sub_type, 'postfix', 'asset_sub_type')

        if status:
            self.debug = status

        if asset_sub_type:
            asset_type += f'_{asset_sub_type}'
        if channel not in self.asset_collections and isinstance(channel, (tuple, list)):
            channel = ('-'.join(channel)).lower()  # Try string version if tuple version not found
        if asset_type in self.asset_collections[channel]:
            asset = self.asset_collections[channel][asset_type]
        else: # asset is somehow None
            if default == 'closest':
                warnings.warn('No exact match found. Using partial matching from start.')
                asset = self.get_closest_matching_asset(asset_type, channel)
            else:
                return default
        if sample_id or extension or version:  # FIXME: subdirectory
            asset = asset.variant(sample_id, extension, version)
        return asset

    def get_closest_matching_asset(self, asset_type, channel=None):  # FIXME: compound keys
        """
        Get the asset that matches the given asset type the closest.
        The matching is done with either of the following methods:
        - There is a '*' or '?' in the asset type, use fnmatch to match the asset type.
        - The asset type starts with the given asset type.
        If one match is found, return it.
        If multiple matches are found, raise an exception.

        Parameters
        ----------
        asset_type: str
            The asset type to match
        channel: str | None
            The channel to search in. If None, search for a global asset.

        Returns
        -------
        Asset object
        """
        if '*' in asset_type or '?' in asset_type:
            import fnmatch
            matching_types = [k for k in self.asset_collections[channel].keys() if fnmatch.fnmatch(k, asset_type)]
        else:
            matching_types = [k for k in self.asset_collections[channel].keys() if k.startswith(asset_type)]
        if len(matching_types) == 0:
            raise KeyError(f'No asset of type {asset_type} found in workspace.')  # FIXME: ClearMapKeyError
        elif len(matching_types) == 1:
            asset = self.asset_collections[matching_types[0]]
        else:
            raise AssetNotFoundError(f'Multiple assets of type {asset_type} found in workspace '
                                     f'({matching_types}). Could not pick one.')
        return asset

    @handle_deprecated_args({'prefix': 'sample_id', 'postfix': 'asset_sub_type'})
    def filename(self, asset_type, channel=None,
                 sample_id=None, asset_sub_type=None,
                 subdirectory=None, extension=None, version=None):
        """
        Get the file path of the asset.

        .. warning::
            This method is here for compatibility reasons and is now considered deprecated.
            Please use the get method instead.
            e.g. workspace.get(asset_type, channel).path

        Parameters
        ----------
        asset_type: str
            The type of the asset. (e.g. 'raw', 'stitched', 'layout' ...)
        channel: str | None
            The channel to search in. If None, search for a global asset.
        sample_id: str | None
            The sample_id of the asset.
        asset_sub_type: str | None
            The subtype of the asset. (historically, the postfix)
        subdirectory: str | None
            The subdirectory where the asset is stored. This is normally set by the asset itself
            as a function of the asset type and the channel.
        extension: str | None
            The extension of the asset. This is typically set by the asset type.
        version: str | None
            The version of the asset.

        Returns
        -------
        str
        """
        asset = self.get(asset_type, channel, asset_sub_type=asset_sub_type, sample_id=sample_id,
                         version=version, extension=extension)
        if subdirectory:
            asset.subdirectory = subdirectory
        return asset.path

    @handle_deprecated_args({'prefix': 'sample_id', 'postfix': 'asset_sub_type'})
    def source(self, asset_type, channel=None,
                 sample_id=None, asset_sub_type=None,
                 subdirectory=None, extension=None, version=None, **kwargs):
        warnings.warn('The source method is deprecated. Use the get method instead and call source on the asset')
        asset = self.get(asset_type, channel, asset_sub_type=asset_sub_type, sample_id=sample_id,
                            version=version, extension=extension)
        if subdirectory:
            asset.subdirectory = subdirectory
        return asset.as_source(**kwargs)

    def plot(self, asset_type, channel=None, prefix=None, postfix=None,
                 subdirectory=None, extension=None, version=None):
        """
        Plot the asset.

        .. warning::
            This method is here for compatibility reasons and is now considered deprecated.
            Please use the get method instead.
            e.g. workspace.get(asset_type, channel).plot()

        Parameters
        ----------
        asset_type: str
            The type of the asset. (e.g. 'raw', 'stitched', 'layout' ...)
        channel: str | None
            The channel to search in. If None, search for a global asset.
        prefix: str | None
            The prefix of the asset name. (typically the sample id)
        postfix: str | None
            The postfix of the asset name. (typically the asset sub type)
        subdirectory: str | None
            The subdirectory where the asset is stored. This is normally set by the asset itself
            as a function of the asset type and the channel.
        extension: str | None
            The extension of the asset. This is typically set by the asset type.
        version: str | None
            The version of the asset.
        """
        asset = self.get(asset_type, channel, asset_sub_type=postfix, sample_id=prefix, version=version,
                         extension=extension)
        if subdirectory:
            asset.subdirectory = subdirectory

        asset.plot()

    def load(self, file_path):
        """Loads the workspace configuration from disk"""
        d = np.load(file_path)[0]
        self.__dict__.update(d)

    def save(self, file_path):
        """Saves the workspace configuration to disk"""
        # prevent np to add .npy to a .workspace file
        with open(file_path, "wb") as fid:
            np.save(fid, [self.__dict__])

    # def add_type_specs(self, *args, **kwargs):

    def info(self):
        """
        Pretty print the workspace information
        """
        out = f'{self}\n'
        out += 'directories:\n'

        len_dirtype = max([len(k) for k in RESOURCE_TYPE_TO_FOLDER.keys()])
        for resource_type, folder in RESOURCE_TYPE_TO_FOLDER.items():
            out += f'  [{resource_type : >{len_dirtype}}]: {folder}\n'

        out += 'assets:\n'

        len_f_type = max([len(k) for k in CHANNELS_ASSETS_TYPES_CONFIG.keys()])
        header = f'  [{{:{len_dirtype}}}] {{:{len_f_type}}}'

        for channel, asset_group in self.asset_collections.items():
            out += f'  Channel: {channel}\n'
            for asset_type, asset in asset_group.items():
                asset.header = header
                try:
                    out += f'{asset}'
                except ClearMapAssetError as e:
                    out += f'{asset.type_spec.basename}'   # FIXME:


def test_context_manager():
    ws = Workspace2(default_channel='auto')
    print(ws.default_channel)
    print(ws.debug)
    with ws.status_manager():
        print(ws.debug)
    with ws.status_manager(context_string='hello'):
        print(ws.debug)
    with ws.status_manager(context_string='world'):
        print(ws.debug)
    print(ws.debug)#


def setup_test_workspace(tmp_path):
    # Create directories and files for the fos channel
    fos_dir = tmp_path / "220324_fos_15-07-15"
    fos_dir.mkdir(parents=True, exist_ok=True)
    for x in range(3):
        for y in range(5):
            file_path = fos_dir / f"15-07-15_fos_Blaze[{y:02} x {x:02}]_C00.ome.tif"
            file_path.touch()

    # Create directories and files for the auto channel
    auto_dir = tmp_path / "220324_auto_17-34-31"
    auto_dir.mkdir(parents=True, exist_ok=True)
    for z in range(30):
        file_path = auto_dir / f"17-34-31_auto_Blaze_C00_xyz-Table Z{z:04}.ome.tif"
        file_path.touch()


def test_asset_creation():
    test_dir = Path('/tmp/test_workspace')
    setup_test_workspace(test_dir)
    ws = Workspace2(directory=test_dir,
                    default_channel='auto')
    auto_expr = '220324_auto_17-34-31/17-34-31_auto_Blaze_C00_xyz-Table Z<Z4>.ome.tif'
    raw_expr = '220324_fos_15-07-15/15-07-15_fos_Blaze[<Y2> x <X2>]_C00.ome.tif'
    ws.add_raw_data(auto_expr, channel_id='auto', data_content_type='reference')  # FIXME:
    ws.add_raw_data(raw_expr, channel_id='cfos', data_content_type='nuclei')
    assert ws.get('raw').channel_spec.name == 'auto'
    print(ws.asset_collections)
    raw_asset = ws.get('raw', 'cfos')
    if raw_asset.is_tiled:
        print(raw_asset.file_list[:min(raw_asset.n_tiles, 10)])
    assert raw_asset.channel_spec.name == 'cfos'
    assert raw_asset.expression.string() == raw_expr, print(f'Expressions do not match: {raw_asset.expression} != {raw_expr}')

    # FIXME: This is not working
    assert raw_asset.type_spec.relevant_pipelines == ['registration', 'stitching'], \
        f"Pipelines do not match: {raw_asset.type_spec.relevant_pipelines} vs {['registration', 'stitching']}"
    print(ws.get('stitched', 'auto').path)
    os.rmdir(test_dir)
