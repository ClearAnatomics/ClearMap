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
from __future__ import annotations

import os
import warnings
from copy import deepcopy
from pathlib import Path
from typing import List, Iterator, Sequence, Optional

import numpy as np

from ClearMap.IO.assets_constants import CONTENT_TYPE_TO_PIPELINE, CHANNELS_ASSETS_TYPES_CONFIG, RESOURCE_TYPE_TO_FOLDER
from ClearMap.IO.assets_specs import ChannelSpec, TypeSpec, StateManager, ChannelId
from ClearMap.IO.workspace_asset import Asset, AssetCollection
from ClearMap.Utils.exceptions import (ClearMapWorkspaceError, ClearMapAssetError, MissingChannelError,
                                       MissingAssetError)
from ClearMap.Utils.utilities import substitute_deprecated_arg, handle_deprecated_args, get_ok_n_ok_symbols

SCHEMA_V_STR = 'clearmap_workspace_v2'


def _build_asset_types(type_spec_dict: dict, resource_type_to_folder: dict) -> dict[str, TypeSpec]:
    """
    Convert the config dict (CHANNELS_ASSETS_TYPES_CONFIG) into
    concrete TypeSpec objects, using the given resource_type_to_folder
    mapping and auto-creating flattened subtypes.
    """
    assets_types_config = {}
    for name, spec in type_spec_dict.items():
        instance = TypeSpec(resource_type=spec.get('resource_type'),
                            type_name=spec.get('type_name', name),
                            sub_types=spec.get('sub_types'),
                            basename=spec.get('basename', ''),
                            file_format_category=spec.get('file_format_category'),
                            resource_type_to_folder=resource_type_to_folder,
                            relevant_pipelines=spec.get('relevant_pipelines'),
                            sub_folder=spec.get('sub_folder'),
                            compression_algorithms=spec.get('compression_algorithms'),
                            checksum_algorithm=spec.get('checksum_algorithm'),
                            extensions=spec.get('extensions'))
        assets_types_config[name] = instance

    subtypes = {}
    for name, spec in assets_types_config.items():
        for st_name in spec.sub_types.keys():  # Create subtypes
            key = f'{name}_{st_name}'
            if key not in assets_types_config:
                subtypes[key] = spec.get_sub_type(st_name)
    return {**assets_types_config, **subtypes}


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
    def __init__(self, directory: str | Path, default_channel: str | None = None, sample_id: str | None = None,
                 resource_type_to_folder: dict | None = None, assets_types_config: dict | None = None):
        self.directory = str(directory)  # TODO: support pathlib.Path?
        self.sample_id = sample_id

        self.resource_type_to_folder = deepcopy(RESOURCE_TYPE_TO_FOLDER)
        if resource_type_to_folder:
            self.resource_type_to_folder.update(resource_type_to_folder)

        raw_config = deepcopy(assets_types_config or CHANNELS_ASSETS_TYPES_CONFIG)
        self.asset_types = _build_asset_types(raw_config, self.resource_type_to_folder)

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

    def __contains__(self, channel: ChannelId) -> bool:
        return channel in self.asset_collections

    def __len__(self) -> int:
        """
        Number of channels in the workspace (excluding global None).
        Returns
        -------
        int
            The number of channels in the workspace.
        """
        return sum(1 for k in self.asset_collections.keys() if k is not None)

    def __iter__(self) -> Iterator[ChannelId]:
        """
        Iterate non global channel names in the workspace.
        Returns
        -------
        Iterator[str]
            An iterator over the channel names in the workspace.
        """
        return (k for k in self.asset_collections.keys() if k is not None)

    def items(self):
        return self.asset_collections.items()

    def __getitem__(self, key):
        """
        ws[channel] -> AssetCollection
        """
        return self.asset_collections[key]

    def _iter_channel_specs(self) -> list[ChannelSpec]:
        specs = []
        for ch_id, col in self.asset_collections.items():
            if ch_id is None:
                continue
            if col.channel_spec not in specs:
                specs.append(col.channel_spec)
        return specs

    def to_dict(self) -> dict:
        """
        Serialize the logical structure of the workspace:
        - workspace metadata (directory, sample_id)
        - resource_type -> folder layout
        - asset type specs (including subfolders)
        - channels (names, content_type, number)
        """

        # Only top-level types, or all? For readability, I’d include all, it’s still small.
        asset_types_dict = {name: spec.to_dict()
                            for name, spec in self.asset_types.items()}

        channels_dict = [ch.to_dict() for ch in self._iter_channel_specs()]

        return {
            'schema': 'clearmap_workspace_v2',
            'directory': str(self.directory),
            'sample_id': self.sample_id,
            'default_channel': getattr(self, 'default_channel', None),
            'resource_type_to_folder': dict(self.resource_type_to_folder),
            'asset_types': asset_types_dict,
            'channels': channels_dict,
        }

    def to_yaml(self, path: str | Path):
        import yaml
        path = Path(path)
        with path.open('w', encoding='utf-8') as f:
            yaml.safe_dump(self.to_dict(), f, sort_keys=False)

    @classmethod
    def from_dict(cls, data: dict) -> "Workspace2":
        """
        Rebuild a workspace *layout* from a dict produced by to_dict().
        """
        schema = data.get('schema')
        if schema not in (None, SCHEMA_V_STR):
            raise ValueError(f"Unsupported workspace schema {schema}")

        types_cfg = data.get('asset_types')

        # Create instance with injected layout, but we’ll overwrite asset_types after
        ws = cls(
            directory=data['directory'],
            sample_id=data.get('sample_id'),
            default_channel=data.get('default_channel'),
            resource_type_to_folder=(data.get('resource_type_to_folder')),
            assets_types_config=types_cfg,
        )

        # Re-hydrate asset types from the dict (using same layout)
        ws.asset_types = {
            name: TypeSpec.from_dict(spec_dict)
            for name, spec_dict in types_cfg.items()
        }

        # Recreate channels
        for ch_data in data.get('channels', []):
            ch_spec = ChannelSpec.from_dict(ch_data)
            ws._add_channel(ch_spec, sample_id=ws.sample_id)

        return ws

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Workspace2":
        import yaml
        path = Path(path)
        with path.open('r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)

    def raw(self, channel: str):
        """Convenience: return the raw Asset for a channel, or None."""
        assets_collection = self.asset_collections.get(channel)
        return None if assets_collection is None else assets_collection.get('raw')

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
        return self.get(asset_type, channel=channel, **kwargs).create_debug(slicing, debug)

    @property
    def channels(self):  # FIXME: check if we should exclude None
        return list(self.asset_collections.keys())

    def ensure_default_channel(self, allowed_channels: List[str], default_channel: str):
        if default_channel and (not self.default_channel or self.default_channel not in allowed_channels):
            self.default_channel = default_channel

    def prune_missing_channels(self, desired_channels: List[str]):
        desired_channels = set(desired_channels)
        for channel in self.channels:
            if channel is None:  # Keep global assets
                continue
            elif channel not in desired_channels:
                channel_spec = self[channel].channel_spec
                if channel_spec.is_simple_channel():  # Simple channels -> direct removal
                    self.asset_collections.pop(channel)
                elif channel_spec.is_compound():  # Compound channels -> check components
                    channels = channel.split('-') if isinstance(channel, str) else channel
                    has_obsolete_components = any([c not in desired_channels for c in channels])
                    if has_obsolete_components:
                        self.asset_collections.pop(channel)


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
        if channel_id in self:
            raise ClearMapWorkspaceError(f'Channel {channel_id} already exists in the workspace.'
                                         f'Use update_raw_data explicitly instead.')
        self.asset_collections[channel_id] = (
            AssetCollection(self.directory, sample_id, channel_spec))
        raw_asset = Asset(self.directory, deepcopy(self.asset_types['raw']),
                          channel_spec,
                          expression=file_path, sample_id=sample_id,
                          status_manager=self.status_manager)
        self.add_asset(raw_asset)

        pipelines = [CONTENT_TYPE_TO_PIPELINE[data_content_type]]
        # if raw_asset.is_expression:

        # FIXME: this shouldn't be the case for just stacking
        pipelines.append('stitching')  # WARNING: We need the "stitched" asset even for file conversion

        for name, spec in self.asset_types.items():
            if any(p in pipelines for p in spec.relevant_pipelines):
                self.create_asset(spec, channel_spec, sample_id=sample_id)

    def update_raw_path(self, channel, expression):
        old_asset = self.raw(channel)
        if not old_asset:
            raise MissingChannelError(f'Channel "{channel}" does not exist in the workspace.'
                                      f'Use add_raw_data to create a new channel.')
        if old_asset.expression != expression:
            self.asset_collections[channel]['raw'] = old_asset.variant(expression=expression)

    def sync_resource_type_to_folder(self, desired: dict | None, *, migrate: bool = False,
                                     dry_run: bool = False, ) -> dict[str, tuple[Path, Path]]:
        """
        Synchronize this workspace's resource_type_to_folder with the new `desired` layout.
        - If no new mapping or identical -> no-op.
        - If mappings differ:
            * Build a plan of directory moves (per resource_type).
            * If `migrate=False` and any source dir exists, raise.
            * If `dry_run=True`, return the plan but don't touch disk.
            * Otherwise, rename/move the folders and update the mapping.

        Returns
        -------
        dict[resource_type, (old_dir, new_dir)]
            The directory-level migration plan (executed or planned).
        """
        if desired is None:
            return {}

        current = self.resource_type_to_folder
        # Fast path: exact same object
        if current is desired:
            return {}
        elif current == desired:  # If equality, ensure identity
            self.resource_type_to_folder = desired
            for ts in self.asset_types.values():
                ts.resource_type_to_folder = self.resource_type_to_folder
            return {}

        root = Path(self.directory)

        # Compute changes per resource_type
        changed: dict[str, tuple[Path, Path]] = {}
        for rtype, old_folder in current.items():
            new_folder = desired.get(rtype, old_folder)
            if old_folder != new_folder:
                changed[rtype] = (root / old_folder, root / new_folder)

        if dry_run:
            return changed

        any_source_exists = any(old_dir.exists() and old_dir != new_dir
                                for (old_dir, new_dir) in changed.values())

        if any_source_exists and not migrate:
            raise ClearMapWorkspaceError('Changing resource_type_to_folder would affect existing on-disk folders. '
                                         'Call sync_resource_type_to_folder(..., migrate=True) or dry_run=True to inspect the plan.')

        # Migrate
        for rtype, (old_dir, new_dir) in changed.items():
            if not old_dir.exists() or old_dir == new_dir:
                continue

            new_dir.parent.mkdir(parents=True, exist_ok=True)

            if not new_dir.exists():  # Just rename
                old_dir.rename(new_dir)
            else:  # Merge w error on name conflicts
                for child in old_dir.iterdir():
                    target = new_dir / child.name
                    if target.exists():
                        raise ClearMapWorkspaceError(f'Cannot migrate folder layout for resource_type={rtype}: '
                                                     f'target already exists: {target}')
                    child.rename(target)
                try:
                    old_dir.rmdir()  # Remove if now empty
                except OSError:
                    pass

        # Now update mapping and rewire TypeSpecs to the workspace dict
        current.clear()
        current.update(desired)
        for ts in self.asset_types.values():
            # make sure all TypeSpecs point to the live mapping
            ts.resource_type_to_folder = current

        return changed

    def _normalize_channel(self, channel: ChannelId) -> ChannelId:
        # Convert list -> tuple, leave str/tuple intact
        if isinstance(channel, list):
            return tuple(channel)
        return channel

    def _is_compound(self, channel: ChannelId) -> bool:
        return isinstance(channel, tuple) and len(channel) > 1

    def _permute_channels(self, channel: ChannelId) -> ChannelId:
        if not self._is_compound(channel):
            raise NotImplementedError(f'Channel permutation is only implemented for compound channels.')
            return channel
        return tuple(reversed(channel))

    def ensure_channel(self, channel_id: ChannelId, channel_content_type: str,
                       sample_id: str, permute_channels: bool):
        channel = self._normalize_channel(channel_id)
        sample_id = sample_id or self.sample_id
        created = []

        def _ensure_one(ch):
            if ch not in self.asset_collections:
                spec = ChannelSpec(ch, channel_content_type)
                self._add_channel(spec, sample_id=sample_id)
                return spec
            return self[ch].channel_spec

        # forward
        created.append(_ensure_one(channel))

        # optionally inverted
        if permute_channels:
            inv = self._permute_channels(channel)
            if inv != channel:
                created.append(_ensure_one(inv))

        return created

    def ensure_pipeline(self, pipeline_name: str, channel_id: ChannelId, sample_id: str,
                        permute_channels: bool=False, create_channel: bool=False, channel_content_type: Optional[str] = None):
        channel_id = self._normalize_channel(channel_id)  # FIXME: are we sure about that?
        sample_id = sample_id or self.sample_id

        if create_channel:
            if channel_content_type is None:
                raise ValueError("channel_content_type required when create_channel=True")
            self.ensure_channel(channel_id, channel_content_type,
                                sample_id=sample_id,
                                permute_channels=permute_channels)

        self.add_pipeline(pipeline_name, channel_id, sample_id=sample_id)
        if permute_channels:
            inv = self._permute_channels(channel_id)
            if inv != channel_id:
                if inv not in self:
                    raise MissingChannelError(f'Channel "{inv}" does not exist in the workspace.'
                                              f'If you want to implicitly create it, set create_channel=True.')
                self.add_pipeline(pipeline_name, inv, sample_id=sample_id)


    def _add_channel(self, channel_spec, sample_id=''):
        self.asset_collections[channel_spec.name] = AssetCollection(self.directory, sample_id, channel_spec)

    def add_pipeline(self, pipeline_name: str, channel_id: Optional[str | Sequence[str]] = None, **kwargs):
        """
        Add a pipeline to the workspace. This implies creating the corresponding assets
        for the given channel and pipeline.

        Parameters
        ----------
        pipeline_name: str
            The name of the pipeline to add.
        channel_id: str | Sequence[str] | None
            The channel id to use for the asset.
        """
        if channel_id not in self:
            raise MissingChannelError(f'Channel "{channel_id}" does not exist in the workspace.'
                                      f'Use add_raw_data to create a new channel.')
        if pipeline_name not in CONTENT_TYPE_TO_PIPELINE.values():
            raise ClearMapWorkspaceError(f'Pipeline {pipeline_name} does not exist in the workspace.')

        if 'sample_id' in kwargs:
            sample_id = kwargs.pop('sample_id')
        else:
            sample_id = self.get('raw', channel_id).sample_id
        channel_spec = self[channel_id].channel_spec
        for name, spec in self.asset_types.items():
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
        if channel not in self:
            self.asset_collections[channel] = AssetCollection(self.directory, self.sample_id, asset.channel_spec)
        self[channel].add_asset(asset=asset)

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

    def rename_channel(self, old_name, new_name):
        if old_name in self:
            asset_collection = self.asset_collections.pop(old_name)
            asset_collection.channel_spec.name = new_name
            self.asset_collections[new_name] = asset_collection
            name_map = {old_name: new_name}
            # REFACTOR: that should be automatic on the ChannelSpec level
            ChannelSpec.channel_names[:] = [name_map.get(n, n) for n in ChannelSpec.channel_names]

    def get(self, asset_type, channel='current',
            asset_sub_type=None, sample_id=None,
            extension=None, version=None,
            status=None, debug=None,
            prefix=None, postfix=None,
            suffix='',
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
        suffix: str
            The suffix of the asset name. This is to be used for assets where the subtype is
            dynamic.

        Returns
        -------
        Asset object (or subclass)
        """

        if channel == 'current':
            channel = self.current_channel
        elif channel == 'default':
            channel = self.default_channel

        if isinstance(channel, list):
            channel = tuple(channel)  # Make sure it is hashable

        # Handle deprecated arguments  # FIXME: use decorator
        if debug is not None:
            status = substitute_deprecated_arg(debug, status, 'debug', 'status')
        if prefix is not None:
            sample_id = substitute_deprecated_arg(prefix, sample_id, 'prefix', 'sample_id')
        if postfix is not None:
            asset_sub_type = substitute_deprecated_arg(postfix, asset_sub_type, 'postfix', 'asset_sub_type')

        if status:
            self.debug = status

        if asset_sub_type and not suffix:
            asset_type += f'_{asset_sub_type}'
        if channel not in self and isinstance(channel, tuple):
            warnings.warn(f'Channel {channel} not found as tuple, trying string version.')
            channel = ('-'.join(channel)).lower()  # Try string version if tuple version not found

        if channel not in self:
            if default == 'closest':
                raise MissingAssetError(f'Unknown channel "{channel}". Available channels: {list(self.channels)}')
            else:
                channel = self.default_channel

        if asset_type in self[channel]:
            asset = self[channel][asset_type]
        else: # asset is somehow None
            if default == 'closest':
                warnings.warn('No exact match found. Using partial matching from start.')
                asset = self.get_closest_matching_asset(asset_type, channel)
            else:
                return default
        if sample_id or extension or version:  # FIXME: subdirectory
            if suffix:
                asset = asset.variant(sample_id, asset_sub_type, extension, version, sub_type=suffix)
            else:
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
        if '*' in asset_type or '?' in asset_type:  # Regex style matching
            import fnmatch
            matching_types = [k for k in self[channel].keys() if fnmatch.fnmatch(k, asset_type)]
        else:  # Prefix style matching
            matching_types = [k for k in self[channel].keys() if k.startswith(asset_type)]

        if len(matching_types) == 0:
            raise MissingAssetError(f'No asset of type "{asset_type}" found in workspace.')
        elif len(matching_types) == 1:
            asset = self[channel][matching_types[0]]
        else:
            raise ClearMapWorkspaceError(f'Multiple assets of type {asset_type} found in workspace '
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
        warnings.warn(f'The load method is deprecated. Use Workspace2.from_yaml or Workspace2.from_dict instead.')
        d = np.load(file_path)[0]
        self.__dict__.update(d)

    def save(self, file_path):
        """Saves the workspace configuration to disk"""
        # prevent np to add .npy to a .workspace file
        warnings.warn(f'The save method is deprecated. Use Workspace2.to_yaml or Workspace2.to_dict instead.')
        with open(file_path, "wb") as fid:
            np.save(fid, [self.__dict__])

    # def add_type_specs(self, *args, **kwargs):

    def info(self):
        """
        Pretty print the workspace information
        """
        out = f'{self}\n'
        out += 'directories:\n'

        ok_symbol, n_ok_symbol = get_ok_n_ok_symbols()

        len_dirtype = max([len(k) for k in self.resource_type_to_folder.keys()])
        for resource_type, folder in self.resource_type_to_folder.items():
            out += f'  [{resource_type : >{len_dirtype}}]: {folder}\n'

        out += 'assets:\n'

        len_f_type = max([len(k) for k in self.asset_types.keys()])
        header = f'  [{{:{len_dirtype}}}] {{:{len_f_type}}}'

        for channel, assets_collection in self.items():
            out += f'  Channel: {channel}\n'
            for asset_type, asset in assets_collection.items():
                asset.header = header
                symbol = ok_symbol if asset.exists else n_ok_symbol
                out += f'    {symbol} {asset_type} ({asset.type_spec.resource_type}): '
                try:
                    if asset.exists:
                        out += f'{asset}\n'
                    else:
                        out += f'{asset.base_name} (not found)\n'
                except ClearMapAssetError as e:
                    out += f'{asset.type_spec.basename}\n'   # FIXME:

        return out


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
