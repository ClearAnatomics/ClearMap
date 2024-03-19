"""
This module is meant to replace the workspace module in ClearMap2.
The main difference is the support for an unlimited number of channels.
This is achieved by having a channel centered approach instead of a processing step centered approach.

The file types are now context dependant. There is no more "raw" or "stitched" files but the user defines
the channel prefix and the type (step) of the file so that there can be multiple files of the same type
depending on the channel.

The "file_type_to_name" dictionaries have thus been extensively modified to reflect this change.

The workspace is now based on a collection of Asset objects that handle the data themselves.
The workspace is now also fully stateful.

The "debug" mode is now a context manager that supports different contexts and is not restricted to debug mode.
"""
import warnings

import numpy as np

from ClearMap.IO.workspace_asset import Asset, ChannelAsset


class WorkspaceStateManager:
    def __init__(self, workspace):
        self.workspace = workspace  # TODO: more generic name (e.g. tracked_object or something)

    def __call__(self, context_string=None):
        if context_string is not None:
            self.workspace.context_keyword = context_string
        return self

    def __enter__(self):
        if self.workspace.context_keyword is None:
            self.workspace.context_keyword = True
        else:
            self.workspace.debug = self.workspace.context_keyword
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.workspace.context_keyword = False


class Workspace2:  # FIXME: subclass dict
    """
    A workspace is a collection of assets.

    The assets are stored in a dictionary with the following key combination:
(channel_name, asset_type)
    """
    def __init__(self, default_channel='raw', sample_id=None):
        self.default_content_type = 'nuclei'
        self.assets = {'single_channels': {
        },
                       'multi_channels': {},
                       'other': {}}
        self.default_channel = default_channel
        self.current_channel = default_channel
        self.sample_id = sample_id
        self._context_keyword = ''
        self.known_contexts = []
        self.context_mgr = WorkspaceStateManager(self)

    def add_raw_data(self, file_path, channel=None, data_content_type=None, sample_id=None):
        if channel is None:
            channel = self.default_channel
        if data_content_type is None:
            data_content_type = self.default_content_type
        if sample_id is None:
            sample_id = self.sample_id
        asset = ChannelAsset(file_path, channel, data_content_type, sample_id)
        self.assets['channels'][channel] = {'content_type': data_content_type,
                                            'assets': {'raw': asset}}
        # FIXME: intermediate ChannelAssetCollection class
        #   Add method that takes dict of asset_type, asset_sub_type, exts, resource_type
        for asset_type, asset_sub_type, exts, resource_type in CONTENT_TYPE_TO_RESOURCE_TYPES[data_content_type]:
            associated_asset = ChannelAsset(asset_type, asset.base_directory, extensions=exts,
                                            sub_type=asset_sub_type, channel=channel, sample_id=sample_id,
                                            data_content_type=data_content_type, resource_type=resource_type)
            self.assets['channels'][channel]['assets']['resource_type'] = associated_asset

    def add_asset(self, asset):
        if asset.channel is None:
            asset.channel = self.default_channel
        self.assets[(asset.channel, asset.asset_type)] = asset

    @property
    def context_keyword(self):
        return self._context_keyword

    @context_keyword.setter
    def context_keyword(self, value):
        if value not in self.known_contexts:
            self.known_contexts.append(value)
        self._context_keyword = value

    def get(self, asset_type, channel='current', asset_sub_type=None, sample_id=None, extension=None, version=None, prefix=None, postfix=None):
        # TODO: handle debug mode
        if channel == 'current':
            channel = self.current_channel
        elif channel == 'auto':
            channel = self.default_channel

        # Handle deprecated arguments
        if prefix is not None:
            if sample_id is None:
                warnings.warn('The prefix argument is deprecated. Use sample_id instead.',
                              DeprecationWarning, stacklevel=2)
                sample_id = prefix
            else:
                raise ValueError('Cannot use both prefix and sample_id arguments.')
        if postfix is not None:
            if asset_sub_type is None:
                warnings.warn('The postfix argument is deprecated. Use asset_sub_type instead.',
                              DeprecationWarning, stacklevel=2)
                asset_sub_type = postfix
            else:
                raise ValueError('Cannot use both postfix and asset_sub_type arguments.')

        # FIXME: use asset_sub_type, sample_id, extension and version
        asset = self.assets.get((asset_type, channel))  # FIXME: use status
        if asset is None:
            print('No exact match found. Using partial matching from start.')
            asset = self.get_closest_matching_asset(asset_type)
        return asset

    def get_closest_matching_asset(self, asset_type, size_ratio=0.5):  # FIXME: compound keys
        """
        Get the asset that matches the given asset type the closest.
        If one match is found, return it.
        If multiple matches are found, pick the shortest one if and only if
        it is much shorter than the others (as determined by size ratio).


        Parameters
        ----------
        asset_type: str
            The asset type to match
        size_ratio : float
            The ratio of the shortest match to the second shortest match.
            If the ratio is smaller than this, an exception is raised.

        Returns
        -------
        Asset object
        """
        # matching_types = difflib.get_close_matches(asset_type, self.assets.keys())
        matching_types = [k for k in self.assets.keys() if k.startswith(asset_type)]  # FIXME: should be [k[0] for k in self.assets.keys()]
        if len(matching_types) == 0:
            raise ValueError(f'No asset of type {asset_type} found in workspace.')
        elif len(matching_types) == 1:
            asset = self.assets[matching_types[0]]
        else:
            print(f'WARNING: Multiple assets of type {asset_type} found in workspace.')

            lengths = [len(k) for k in matching_types]
            others = [ln for ln in lengths if ln != min(lengths)]
            # if length of one match is much shorter, warn but pick it
            if min(lengths) < size_ratio * min(others):
                warnings.warn('Matches length differ sufficiently. Picking shortest match.')
                asset = self.assets[matching_types[np.argmin(lengths)]]
            else:  # Otherwise, raise error
                raise ValueError(f'Warning: multiple matches found with similar length ({lengths}).'
                                 'Picking first match.')
        return asset

    def file_path(self, asset_type, channel=None, version=None, keywords=None, prefix=None, postfix=None):
        return self.get(asset_type, channel, version, keywords, prefix, postfix).file_path

    def plot(self, asset_type, channel=None, version=None, keywords=None, prefix=None, postfix=None):
        return self.get(asset_type, channel, version, keywords, prefix, postfix).plot()

    def load(self, file_path):
        """Loads the workspace configuration from disk"""
        d = np.load(file_path)[0]
        self.__dict__.update(d)

    def save(self, file_path):
        """Saves the workspace configuration to disk"""
        # prevent np to add .npy to a .workspace file
        with open(file_path, "wb") as fid:
            np.save(fid, [self.__dict__])

    def update(self, *args, **kwargs):
        # FIXME: add check that
        self._file_type_to_name.update(*args, **kwargs)


def test():
    ws = Workspace2(default_channel='auto')
    print(ws.default_channel)
    print(ws.context_status)
    with ws.context_mgr():
        print(ws.context_status)
    with ws.context_mgr(context_string='hello'):
        print(ws.context_status)
    with ws.context_mgr(context_string='world'):
        print(ws.context_status)
    print(ws.context_status)
