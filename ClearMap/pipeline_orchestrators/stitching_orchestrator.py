import platform
import re
import warnings
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path
from typing import Optional

import numpy as np

from ClearMap.Alignment.Stitching import StitchingRigid as stitching_rigid, StitchingWobbly as stitching_wobbly
from ClearMap.Alignment.Stitching.StitchingWobbly import WobblyLayout
from ClearMap.IO import IO as clearmap_io
from ClearMap.IO.metadata import define_auto_stitching_params, parse_ome_info
from ClearMap.Utils.exceptions import MissingRequirementException
from ClearMap.Utils.tag_expression import Expression
from ClearMap.Utils.utilities import check_stopped
from ClearMap.Visualization.Color.Color import gray_image_to_rgb
from ClearMap.Visualization.Qt import Plot3d as plot_3d
from ClearMap.config.config_coordinator import ConfigCoordinator
from ClearMap.gui.widgets import ProgressWatcher
from ClearMap.pipeline_orchestrators.generic_orchestrators import PipelineOrchestrator
from ClearMap.pipeline_orchestrators.sample_info_management import SampleManager


class StitchingProcessor(PipelineOrchestrator):
    """
    This class is used to manage the stitching process
    Handle image stitching operations.
    Manage stitching configurations and processes.
    """
    config_name = 'stitching'
    def __init__(self, sample_manager: SampleManager, cfg_coordinator: ConfigCoordinator):
        super().__init__(cfg_coordinator)
        self.sample_manager: SampleManager = sample_manager
        self.progress_watcher: Optional[ProgressWatcher] = None
        self.__wobbly_stitching_place_re = 'done constructing constraints for component'
        self.__wobbly_stitching_align_lyt_re = ('Alignment: Wobbly alignment',
                                                re.compile(r"Alignment:\sWobbly alignment \(\d+, \d+\)->\(\d+, \d+\) "
                                                           r"along axis [0-3] done: elapsed time: \d+:\d{2}:\d{2}.\d+"))
        self.__wobbly_stitching_stitch_re = ('Stitching: stitching',
                                             re.compile(r'Stitching: stitching wobbly slice \d+/\d+'))
        self.__rigid_stitching_align_re = ('done',
                                           re.compile(r"Alignment: aligning \(\d+, \d+\) with \(\d+, \d+\), alignment"
                                                      r" pair \d+/\d+ done, shift = \(-?\d+, -?\d+, -?\d+\),"
                                                      r" quality = -\d+\.\d+e\+\d+!"))

    def setup(self, sample_manager: Optional[SampleManager] = None, convert_tiles: bool = False):
        self.sample_manager = sample_manager if sample_manager else self.sample_manager
        if not self.cfg_coordinator.get_config_view('stitching'):
            raise ValueError('Stitching config not set in config coordinator')
        if self.sample_manager.setup_complete:
            self.workspace = self.sample_manager.workspace
            if convert_tiles:
                self.convert_tiles()  # TODO: check if needed
            self.setup_complete = True
        else:
            self.setup_complete = False
            warnings.warn('Sample manager not set up yet, stitching processor setup incomplete', stacklevel=2)

    @check_stopped
    def convert_tiles(self, _force=False):
        """Convert list of input files to numpy files for efficiency reasons"""
        for channel in self.sample_manager.stitchable_channels:
            self.convert_tiles_channel(channel, _force=_force)
        self.update_watcher_main_progress()

    def convert_tiles_channel(self, channel, _force=False):
        if self.config['channels'][channel].get('use_npy', False) or _force:
            asset = self.get('raw', channel=channel, prefix=self.sample_manager.prefix)
            n_procs = self.config['performance']['file_conversion']['n_processes']
            file_list = asset.file_list
            if not file_list or Path(file_list[0]).suffix == '.tif':
                try:
                    asset.convert('.npy', processes=n_procs, workspace=self.workspace, verbose=self.verbose)
                except BrokenProcessPool:
                    print(f'File conversion of {channel} to numpy canceled')
                    return

    def get_stitching_order(self, strict=False):
        """
        Returns a list of trees (each tree is a list of channels in hierarchical order).
        Raises a ValueError if any channel is unreachable or if there is a cycle.
        """
        config = self.config
        channels = set(config['channels'].keys())  # All channels
        root_channels = [ch for ch, cfg in config['channels'].items()
                         if cfg.get('layout_channel', ch) == ch]
        visited = set()
        forest = {}

        def traverse_tree(root):
            stack = [root]
            tree = []
            while stack:
                node = stack.pop()
                tree.append(node)
                visited.add(node)
                # Find immediate children whose layout_channel == node
                for ch, c_cfg in config['channels'].items():
                    if ch not in visited and c_cfg.get('layout_channel', ch) == node:
                        stack.append(ch)
            return tree

        def tree_is_stitchable(root_channel, strict):
            cfg = self.config['channels'].get(root_channel)
            has_layout = self.get('layout', channel=root_channel, asset_sub_type='placed').exists
            if not (cfg['run'] or has_layout):
                msg = (f"Cannot stitch tree with root `{root_channel}`:"
                       f" {root_channel} does not have a layout and is not set for stitching.")
                if strict:
                    raise ValueError(msg)
                else:
                    warnings.warn(msg, stacklevel=2)
                return False
            return True

        for root in root_channels:
            if not tree_is_stitchable(root, strict):
                continue
            if root not in visited:
                forest[root] = traverse_tree(root)

        unassigned = channels - visited
        if unassigned:
            raise ValueError(f"Unreachable or extra channels found: {unassigned}")
        return forest

    def _pick_overlap_px(self, stitching: dict) -> tuple[int, int] | None:
        """Prefer declared overlap (px); else derived; return ints if both axes present."""
        ov = (stitching or {}).get('overlap_px') or {}
        dec = ov.get('declared') or (None, None)
        der = ov.get('derived') or (None, None)
        ox = dec[0] if dec[0] is not None else der[0]
        oy = dec[1] if dec[1] is not None else der[1]
        if ox is None or oy is None:
            return None
        return int(round(ox)), int(round(oy))

    def create_layout_from_ome(self, channel: str) -> None:
        """
        Build and persist a real WobblyLayout for 'channel' from OME TileConfiguration.
        Marks the channel as 'use_existing_layout' so rigid placement can be skipped.
        """
        raw_asset = self.get('raw', channel=channel, prefix=self.sample_manager.prefix, extension='.tif')
        if not raw_asset.file_list:
            raise FileNotFoundError(f"No tiles found for channel '{channel}'")

        first_tif = Path(raw_asset.file_list[0])
        ome_info = parse_ome_info(first_tif)

        # Map this channel to OME TileConfiguration channel index
        ch_idx = self.sample_manager.infer_channel_index_from_name(first_tif)
        tile_cfg = ome_info.get('tile_configuration') or {}
        if ch_idx not in tile_cfg:
            raise ValueError(f"Cannot map channel '{channel}' to OME TileConfiguration channel index; "
                             f"available: {sorted(tile_cfg.keys())}")

        mapping = tile_cfg[ch_idx]  # {(ix, iy): {'filename', 'relative_x', 'relative_y', ...}}
        if not mapping:
            raise ValueError(f"Empty TileConfiguration for channel index {ch_idx}")

        # Grid size from indices (unzip)
        keys_sorted = sorted(mapping.keys())
        xs, ys = zip(*keys_sorted)  # safe because mapping is not empty
        nx, ny = max(xs) + 1, max(ys) + 1

        # Basename -> full path (all tiles for this channel)
        basename_to_full = {Path(p).name: Path(p) for p in raw_asset.file_list}

        # Build sources/tile_positions/positions in row-major (ix,iy) order
        sources = []
        tile_positions = []
        positions = []  # (x, y, z=0)
        for (ix, iy) in keys_sorted:
            rec = mapping[(ix, iy)]
            base = rec['filename']
            sources.append(str(basename_to_full[base]))
            tile_positions.append((ix, iy))
            positions.append((float(rec['relative_x']), float(rec['relative_y']), 0.0))  # z=0 for 2D tiling

        # Overlap in pixels (int,int) or None
        overlaps_px = self._pick_overlap_px(ome_info.get('stitching'))

        # Real WobblyLayout
        lyt = WobblyLayout(sources=sources, tile_shape=(nx, ny), tile_positions=tile_positions,
                           positions=positions, overlaps=overlaps_px, axis=2)  # Axis = which axis "wobbles"
        lyt.lower_to_origin()  # Just in case

        placed_asset_path = self.get_path('layout', channel=channel, asset_sub_type='placed')
        lyt.save(str(placed_asset_path))

        self.cfg_coordinator.submit_patch({'stitching': {'channels': {channel: {'use_existing_layout': True}}}},
                                          sample_manager=self.sample_manager)

    def stack_columns(self, channel):
        asset = self.get('raw', channel=channel, prefix=self.sample_manager.prefix)
        exp = asset.expression

        x_min, x_max = exp.tag_range('X')
        y_min, y_max = exp.tag_range('Y')

        z_expression = '_xyz-Table Z<Z,4>'    # REFACTOR: brittle, find better way
        if exp.n_tags() >= 3:
            for y in range(y_min, y_max + 1):
                for x in range(x_min, x_max + 1):
                    column_expression = exp.string(values={'X': x, 'Y': y})
                    dest = column_expression.replace(z_expression, '')
                    clearmap_io.convert(column_expression, dest)
            # squash Z axis
            asset.expression = exp.string().replace(z_expression, '')  # overwrite expression
            self.sample_manager.set_channel_expression(channel, asset.expression)

    def copy_or_stack(self, channel):
        """
        Copy or stack or convert to npy the channel data in case there is no X/Y tiling

        Parameters
        ----------
        channel : str
            The channel to copy or stack
        """
        clearmap_io.convert(self.get_path('raw', channel=channel),
                            self.get_path('stitched', channel=channel))

    def stitch(self):
        if self.stopped:
            return

        # Get the channels in dependency order (based on layout_channel)
        for stitching_tree in self.get_stitching_order().values():
            for channel in stitching_tree:
                self.stack_columns(channel)  # If x/y/z, stack first
                channel_cfg = self.config['channels'][channel]

                has_layout = self.get('layout', channel=channel, asset_sub_type='placed').exists
                if channel_cfg.get('use_existing_layout'):
                    if not has_layout:
                        raise MissingRequirementException(f'Channel {channel} set to use existing layout,'
                                                          f' but no layout found')
                    self._stitch_layout_wobbly(channel)
                    if self.stopped: return
                    continue

                if channel == channel_cfg['layout_channel']:
                    if not channel_cfg['rigid']['skip']:
                        self.align_channel_rigid(channel)
                    if self.stopped: return
                    if not channel_cfg['wobbly']['skip']:
                        self.stitch_channel_wobbly(channel)
                    else:
                        self.place_layout_rigid(channel)
                        if self.stopped: return
                        self._stitch_layout_wobbly(channel)
                else:
                    self._stitch_layout_wobbly(channel)

                if self.stopped:
                    return

    def channel_was_stitched_rigid(self, channel):
        return self.get('layout', channel=channel, asset_sub_type='aligned_axis').exists

    @property
    def n_rigid_steps_to_run(self):
        cfg = self.config['channels']
        return [not cfg[channel]['rigid']['skip'] for channel in cfg.keys() if 'rigid' in cfg[channel]].count(True)

    # @check_stopped
    # @requires_assets([FilePath('raw')])
    def align_channel_rigid(self, channel, _force=False):
        if not self.sample_manager.check_has_all_tiles(channel):
            if self.sample_manager.use_npy(channel):
                self.convert_tiles_channel(channel)
                if not self.sample_manager.check_has_all_tiles(channel):
                    raise MissingRequirementException(f'Channel {channel} missing tiles')
            else:
                raise MissingRequirementException(f'Channel {channel} missing tiles')
        self.set_watcher_step(f'Stitching {channel} rigid')
        rigid_cfg = self.config['channels'][channel]['rigid']

        raw_asset = self.get('raw', channel=channel)
        if raw_asset.is_expression:
            params_file = raw_asset.file_list[0]
        else:
            params_file = raw_asset.existing_path
        overlaps, projection_thickness = define_auto_stitching_params(params_file, rigid_cfg)
        layout = self.get_wobbly_layout(channel, overlaps)
        if rigid_cfg['background_pixels'] is None:
            background_params = rigid_cfg['background_level']
        else:
            background_params = (rigid_cfg['background_level'],
                                 rigid_cfg['background_pixels'])
        max_shifts = [rigid_cfg[f'max_shifts_{ax}'] for ax in 'xyz']
        n_procs = self.config['performance']['stitching']['n_processes']
        self.prepare_watcher_for_substep(len(layout.alignments), self.__rigid_stitching_align_re, 'Align layout rigid')
        try:
            stitching_rigid.align_layout_rigid_mip(layout, depth=projection_thickness, max_shifts=max_shifts,
                                                   ranges=[None, None, None], background=background_params,
                                                   clip=25000, processes=n_procs, workspace=self.workspace, verbose=True)
        except BrokenProcessPool:
            print('Stitching canceled')
            self.stopped = True  # FIXME: see if appropriate solution
            return  # WARNING: do not run stitching_wobbly if rigid failed
        layout.place(method='optimization', min_quality=-np.inf, lower_to_origin=True, verbose=True)
        self.update_watcher_main_progress()

        stitching_rigid.save_layout(self.get_path('layout', channel=channel, asset_sub_type='aligned_axis'),
                                    layout)

    def place_layout_rigid(self, channel):
        layout = stitching_rigid.load_layout(self.get_path('layout', channel=channel,
                                                          asset_sub_type='aligned_axis'))
        layout.place(method='optimization', min_quality=-np.inf, lower_to_origin=True, verbose=True)

        stitching_rigid.save_layout(self.get_path('layout', channel=channel, asset_sub_type='aligned'),
                                    layout)

    # @requires_assets([FilePath('raw')])  # TODO: optional requires npy + requires that channel is kwarg
    def get_wobbly_layout(self, channel, overlaps=None):
        if overlaps is None:
            rigid_cfg = self.config['channels'][channel]['rigid']
            overlaps, _ = define_auto_stitching_params(
                self.get('raw', channel=channel).file_list[0], rigid_cfg).as_source()
        extension = '.npy' if self.sample_manager.use_npy(channel) else None  # TODO: optional requires
        raw_expr = str(self.get_path('raw', channel=channel, extension=extension))
        tag_names = tuple(sorted(Expression(raw_expr).tag_names()))  # sort alphabetically to ensure consistent order
        # Drop irrelevant axes if e.g. scanning only rows or columns
        overlaps = [overlap for name, overlap in zip(('X', 'Y'), overlaps) if name in tag_names]
        layout = stitching_wobbly.WobblyLayout(expression=raw_expr, tile_axes=tag_names, overlaps=overlaps)
        return layout

    @property
    def n_wobbly_steps_to_run(self):
        out = len(self.sample_manager.stitchable_channels) - 1
        for channel in self.sample_manager.stitchable_channels:
            cfg = self.config['channels'][channel]
            if 'wobbly' in cfg and not cfg['wobbly']['skip']:
                out += 3
        return out

    def __align_layout_wobbly(self, channel, layout):
        wobbly_cfg = self.config['channels'][channel]['wobbly']
        max_shifts = [
            wobbly_cfg['max_shifts_x'],
            wobbly_cfg['max_shifts_y'],
            wobbly_cfg['max_shifts_z'],
        ]
        stack_validation_params = dict(
            method='foreground',
            valid_range=wobbly_cfg["stack_valid_range"],
            size=wobbly_cfg["stack_pixel_size"]
        )
        slice_validation_params = dict(
            method='foreground',
            valid_range=wobbly_cfg["slice_valid_range"],
            size=wobbly_cfg["slice_pixel_size"]
        )

        n_pairs = len(layout.alignments)
        self.prepare_watcher_for_substep(n_pairs, self.__wobbly_stitching_align_lyt_re, 'Align layout wobbly')
        try:
            n_processes = self.config['performance']['stitching']['n_processes']
            stitching_wobbly.align_layout(layout, axis_range=(None, None, 3), max_shifts=max_shifts, axis_mip=None,
                                          stack_validation_params=stack_validation_params,
                                          prepare=dict(method='normalization', clip=None, normalize=True),
                                          slice_validation_params=slice_validation_params,
                                          prepare_slice=None,
                                          find_shifts=dict(method='tracing', cutoff=3 * np.sqrt(2)),
                                          processes=n_processes, workspace=self.workspace, verbose=True)
        except BrokenProcessPool:
            print('Wobbly stitching canceled')
            return
        self.update_watcher_main_progress()

    def __place_layout_wobbly(self, layout):
        self.prepare_watcher_for_substep(len(layout.alignments) // 2,  # WARNING: bad estimation
                                         self.__wobbly_stitching_place_re, 'Place layout wobbly')
        try:
            n_processes = self.config['performance']['stitching']['n_processes']
            if platform.system().lower().startswith('darwin'):  # No parallel on MacOS
                n_processes = 1
            stitching_wobbly.place_layout(layout, min_quality=-np.inf,
                                          method='optimization',
                                          smooth=dict(method='window', window='bartlett', window_length=100,
                                                      binary=None),
                                          smooth_optimized=dict(method='window', window='bartlett',
                                                                window_length=20, binary=10),
                                          fix_isolated=False, lower_to_origin=True,
                                          processes=n_processes, workspace=self.workspace, verbose=True)
        except BrokenProcessPool:
            print('Wobbly stitching canceled')
            return
        self.update_watcher_main_progress()

    def _stitch_layout_wobbly(self, channel):
        layout_channel = self.config['channels'][channel]['layout_channel']
        layout = stitching_rigid.load_layout(self.get_path('layout', channel=layout_channel,
                                                           asset_sub_type='placed'))

        try:
            ref_asset = self.get('raw', channel=self.sample_manager.alignment_reference_channel)
            n_slices = len(ref_asset.file_list)  # TODO: find better proxy
        except KeyError:
            n_slices = clearmap_io.shape(self.get('raw', channel=channel).file_list[0])[0]
        self.prepare_watcher_for_substep(n_slices, self.__wobbly_stitching_stitch_re,
                                         'Stitch layout wobbly', True)
        try:
            self._replace_layout_sources(layout, channel, layout_channel)
            n_processes = self.config['performance']['stitching']['n_processes']
            stitching_wobbly.stitch_layout(layout,
                                           sink=str(self.get_path('stitched', channel=channel)),
                                           method='interpolation',
                                           processes=n_processes, workspace=self.workspace, verbose=True)
        except BrokenProcessPool:
            print('Wobbly stitching canceled')
            return

        self.update_watcher_main_progress()

    def _replace_layout_sources(self, layout, channel, layout_channel):
        if channel == layout_channel:
            return
        # layout_channel_asset = self.get('raw', channel=layout_channel)
        channel_asset = self.get('raw', channel=channel)
        if layout_channel != channel:
            # layout_extension = Path(layout.sources[0].location).suffix  # Use the actual extension that was used
            if self.sample_manager.use_npy(channel):  # FIXME: check if we need to copy layout first
                channel_pattern = channel_asset.with_extension(extension='.npy')
            else:
                channel_pattern = channel_asset.path
            layout.replace_source_location(None, str(channel_pattern), method='infer')

    @check_stopped
    def stitch_channel_wobbly(self, channel, _force=False):  # Warning, will stitch channel on its own
        def get_layout_path(layout_sub_type):
            return self.get_path('layout', channel=channel, asset_sub_type=layout_sub_type)
# FIXME: check if ui_to_cfg is needed
        if not self.channel_was_stitched_rigid(channel):
            raise MissingRequirementException(f'Channel {channel} not stitched rigid yet')
        self.set_watcher_step('Stitching wobbly')
        layout = stitching_rigid.load_layout(get_layout_path('aligned_axis'))
        self.__align_layout_wobbly(channel, layout)
        if self.stopped:
            return
        stitching_rigid.save_layout(get_layout_path('aligned'), layout)

        self.__place_layout_wobbly(layout)
        if self.stopped:
            return
        stitching_rigid.save_layout(get_layout_path('placed'), layout)

        self._stitch_layout_wobbly(channel)
        if self.stopped:
            return

    def plot_stitching_results(self, channels=None, mode='side-by-side', parent=None):
        if channels is None:
            channels = self.sample_manager.stitchable_channels
        paths = []
        titles = []
        for c in channels:
            asset = self.get('stitched', channel=c)
            if asset.exists:
                paths.append(asset.path)
                titles.append(f'{c.title()} stitched')
            else:
                print(f'No stitched file for channel {c}')
        if not paths:
            raise MissingRequirementException('No stitched files found')
        if len(paths) == 1:
            paths = paths[0]
        if mode == 'overlay':
            titles = ' and '.join(titles)
            paths = [paths]
        elif mode == 'side-by-side':
            pass
        else:
            raise ValueError(f'Unknown mode {mode}')

        dvs = plot_3d.plot(paths, title=titles, arrange=False, lut='white', parent=parent)
        return dvs

    def stitch_overlay(self, channel, color=True):
        """
        This creates a *dumb* overlay of the tiles
        i.e. only using the fixed guess overlap

        Parameters
        ----------
        channel
        color

        Returns
        -------
        np.array(dtype=uint8)
            The overlay image
        """
        asset = self.get('raw', channel=channel, sample_id=self.sample_manager.prefix)
        positions = asset.positions
        tile_shape = {k: v for k, v in zip('XYZ', asset.tile_shape)}  # TODO: use asset.tile_grid_shape
        middle_z = int(tile_shape['Z'] / 2)
        overlaps = self._get_overlaps(channel)
        output_shape = self._compute_stitched_shape_from_overlaps(overlaps, positions, tile_shape)
        layers = [np.zeros(output_shape, dtype=int), np.zeros(output_shape, dtype=int)]
        if self.sample_manager.has_npy:
            files = asset.variant(extension='.npy').file_list
        else:
            files = asset.file_list
        for tile_path, pos in zip(files, positions):
            tile = self.__read_tile_middle_plane(tile_path, middle_z)

            starts = {ax: pos[ax] * tile_shape[ax] - pos[ax] * overlaps[ax] for ax in 'XY'}
            ends = {ax: starts[ax] + tile_shape[ax] for ax in starts.keys()}
            layer = layers[(pos['Y'] + pos['X']) % 2]  # Alternate colors
            layer[starts['X']: ends['X'], starts['Y']: ends['Y']] = tile
        if color:
            layers[0] = gray_image_to_rgb(layers[0], 'cyan', pseudo_z_score=True, range_max=255)
            layers[1] = gray_image_to_rgb(layers[1], 'magenta', pseudo_z_score=True, range_max=255)
        output_image = layers[0] + layers[1]
        if color:
            output_image = output_image.clip(0, 255).astype(np.uint8)
        return output_image

    def __read_tile_middle_plane(self, tile_path, middle_z):
        if self.sample_manager.has_npy:  # use memmap
            tile = clearmap_io.buffer(tile_path)[:, :, middle_z]
        else:
            tile = clearmap_io.read(tile_path)[:, :, middle_z]
        return tile

    def _compute_stitched_shape_from_overlaps(self, overlaps, positions, tile_shape):
        mosaic_shape = {ax: max([p[ax] for p in positions]) + 1 for ax in 'XY'}  # +1 because 0 indexing
        output_shape = [tile_shape[ax] * mosaic_shape[ax] - overlaps[ax] * (mosaic_shape[ax] - 1) for ax in 'XY']
        return output_shape

    def _get_overlaps(self, channel):
        layout_channel = self.config['channels'][channel]['layout_channel']
        overlaps = {k: self.config['channels'][layout_channel]['rigid'][f'overlap_{k.lower()}'] for k in 'XY'}
        return overlaps

    def overlay_layout_plane(self, layout):  # REFACTOR: move to e.g. layout class
        """Overlays the sources to check their placement.

        Arguments
        ---------
        layout : Layout class
          The layout with the sources to overlay.

        Returns
        -------
        image : array
          A color image.
        """
        dest_shape = tuple(layout.extent[:-1])
        full_lower = layout.lower
        middle_z = round(layout.sources[0].shape[-1] / 2)

        color_layers = [np.zeros(dest_shape, dtype=int), np.zeros(dest_shape, dtype=int)]
        # construct full image
        for src in layout.sources:
            tile = self.__read_tile_middle_plane(src.location,
                                                 middle_z)  # So as not to load the data into the list for memory efficiency
            is_odd = sum(src.tile_position) % 2
            layer = color_layers[is_odd]  # Alternate colors

            current_slicing = self.__compute_overlay_slicing(full_lower, src)
            layer[current_slicing] = tile

        cyan_image = gray_image_to_rgb(color_layers[0], 'cyan', pseudo_z_score=True, range_max=255)
        magenta_image = gray_image_to_rgb(color_layers[1], 'magenta', pseudo_z_score=True, range_max=255)

        # TODO: normalise
        output_image = np.clip(cyan_image + magenta_image, 0, 255).astype(np.uint8)

        return output_image

    def __compute_overlay_slicing(self, full_lower, src):
        l = src.lower
        u = src.upper
        current_slicing = tuple(slice(ll - fl, uu - fl) for ll, uu, fl in zip(l, u, full_lower))[:2]
        return current_slicing

    def plot_layout(self, channel, asset_sub_type='aligned_axis'):
        valid_sub_types = ("aligned_axis", "aligned", "placed")
        if asset_sub_type not in valid_sub_types:
            raise ValueError(f'Expected on of {valid_sub_types} for asset_sub_type, got "{asset_sub_type}"')
        layout = stitching_rigid.load_layout(self.get_path('layout', channel=channel, asset_sub_type=asset_sub_type))
        overlay = self.overlay_layout_plane(layout)
        return overlay
