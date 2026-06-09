import math
from pathlib import Path
from typing import Iterable, Tuple, List, Dict, Optional, TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from pandas import DataFrame
from scipy import stats


# import mpld3  # WARNING: local import. Present here only for reference
# from PyQt5.QtWidgets import QApplication

from ClearMap.IO  import IO as clm_io
from ClearMap.Analysis.Statistics.group_statistics import (generate_summary_table, group_region_counts,
                                                           remove_p_val_nans, stack_voxelizations, get_colored_p_vals)

from .generic_orchestrators import GroupOrchestratorBase
from .utils import init_sample_manager_and_processors
from ..Alignment.utils import get_all_region_ids
from ..Analysis.Statistics.group_assets import GroupResultsAssets
from ..IO.assets_constants import CHANNELS_ASSETS_TYPES_CONFIG
from ..Utils.exceptions import GroupStatsError
from ..Visualization.Qt.Plot3d import PlotPanel, multi_plot_from_panels
from ..config.atlas import ATLAS_NAMES_MAP
from ..config.compound_keys import CompoundKey

if TYPE_CHECKING:
    from ClearMap.IO.workspace_asset import Asset
    from PyQt5.QtWebEngineWidgets import QWebEngineView

Pair = Tuple[str, str]

FILE_SIZE_TOLERANCE = 1024


def _pipeline_points_asset_map() -> dict[str, str]:
    """
    For each pipeline, return the first table-format asset that is relevant to it.
    Derivation order matches insertion order of CHANNELS_ASSETS_TYPES_CONFIG.

    Examples
    -------
    >>>print(_pipeline_points_asset_map())
    {'CellMap': 'cells',
     'TractMap': 'tract_voxels',
     'Colocalization': 'colocalization',
     'TubeMap': 'vertices'
    }
    """
    result: dict[str, str] = {}
    for asset_type, cfg in CHANNELS_ASSETS_TYPES_CONFIG.items():
        if cfg.get('file_format_category') != 'table':
            continue
        for pipeline in cfg.get('relevant_pipelines', []):
            if pipeline not in result:
                result[pipeline] = asset_type
    return result

_PIPELINE_POINTS_ASSET: dict[str, str] = _pipeline_points_asset_map()


class DensityGroupAnalysisOrchestrator(GroupOrchestratorBase):
    """Group-level statistics & plots for density maps."""

    def __init__(self, *, group_controller, pipeline: str, groups=None, channel=None):
        super().__init__(group_controller=group_controller,
                         groups=groups, channel=channel)
        self.pipeline = pipeline
        self._assets: Optional[GroupResultsAssets] = None

    @property
    def assets(self) -> GroupResultsAssets:
        current = self.results_folder
        if self._assets is None or self._assets.root != current:
            self._assets = GroupResultsAssets(current)
        return self._assets

    # ── sample asset resolution ─────────

    def _points_df(self, sample_dir: Path, channel) -> pd.DataFrame:
        sm = self.get_sample_manager_for(sample_dir)

        if CompoundKey.is_valid_key_str(channel):  #  If we deal with CompoundKey -> split to components
            channel = CompoundKey.from_string(channel).as_tuple()

        asset = sm.get(self._points_asset_type(), channel=channel)
        if not asset.exists:
            raise FileNotFoundError(f'No {self._points_asset_type()} for {channel=} in {sample_dir}')
        df = pd.read_feather(asset.path)
        self._ensure_registered(df, channel)
        return df

    def _ensure_registered(self, df: DataFrame, channel: tuple[str, ...] | tuple[str, str] | Any):
        if 'id' not in df.columns:
            raise ValueError(
                f'{self._points_asset_type()} for {channel=} has no "id" column. '
                f'Ensure registration and annotation were run.')

    def _density_asset(self, sample_dir: Path, channel: str, suffix: str) -> 'Asset':
        sm = self.get_sample_manager_for(sample_dir)
        asset = sm.get('density', channel=channel, asset_sub_type=suffix or None)
        return asset

    def _density_array(self, sample_dir: Path, channel: str, suffix: str) -> np.ndarray:
        asset = self._density_asset(sample_dir, channel, suffix)
        if not asset.exists:
            raise FileNotFoundError(f'No density for {channel=}, {suffix=} in {sample_dir}')
        return clm_io.read(asset.path)

    def _points_asset_type(self) -> str:
        asset = _PIPELINE_POINTS_ASSET.get(self.pipeline)
        if asset is None:
            raise ValueError(f'No points asset defined for {self.pipeline=}')
        return asset

    def _cells_df(self, sample_dir: Path, channel: str) -> pd.DataFrame:
        sm = self.get_sample_manager_for(sample_dir)
        asset = sm.get('cells', channel=channel)
        if not asset.exists:
            raise FileNotFoundError(f'No cells for {channel=} in {sample_dir}')
        return pd.read_feather(asset.path)

    def _sample_id(self, sample_dir: Path) -> str:
        sm = self.get_sample_manager_for(sample_dir)
        return sm.config['sample_id']

    # ── validation ───────────────────────────────────────────────────────

    def density_files_are_comparable(self, gp1_dirs, gp2_dirs,
                                     channel: str, suffix: str = '') -> bool:
        """Raise GroupStatsError if density file sizes differ across groups."""
        sizes = []
        for d in list(gp1_dirs) + list(gp2_dirs):
            asset = self._density_asset(d, channel=channel, suffix=suffix or None)
            sizes.append(asset.path.stat().st_size)

        def format_sizes(paths_list):
            return '\t\n'.join([f'{f_p}: {f_p.stat().st_size:,}' for f_p in paths_list])

        def _resolve_density_path(folder):
            asset = self._density_asset(d, channel=channel, suffix=suffix)
            return asset.path

        if not all(math.isclose(s, sizes[0], abs_tol=FILE_SIZE_TOLERANCE) for s in sizes):
            gp1_f_list = [_resolve_density_path(d) for d in gp1_dirs]
            gp2_f_list = [_resolve_density_path(d) for d in gp2_dirs]
            raise GroupStatsError(f'Density file sizes differ for {channel=}, sizes differ\n'
                                  f'Group 1: \t\n{format_sizes(gp1_f_list)}\n'
                                  f'Group 2: \t\n{format_sizes(gp2_f_list)}\n')
        return True

    def check_ids_are_unique(self, gp1_dirs, gp2_dirs) -> None:
        ids = [self._sample_id(Path(d)) for d in list(gp1_dirs) + list(gp2_dirs)]
        if len(ids) != len(set(ids)):
            raise GroupStatsError(f'Sample IDs are not unique: {ids=}')

    # ── computation ──────────────────────────────────────────────────────

    def compute_p_values(self, comparisons: List[Pair], *, channels: Iterable[str],
                         advanced: bool, density_files_suffix: str) -> None:
        if not self.groups:
            raise ValueError('No groups defined')
        self.results_folder.mkdir(parents=True, exist_ok=True)

        for gp1_name, gp2_name in comparisons:
            gp1_dirs = self.groups[gp1_name]
            gp2_dirs = self.groups[gp2_name]

            # sanity checks
            for ch in channels:
                self.density_files_are_comparable(gp1_dirs, gp2_dirs, ch, density_files_suffix)
            self.check_ids_are_unique(gp1_dirs, gp2_dirs)

            # compute (threaded if wrapper provided)
            self._threaded(self._compute_pair, gp1_name, gp2_name,
                           channels=channels, advanced=advanced,
                           suffix=density_files_suffix)
            self._increment_progress_main()

    # REFACTOR: more explicit name
    def _compute_pair(self, gp1_name: str, gp2_name: str, *,
                      channels: list[str], advanced: bool, suffix: str) -> None:

        for channel in channels:
            # ── load ─────────────────────────────────────────────────────────
            gp1_arrays = [self._density_array(Path(d), channel, suffix)
                          for d in self.groups[gp1_name]]
            gp2_arrays = [self._density_array(Path(d), channel, suffix)
                          for d in self.groups[gp2_name]]

            # ── pure math (group_statistics) ─────────────────────────────────
            gp1_stacked = stack_voxelizations(gp1_arrays)  # (X, Y, Z, N)
            gp2_stacked = stack_voxelizations(gp2_arrays)

            t_vals, p_vals = stats.ttest_ind(gp1_stacked, gp2_stacked,
                                             axis=3, equal_var=False)
            p_vals, t_vals = remove_p_val_nans(p_vals, t_vals)

            colored = np.maximum(
                get_colored_p_vals(p_vals, t_vals, 0.05, ('red', 'green')),
                get_colored_p_vals(p_vals, t_vals, 0.01, ('green', 'blue'))
            ).astype(np.uint8)

            if advanced:
                effect_size = np.abs(gp1_stacked.mean(axis=3).astype(int) - gp2_stacked.mean(axis=3).astype(int))
                effect_size = effect_size.astype(np.uint16)   # for imagej compatibility

            # ── write ────────
            self.assets.write_avg_density(gp1_stacked.mean(axis=3), channel, gp1_name, suffix)
            self.assets.write_avg_density(gp2_stacked.mean(axis=3), channel, gp2_name, suffix)
            self.assets.write_p_val_colors(colored, channel, gp1_name, gp2_name, suffix)

            if advanced:
                self.assets.write_sd_density(gp1_stacked.std(axis=3), channel, gp1_name, suffix)
                self.assets.write_sd_density(gp2_stacked.std(axis=3), channel, gp2_name, suffix)
                self.assets.write_effect_size(effect_size, channel, gp1_name, gp2_name, suffix)
        # FIXME: return dict {channel_name: colored_p_vals}

    def find_available_density_suffixes(self) -> List[str]:
        """Scan the first sample's workspace to find available density suffixes."""
        if not self.groups:
            return []
        first_group = next(iter(self.groups.values()), None)

        sm = self.get_sample_manager_for(Path(first_group[0]))

        # Get target channel(s) depending on the pipeline
        channels = sm.get_channels_by_pipeline(self.pipeline, as_list=True)
        if not channels:
            return []

        if self.pipeline == 'TubeMap':
            target_ch = tuple(channels)  # Compund channel for the graph
        else:
            target_ch = channels[0]  # Just use the first one to probe

        suffixes = set()

        # Probe the workspace info dictionary for density files matching our channel
        for asset_key in sm.workspace.info_dict().keys():
            # asset_key is a tuple: (type, channel, subtype)
            if len(asset_key) >= 3 and asset_key[0] == 'density' and asset_key[1] == target_ch:
                suffixes.add(asset_key[2] or '')  # subtype is the suffix

        # Fallback if scanning fails/is empty but we know defaults
        if not suffixes:
            if self.pipeline == 'TubeMap':
                suffixes = {'branches'}
            else:
                suffixes = {'counts'}

        return sorted(list(suffixes))

    def find_analysable_channels(self, *, density_suffix: str | None) -> List[str]:
        """
        Inspects the first sample of the first group to infer which channels have density maps.
        """
        if not self.groups:
            return []
        first_group = next(iter(self.groups.values()))
        if not first_group:
            return []
        first_folder = Path(first_group[0])
        sample_mgr = self.get_sample_manager_for(sample_src_dir=first_folder)

        channels = []
        for ch in sample_mgr.pipeline_ready_channels:  # individual channels (CellMap, TractMap, Colocalization)
            asset = sample_mgr.get('density', channel=ch, suffix=density_suffix, default=None)
            if asset is not None and asset.exists:
                channels.append(ch)

        vasc_channels = sample_mgr.get_channels_by_pipeline('TubeMap', as_list=True)
        if vasc_channels:
            compound = tuple(vasc_channels)
            asset = sample_mgr.get('density', channel=compound, asset_sub_type=density_suffix or None, default=None)
            if asset is not None and asset.exists:
                    channels.append(compound)

        return channels

    def _get_annotator(self, sample_dir: Path, channel: str):
        reg = self.get_worker_for_sample(sample_dir, 'registration', channel=None)
        reg.setup_if_needed()
        return reg.annotators[channel]

    def run_plots(self, plot_function, comparisons: List[Pair], *,
                  channel: str, plot_kw_args: Dict) -> list['QWebEngineView']:
        from PyQt5.QtWidgets import QApplication
        app = QApplication.instance()
        if app is not None and app.applicationName() == 'ClearMap':
            from PyQt5.QtWebEngineWidgets import QWebEngineView

        import mpld3
        dvs = []
        for gp1_name, gp2_name in comparisons:
            if plot_kw_args.get('group_names') is None:
                kwargs = dict(plot_kw_args, group_names=(gp1_name, gp2_name))
            else:
                kwargs = plot_kw_args
            df = pd.read_csv(self.assets.stats_table_path(channel, gp1_name, gp2_name))
            fig = plot_function(df, **kwargs)

            web_view = QWebEngineView()
            web_view.setHtml(mpld3.fig_to_html(fig))
            dvs.append(web_view)
        return dvs

    def compute_stats_tables(self, comparisons, *, channel: str | None = None, density_suffix: str | None = None, save=True):
        # FIXME: add save
        """
        Moves the heavy lifting out of the tab:
          - runs make_summary per pair (threaded if a wrapper is set)
          - returns { (gp1, gp2): { channel: DataFrame } }
        """
        out = {}
        for gp1_name, gp2_name in comparisons:
            dfs = self._threaded(
                self._compute_summary, gp1_name, gp2_name,
                channel=channel, density_suffix=density_suffix, save=save)
            out[(gp1_name, gp2_name)] = dfs
            self._increment_progress_main()
        return out

    def _compute_summary(self, gp1_name: str, gp2_name: str, *,
                         channel: str | None, density_suffix: str | None = None, save: bool) -> dict[str, pd.DataFrame]:
        ref_dir = Path(self.groups[gp1_name][0])
        orchestrators = init_sample_manager_and_processors(folder=ref_dir)
        aligner = orchestrators['registration_processor']

        sm = self.get_sample_manager_for(ref_dir)
        channels = [channel] if channel else self.find_analysable_channels(density_suffix=density_suffix)

        result = {}
        for ch in channels:
            # FIXME: replace by _get_annotator
            # FIXME: reinstate annotator_factory
            #  annotator_factory : callable(folder, channel) -> (annotator, atlas_scale), optional
            #  If provided, used to obtain an annotator and atlas scale for a sample.
            #  Defaults to building full processors via ``init_sample_manager_and_processors``.

            annotator = aligner.annotators[ch or sm.alignment_reference_channel]
            atlas_id = aligner.config['atlas']['id']
            atlas_scale = [ATLAS_NAMES_MAP[atlas_id]['resolution']] * 3
            volume_map = annotator.get_lateralised_volume_map(atlas_scale)

            gp1_dfs = [self._points_df(Path(d), ch) for d in self.groups[gp1_name]]
            gp2_dfs = [self._points_df(Path(d), ch) for d in self.groups[gp2_name]]
            region_ids = get_all_region_ids(gp1_dfs + gp2_dfs)

            aggregated = {
                gp1_name: group_region_counts(
                    annotator, region_ids, gp1_dfs,
                    [self._sample_id(Path(d)) for d in self.groups[gp1_name]],
                    volume_map),
                gp2_name: group_region_counts(
                    annotator, region_ids, gp2_dfs,
                    [self._sample_id(Path(d)) for d in self.groups[gp2_name]],
                    volume_map),
            }
            total_df = generate_summary_table(aggregated)

            if save:
                self.assets.write_stats_table(total_df, ch, gp1_name, gp2_name)
            result[ch] = total_df
        return result

    # ---------- plots ----------
    def plot_p_value_maps(self, comparisons: List[Pair], *, channel: str, suffix: str, advanced: bool = False, parent=None):
        from ClearMap.Visualization.Qt import Plot3d as plot_3d
        from ClearMap.Visualization.Qt.utils import link_dataviewers_cursors

        if len(comparisons) > 1:  # Multiple comparisons: just show p-value maps
            p_val_imgs = []
            for gp1, gp2 in comparisons:
                p_path = self.assets.p_val_colors_path(channel, gp1, gp2, suffix)
                p_val_imgs.append(clm_io.read(p_path))

            titles = [f'{gp1} vs {gp2} p values' for gp1, gp2 in comparisons]
            dvs = plot_3d.plot(p_val_imgs, title=titles, arrange=False, sync=True, parent=parent)
        else:  # If only one comparison, show more details (avg, sd, effect size, atlas (+ sd and effect_size if advanced)
            gp1, gp2 = comparisons[0]

            res = self.assets.load_p_val_results(channel, gp1, gp2, suffix)

            sample_dir = self._any_sample_in(gp1)
            annotator = self._get_annotator(sample_dir, channel=channel)
            colored_atlas = annotator.create_color_annotation()

            panels = [
                PlotPanel(images=res.gp1_avg, title=f'{gp1} avg', lut='flame'),
                PlotPanel(images=res.gp2_avg, title=f'{gp2} avg', lut='flame'),
                PlotPanel(images=res.p_vals, title='P values', lut=None),
            ]

            if advanced:
                if res.gp1_sd is not None:
                    panels.append(PlotPanel(images=res.gp1_sd, title=f'{gp1} SD', lut='viridis'))
                if res.gp2_sd is not None:
                    panels.append(PlotPanel(images=res.gp2_sd, title=f'{gp2} SD', lut='viridis'))
                if res.effect_size is not None:
                    panels.append(PlotPanel(images=res.effect_size, title='Effect size', lut='flame'))

            panels.append(
                PlotPanel(images=colored_atlas, title='Atlas', lut=None, min_max=(0, 255))
            )

            dvs = multi_plot_from_panels(panels, arrange=False, sync=True, parent=parent)

            names_map = annotator.get_names_map()
            for dv in dvs:
                dv.atlas = annotator.atlas
                dv.structure_names = names_map

        link_dataviewers_cursors(dvs)
        return dvs

    def plot_density_maps(self, group_folders: List[str], *, channel: str, density_suffix: str, parent=None):
        from ClearMap.Visualization.Qt import Plot3d as plot_3d
        from ClearMap.Visualization.Qt.utils import link_dataviewers_cursors
        paths, titles = [], []
        for folder in group_folders:
            asset = self._density_asset(folder, channel, suffix=density_suffix or None)
            paths.append(asset.path)
            titles.append(self._sample_id(folder))
        dvs = plot_3d.plot(paths, title=titles, arrange=False, sync=True, lut=['flame']*len(paths), parent=parent)
        link_dataviewers_cursors(dvs)
        return dvs