from pathlib import Path
from typing import Iterable, Tuple, List, Dict, Optional

import mpld3
import pandas as pd
from PyQt5.QtWidgets import QApplication

from ClearMap.IO  import IO as clm_io
from ClearMap.Visualization.Qt import Plot3d as plot_3d
from ClearMap.Visualization.Qt.utils import link_dataviewers_cursors
from ClearMap.Analysis.Statistics.group_statistics import (compare_groups, density_files_are_comparable,
                                                           check_ids_are_unique, PValueAssets, p_val_assets_for_pair,
                                                           make_summary, LoadedPValueResults)

from .generic_orchestrators import GroupOrchestratorBase
from ..Visualization.Qt.Plot3d import PlotPanel, multi_plot_from_panels

Pair = Tuple[str, str]


def load_p_val_results(assets: PValueAssets) -> LoadedPValueResults:

    def _load_or_none(path: Optional[Path]) -> Optional[object]:
        return clm_io.read(path) if path is not None else None

    return LoadedPValueResults(gp1_avg=clm_io.read(assets.gp1_avg), gp1_sd=_load_or_none(assets.gp1_sd),
                               gp2_avg=clm_io.read(assets.gp2_avg), gp2_sd=_load_or_none(assets.gp2_sd),
                               p_vals=clm_io.read(assets.p_vals), effect_size=_load_or_none(assets.effect_size))


class DensityGroupAnalysisOrchestrator(GroupOrchestratorBase):
    """Group-level statistics & plots for density maps."""

    def compute_p_values(self, comparisons: List[Pair], *, channels: Iterable[str],
                         advanced: bool, density_files_suffix: str) -> None:
        if not self.groups:
            raise ValueError('No groups defined')
        Path(self.results_folder).mkdir(parents=True, exist_ok=True)

        for gp1_name, gp2_name in comparisons:
            gp1_paths = self.groups[gp1_name]
            gp2_paths = self.groups[gp2_name]

            # sanity checks
            for ch in channels:
                _ = density_files_are_comparable(
                    self.results_folder, gp1_dirs=gp1_paths, gp2_dirs=gp2_paths,
                    channel=ch, density_files_suffix=density_files_suffix)
            check_ids_are_unique(gp1_paths, gp2_paths)

            # compute (threaded if wrapper provided)
            self._threaded(compare_groups, self.results_folder,
                           gp1_name, gp2_name, gp1_paths, gp2_paths,
                           advanced=advanced, density_files_suffix=density_files_suffix)
            self._increment_progress_main()

    def find_analysable_channels(self, *, density_suffix: str) -> List[str]:
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
        for ch in sample_mgr.channels:
            asset = sample_mgr.get('density', channel=ch, suffix=density_suffix, default=None)
            if asset is not None and asset.exists:
                channels.append(ch)
        return channels

    def _get_annotator(self, sample_dir: Path, channel: str):
        reg = self.get_worker_for_sample(sample_dir, 'registration', channel=None)
        reg.setup_if_needed()
        return reg.annotators[channel]

    # ---------- plots ----------

    def plot_p_value_maps(self, comparisons: List[Pair], *, channel: str, suffix: str, parent=None):
        results_folder = Path(self.results_folder)

        p_val_imgs = []
        for gp1, gp2 in comparisons:
            if suffix:
                p_path = results_folder / f'{channel}_p_val_colors_{gp1}_{gp2}_{suffix}.tif'
            else:
                p_path = results_folder / f'{channel}_p_val_colors_{gp1}_{gp2}.tif'
            p_val_imgs.append(clm_io.read(p_path))

        if len(comparisons) > 1:
            titles = [f'{gp1} vs {gp2} p values' for gp1, gp2 in comparisons]
            dvs = plot_3d.plot(p_val_imgs, title=titles, arrange=False, sync=True, parent=parent)
        else:  # If only one comparison, show more details (avg, sd, effect size, atlas)
            gp1, gp2 = comparisons[0]

            assets = p_val_assets_for_pair(self.results_folder, channel, gp1, gp2, suffix)
            res = load_p_val_results(assets)
            sample_dir = self._any_sample_in(gp1)
            annotator = self._get_annotator(sample_dir, channel=channel)
            colored_atlas = annotator.create_color_annotation()

            stats_title = f'P values {"and effect size" if res.has_effect else ""}'
            stats_lut = [None, 'flame'] if res.has_effect else None

            panels = [
                PlotPanel(images=res.gp1_imgs, title=gp1, lut='flame'),
                PlotPanel(images=res.gp2_imgs, title=gp2, lut='flame'),
                PlotPanel(images=res.stats_imgs, title=stats_title, lut=stats_lut),
                PlotPanel(images=colored_atlas, title='colored_atlas', lut=None, min_max=(0, 255)),
            ]
            dvs = multi_plot_from_panels(panels, arrange=False, sync=True, parent=parent)

            names_map = annotator.get_names_map()
            for dv in dvs:
                dv.atlas = annotator.atlas
                dv.structure_names = names_map

        link_dataviewers_cursors(dvs)
        return dvs

    def plot_density_maps(self, group_folders: List[str], *, channel: str, density_suffix: str, parent=None):
        paths, titles = [], []
        for folder in group_folders:
            sample_mgr = self.get_sample_manager_for(folder)
            paths.append(sample_mgr.get_path('density', channel=channel, suffix=density_suffix))
            titles.append(sample_mgr.config['sample_id'])
        dvs = plot_3d.plot(paths, title=titles, arrange=False, sync=True, lut=['flame']*len(paths), parent=parent)
        link_dataviewers_cursors(dvs)
        return dvs

    def run_plots(self, plot_function, comparisons: List[Pair], plot_kw_args: Dict):
        app = QApplication.instance()
        if app is not None and app.applicationName() == 'ClearMap':
            from PyQt5.QtWebEngineWidgets import QWebEngineView

        dvs = []
        for gp1_name, gp2_name in comparisons:
            if plot_kw_args.get('group_names') is None:
                kwargs = dict(plot_kw_args, group_names=(gp1_name, gp2_name))
            else:
                kwargs = plot_kw_args
            df = pd.read_csv(self.results_folder / f'statistics_{gp1_name}_{gp2_name}.csv')
            fig = plot_function(df, **kwargs)

            web_view = QWebEngineView()
            web_view.setHtml(mpld3.fig_to_html(fig))
            dvs.append(web_view)
        return dvs

    def compute_stats_tables(self, comparisons: List[Pair], *, save: bool = True):
        """
        Moves the heavy lifting out of the tab:
          - runs make_summary per pair (threaded if a wrapper is set)
          - returns { (gp1, gp2): { channel: DataFrame } }
        """
        out =  {}
        for gp1, gp2 in comparisons:
            dfs = self._threaded(
                make_summary,
                self.results_folder,
                gp1, gp2,
                self.groups[gp1], self.groups[gp2],
                output_path=None,
                save=save,
            )
            out[(gp1, gp2)] = dfs
            self._increment_progress_main()
        return out