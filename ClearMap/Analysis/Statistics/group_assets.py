from pathlib import Path
from dataclasses import dataclass

import numpy as np
import pandas as pd

import ClearMap.IO.IO as clearmap_io
from ClearMap.Analysis.Statistics.group_statistics import LoadedPValueResults


@dataclass(frozen=True)
class PairKey:
    channel: str
    gp1: str
    gp2: str
    suffix: str = ''

    def _sfx(self) -> str:
        return f'_{self.suffix}' if self.suffix else ''

    def _grp_sfx(self, group_name: str) -> str:
        sfx = f'_{self.suffix}' if self.suffix else ''
        return f'{self.channel}_{group_name}{sfx}'


class GroupResultsAssets:
    """
    Single source of truth for paths of group-level result assets.

    Mirrors Workspace2 for single samples — the orchestrator asks this
    object for paths; it never builds paths itself.
    """

    def __init__(self, results_folder: Path):
        self.root = Path(results_folder)

    # ── per-group ─────────────────────────────────────────────────────────

    def avg_density_path(self, channel: str, group_name: str, suffix: str = '') -> Path:
        return self._group_path(channel, 'avg_density', group_name, suffix)

    def sd_density_path(self, channel: str, group_name: str, suffix: str = '') -> Path:
        return self._group_path(channel, 'sd_density', group_name, suffix)

    def stacked_density_path(self, channel: str, group_name: str, suffix: str = '') -> Path:
        return self._group_path(channel, 'stacked_density', group_name, suffix)

    # ── per-pair ──────────────────────────────────────────────────────────

    def p_val_colors_path(self, channel: str, gp1: str, gp2: str, suffix: str = '') -> Path:
        return self._pair_path(channel, 'p_val_colors', gp1, gp2, suffix, ext='.tif')

    def effect_size_path(self, channel: str, gp1: str, gp2: str, suffix: str = '') -> Path:
        return self._pair_path(channel, 'effect_size', gp1, gp2, suffix, ext='.tif')

    def stats_table_path(self, channel: str, gp1: str, gp2: str, suffix: str = '') -> Path:
        return self._pair_path(channel, 'statistics', gp1, gp2, suffix, ext='.csv')

    # ── existence checks ─────────────────────────────────────────────────

    def p_val_assets_exist(self, channel: str, gp1: str, gp2: str, suffix: str = '') -> bool:
        return self.p_val_colors_path(channel, gp1, gp2, suffix).exists()

    # ── loaders ──────────────────────────────────────────────────────────

    def load_p_val_results(self, channel: str, gp1: str, gp2: str,
                           suffix: str = '') -> 'LoadedPValueResults':
        def _opt(p: Path):
            return clearmap_io.read(p) if p.exists() else None

        return LoadedPValueResults(
            gp1_avg=clearmap_io.read(self.avg_density_path(channel, gp1, suffix)),
            gp1_sd=_opt(self.sd_density_path(channel, gp1, suffix)),
            gp2_avg=clearmap_io.read(self.avg_density_path(channel, gp2, suffix)),
            gp2_sd=_opt(self.sd_density_path(channel, gp2, suffix)),
            p_vals=clearmap_io.read(self.p_val_colors_path(channel, gp1, gp2, suffix)),
            effect_size=_opt(self.effect_size_path(channel, gp1, gp2, suffix)),
        )

    # ── writers ──────────────────────────────────────────────────────────

    def write_avg_density(self, array: np.ndarray, channel: str,
                          group_name: str, suffix: str = '') -> None:
        clearmap_io.write(self.avg_density_path(channel, group_name, suffix), array)

    def write_sd_density(self, array: np.ndarray, channel: str,
                         group_name: str, suffix: str = '') -> None:
        clearmap_io.write(self.sd_density_path(channel, group_name, suffix), array)

    def write_p_val_colors(self, array: np.ndarray, channel: str,
                           gp1: str, gp2: str, suffix: str = '') -> None:
        clearmap_io.write(self.p_val_colors_path(channel, gp1, gp2, suffix),
                          array, photometric='rgb', imagej=True)

    def write_effect_size(self, array: np.ndarray, channel: str,
                          gp1: str, gp2: str, suffix: str = '') -> None:
        clearmap_io.write(self.effect_size_path(channel, gp1, gp2, suffix),
                          array, imagej=True)

    def write_stats_table(self, df: pd.DataFrame, channel: str,
                          gp1: str, gp2: str, suffix: str = '') -> None:
        df.to_csv(self.stats_table_path(channel, gp1, gp2, suffix))

    # ── internal ─────────────────────────────────────────────────────────

    def _group_path(self, channel: str, kind: str,
                    group_name: str, suffix: str, ext: str = '.tif') -> Path:
        sfx = f'_{suffix}' if suffix else ''
        return self.root / f'{channel}_{kind}_{group_name}{sfx}{ext}'

    def _pair_path(self, channel: str, kind: str, gp1: str, gp2: str,
                   suffix: str, ext: str = '.tif') -> Path:
        sfx = f'_{suffix}' if suffix else ''
        return self.root / f'{channel}_{kind}_{gp1}_{gp2}{sfx}{ext}'