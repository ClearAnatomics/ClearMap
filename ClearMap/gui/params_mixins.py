"""
Mixins for params classes
The `OrthoviewerSlicingMixin` provides orthoviewer-related functionality.
"""
from typing import Protocol, runtime_checkable, Literal, Any, Optional, Mapping

import numpy as np

from ClearMap.IO.assets_constants import CONTENT_TYPE_TO_PIPELINE


@runtime_checkable
class HasSlicingFields(Protocol):
    """Implemented by params classes that expose crop_* fields."""
    crop_x_min: int
    crop_x_max: int
    crop_y_min: int
    crop_y_max: int
    crop_z_min: int
    crop_z_max: int


class OrthoviewerSlicingMixin:
    """
    Provides:
      - ratios -> np.ndarray([rx, ry, rz])
      - scale_axis(val, axis='x') / reverse_scale_axis(val, axis='x')
      - slice_tuples / slicing (if HasSlicingFields)

    Defaults to ratios = resampled_resolution / raw_resolution.

    Host must provide `_get_view()`. If the host has `name`, that channel
    is used. If not, set `pipeline = 'TubeMap'|'CellMap'|...` OR
    set `reference_channel` explicitly. The first channel matching the pipeline
    (via CONTENT_TYPE_TO_PIPELINE) is used as fallback.

    Tunables:
      - _target_space: 'resampled' | 'native' (default 'resampled')
      - set_target_space('resampled'|'native')
    """
    pipeline: Optional[str] = None
    reference_channel: Optional[str] = None
    _target_space: Literal['resampled', 'native'] = 'resampled'

    def set_target_space(self, space: Literal['resampled', 'native']):
        if space not in ('resampled', 'native'):
            raise ValueError("space must be 'resampled' or 'native'")
        self._target_space = space

    def _sample_channels_cfg(self) -> Mapping[str, Any]:
        view = self._get_view()
        return view['sample'].get('channels', {})

    def _raw_resolution(self, channel_name: str) -> np.ndarray:
        res = self._sample_channels_cfg().get(channel_name, {}).get('resolution', [1.0, 1.0, 1.0])
        return np.array(res, dtype=float)

    def _resampled_res(self, channel_name: str) -> np.ndarray[Any, np.dtype[Any]]:
        view = self._get_view()
        res = view['registration']['channels'][channel_name]['resampled_resolution']
        return np.array(res, dtype=float)

    def _target_resolution(self, channel_name: str) -> np.ndarray:
        if self._target_space == 'resampled':
            return self._resampled_res(channel_name)
        else:
            return self._raw_resolution(channel_name)

    def _pick_reference_channel(self) -> Optional[str]:
        # 1) explicit override
        if isinstance(self.reference_channel, str):
            return self.reference_channel

        # 2) per-channel objects (ChannelUiParameters)
        channel_name = getattr(self, 'name', None)
        if channel_name is not None and channel_name:
            return channel_name

        # 3) pipeline-level objects -> derive from relevant channels
        pipeline_name = getattr(self, 'pipeline', None)
        if pipeline_name is not None:
            for ch, cfg in self._sample_channels_cfg().items():
                dt = cfg.get('data_type')
                if dt and CONTENT_TYPE_TO_PIPELINE.get(dt) == pipeline_name:  # FIXME: use self.relevant_channels
                    return ch

        # 4) last resort
        chans = list(self._sample_channels_cfg().keys())
        return chans[0] if chans else None

    @property
    def ratios(self) -> np.ndarray:
        channel = self._pick_reference_channel()
        raw_res = self._raw_resolution(channel)
        tgt_res = self._target_resolution(channel)
        with np.errstate(divide='ignore', invalid='ignore'):
            r = tgt_res / raw_res
        r[~np.isfinite(r)] = 1.0
        return r

    def scale_axis(self, val: float | int, axis: str = 'x') -> int:
        axis_ratio = float(self.ratios['xyz'.index(axis)])
        scaled = round(val * axis_ratio)
        return int(scaled)

    def reverse_scale_axis(self, val: float | int, axis: str = 'x') -> int:
        axis_ratio = float(self.ratios['xyz'.index(axis)])
        scaled = round(val / axis_ratio) if axis_ratio else val
        return int(scaled)

    @property
    def slice_tuples(self):
        if not isinstance(self, HasSlicingFields):
            raise AttributeError('slice_tuples requires crop_* fields.')
        return ((self.crop_x_min, self.crop_x_max),
                (self.crop_y_min, self.crop_y_max),
                (self.crop_z_min, self.crop_z_max))

    @property
    def slicing(self):
        return tuple(slice(rng_min, rng_max) for rng_min, rng_max in self.slice_tuples)
