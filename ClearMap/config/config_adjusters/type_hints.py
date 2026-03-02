from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Mapping, Any, Dict, Tuple, Callable, Optional, Protocol, List


ConfigView = Mapping[str, Any]
ConfigPatch = Dict[str, Any]
ConfigKeys = tuple[str, ...]
ConfigKeysLike = str | ConfigKeys


class AdjusterScope(Enum):
    EXPERIMENT = 'experiment'  # Individual sample
    GROUP = 'group'

@dataclass(frozen=True, slots=True)
class AdjustmentContext:
    scope: AdjusterScope
    # Experiment context
    sample_manager: Optional[SampleManagerProtocol] = None
    # Group context
    group_base_dir: Optional[Path] = None

    run_label: str = ""    # Optional: useful for tracing / provenance


# REFACTOR: move to exceptions.py
class PatchConflictError(RuntimeError):
    def __init__(self, path: Tuple[str, ...], dst_value: object, src_value: object, reason: str):
        dotted = '.'.join(path) if path else '<root>'
        super().__init__(f'Patch conflict at "{dotted}": {reason}, {dst_value=}, {src_value=}')
        self.path = path
        self.dst_value = dst_value
        self.src_value = src_value
        self.reason = reason


TemplateForKey = Callable[[str], Optional[dict]]
KeyTransform    = Callable[[str], str]  # usually identity


def identity_fn(k):
    return k


class SampleManagerProtocol(Protocol):
    @property
    def channels(self) -> List[str]: ...
    @property
    def stitchable_channels(self) -> List[str]: ...
    @property
    def renamed_channels(self) -> Dict[str, str]: ...
    def get_channels_by_pipeline(self, pipeline: str, as_list: bool = False) -> List[str]: ...
    def data_type(self, channel: str) -> Optional[str]: ...
    @property
    def alignment_reference_channel(self) -> Optional[str]: ...
    @property
    def is_colocalization_compatible(self) -> bool: ...
    @property
    def channels_to_detect(self) -> List[str]: ...
    def colocalization_pairs(self, *, oriented: bool = False) -> List[Tuple[str, str]]: ...
    def colocalization_pair_keys(self, *, oriented: bool = False) -> List[str]: ...
    def compute_required_sections(self): ...


AdjusterFn = Callable[[ConfigView, AdjustmentContext], ConfigPatch]
KeysPath = Tuple[str, ...]
