from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple, Iterable, List, Callable

from ClearMap.config.config_adjusters.type_hints import (AdjusterFn, ConfigKeys, ConfigKeysLike,
                                                         AdjusterScope)


class Phase(str, Enum):
    PRE_VALIDATE  = 'pre-validate'
    POST_VALIDATE = 'post-validate'
    PRE_COMMIT    = 'pre-commit'
    POST_COMMIT   = 'post-commit'


class Step(str, Enum):
    # Creation/materialization steps
    APPLY_RENAMES             = 'apply-renames'         # sample/channel_x → renamed channels
    CREATE_PIPELINE_SECTIONS  = 'create-pipeline-sections'  # pipeline-level blocks

    # Instance owners
    CREATE_CHANNELS_RECONCILE = 'create-channels-reconcile'  # add missing channels

    # Population steps
    ENSURE_STATIC_BLOCKS        = 'ensure-static-blocks'  # fill missing keys with defaults
    POPULATE_DEFAULTS_MATERIALIZE = 'populate-defaults-materialize'
    # FIXME: rename to more explicit step name or use an existing one
    ADJUST                    = 'adjust'            # custom adjustments

    # Normalization steps  (currently unused)
    # NORMALIZE_SHAPES          = 'normalize-shapes'  # scalars↔vectors, sentinels, etc.


class AdjusterKind(str, Enum):
    STATIC_FILLER = "static_filler"
    INSTANCE_OWNER = "instance_owner"
    OTHER = "other"


@dataclass(frozen=True)
class AdjusterSpec:
    name: str
    fn: AdjusterFn
    step: Step
    phase: Phase
    watched_keys: Optional[Tuple[ConfigKeys, ...]]    # run only if these prefixes match changed keys
    owned_keys: Optional[Tuple[ConfigKeys, ...]]  # which keys this adjuster may modify
    kind: AdjusterKind = AdjusterKind.OTHER
    scope: AdjusterScope = AdjusterScope.EXPERIMENT
    requires_sample_manager: bool = True
    order: int = 100  # lower runs first. Alphabetically if identical.


def to_config_keys(x: ConfigKeysLike) -> ConfigKeys:
    return x if isinstance(x, tuple) else tuple(x.split('.'))


def to_config_keys_tuples(xs: Optional[Iterable[ConfigKeysLike]]) -> Optional[Tuple[ConfigKeys]]:
    if xs is None:
        return None
    out = tuple(to_config_keys(x) for x in xs)
    return out or None  # empty -> None

def patch_adjuster(*, step: Step | str, phase: Phase | str = Phase.PRE_VALIDATE,
                   scope: AdjusterScope | str = AdjusterScope.EXPERIMENT,
                   requires_sample_manager: bool = True,
                   watched_keys: Optional[Iterable[ConfigKeysLike]] = None,
                   owned_keys: Optional[Iterable[ConfigKeysLike]] = None,
                   kind: AdjusterKind | str = AdjusterKind.OTHER, order: int = 100) -> Callable[[AdjusterFn], AdjusterFn]:
    """
    Decorator for adjusters.
    Decorate a pure, idempotent (view, sm) -> patch function.

    Parameters
    ----------
    step: Step | str
        The step (one of APPLY_RENAMES, CREATE_PIPELINE_SECTIONS, CREATE_CHANNELS_RECONCILE,
                          ENSURE_STATIC_BLOCKS, POPULATE_DEFAULTS_MATERIALIZE, ADJUST)
                           at which to run this adjuster.
    phase: Phase | str
        The lifecycle phase (one of PRE_VALIDATE, POST_VALIDATE, PRE_COMMIT, POST_COMMIT)
                            at which to run this adjuster.
    watched_keys: Optional[Iterable[ConfigKeysLike]]
        If provided, only run this adjuster if any of these config key prefixes
        overlap with the changed keys in the current adjustment context.
    owned_keys: Optional[Iterable[ConfigKeysLike]]
        The config key prefixes that this adjuster may modify.
        Attempting to modify keys outside this set is an error.
    kind: AdjusterKind | str
        The kind of adjuster (STATIC_FILLER, INSTANCE_OWNER, OTHER).
    order: int
        The order in which to run this adjuster relative to others.
        This is to break ties when multiple adjusters are registered for the same
        step/phase. Lower order runs first.
    """  # FIXME: document args
    step = Step(step) if isinstance(step, str) else step
    phase = Phase(phase) if isinstance(phase, str) else phase
    kind = AdjusterKind(kind) if isinstance(kind, str) else kind
    def _wrap(fn: AdjusterFn) -> AdjusterFn:
        _REGISTRY.append(
            AdjusterSpec(
                name=fn.__name__, fn=fn, step=step, phase=phase,
                watched_keys=to_config_keys_tuples(watched_keys), owned_keys=to_config_keys_tuples(owned_keys),
                kind=kind, scope=scope, requires_sample_manager=requires_sample_manager, order=order))
        return fn
    return _wrap


_REGISTRY: List[AdjusterSpec] = []


def _config_keys_overlap(changed_keys: Optional[Iterable[ConfigKeysLike]],
                         watched_prefixes: Optional[Tuple[ConfigKeysLike, ...]]) -> bool:
    """
    True if `changed` touches any `keys` prefix.
    Example:
      changed=["registration.channels.Ch488.align_with"]
      keys=("registration.channels",)  -> True
      keys=("stitching.channels",)     -> False
    If `keys` is None or `changed` is None/empty, return True (do not filter).
    """
    changed_t = to_config_keys_tuples(changed_keys)
    watched_prefixes_t = to_config_keys_tuples(watched_prefixes)
    if not watched_prefixes_t or not changed_t:
        return True

    for changed_k in changed_t:
        for watched_k in watched_prefixes_t:
            if len(changed_k) >= len(watched_k) and changed_k[:len(watched_k)] == watched_k:
                return True
    return False
