from __future__ import annotations

import warnings
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from types import MappingProxyType
from typing import Dict, Any, Optional, Iterable, Mapping, List, Callable, TYPE_CHECKING
from copy import deepcopy

from ClearMap.Utils.event_bus import EventBus, BusSubscriberMixin
from ClearMap.Utils.events import CfgChanged, ChannelRenamed, ChannelsChanged
from ClearMap.Utils.exceptions import ChannelNameTakenError
from ClearMap.Utils.utilities import infer_origin_from_caller
from ClearMap.config.config_adjusters.patch_ops import deep_merge

from .channel import folded_channel_name
from .config_adjusters.type_hints import AdjustmentContext
from .config_adjusters.adjusters_api import Phase, ConfigKeys, AdjusterScope
from .config_adjusters.engine import run_adjusters
from .config_adjusters.policy import INSTANCE_SPECS_REGISTRY
from .config_handler import ALTERNATIVES_REG, ConfigHandler
from .config_repository import ConfigRepository
from .defaults_provider import DefaultsProvider, get_defaults_provider, SCHEMAS_DIR, set_defaults_provider
from .validators import validate_all, SectionValidators

if TYPE_CHECKING:
    from ClearMap.pipeline_orchestrators.sample_info_management import SampleManager


INSTALL_CFG_DIR = Path(__file__).parent
CURRENT_SCHEMAS_DIR = INSTALL_CFG_DIR / 'schemas' / 'v3.1'


def refuse_renames_onto_existing_channels(requested: Mapping[str, str], existing_channels: Iterable[str]) -> None:
    """
    Refuse (before anything changes) to rename a channel to the name of another existing channel (whatever the
    case): it would take over the settings and the results of that channel. The other channel must be renamed
    first. A channel can change the case of its own name.

    Raises
    ------
    ChannelNameTakenError
    """
    existing = list(existing_channels)
    for old, new in requested.items():
        if folded_channel_name(new) in {folded_channel_name(other) for other in existing if other != old}:
            raise ChannelNameTakenError(f'Cannot rename channel {old!r} to {new!r}: a channel named {new!r} exists '
                                        f'(names are unique whatever the case). Rename it first.', value=new)


def committed_channel_renames(requested: Mapping[str, str], channels_before: Iterable[str],
                              channels_after: Iterable[str]) -> dict[str, str]:
    """
    The requested channel renames that the commit carried out, as told by the channels before and after it:
    the old name is gone and the new name is new. The adjusters may drop a rename (e.g. of an unknown channel):
    a fact must describe what was written, not what was asked.
    """
    before, after = set(channels_before), set(channels_after)
    return {old: new for old, new in requested.items()
            if old in before and old not in after and new in after and new not in before}


@dataclass(frozen=True)
class CommitRecord:
    """
    What a commit did to the channels, taken right after it is written and before the post-commit hooks run:
    the hooks and the facts published for this commit all describe the same change, whatever the commits
    nested in the meantime.

    Attributes
    ----------
    channels_before, channels_after : tuple[str, ...]
        The names of the sample channels before and after the commit, in the order of the config.
    renames : Mapping[str, str]
        The renames the commit carried out ({old name: new name}), not the ones that were requested
        (see committed_channel_renames).
    """
    channels_before: tuple[str, ...]
    channels_after: tuple[str, ...]
    renames: Mapping[str, str]

    @property
    def channels_changed(self) -> bool:
        return set(self.channels_before) != set(self.channels_after)


def _patch_to_config_keys_sets(patch: Dict[str, Any]) -> List[ConfigKeys]:
    out: List[ConfigKeys] = []
    def rec(node, prefix: List[str]):
        if isinstance(node, dict):
            if not node:  # empty dict still means the prefix changed
                if prefix:
                    out.append(tuple(prefix))
            for k, v in node.items():
                rec(v, prefix + [k])
        else:
            if prefix:
                out.append(tuple(prefix))
    rec(patch, [])
    # de-duplicate and keep coarse granularity
    return sorted(set(out))


class ConfigCoordinator(BusSubscriberMixin):
    """
    Coordinates the app’s working configuration as one cohesive unit.

    Primitives
    ----------
    adjust_config(...):  run adjusters; return their patch (no validation, no IO)
    validate():        run validators; raise if invalid (no IO)
    commit():          write working config to disk atomically

    One-shots
    ---------
    submit_patch(patch, ...): merge → [adjust_config(filtered)] → [validate] → [commit]
    submit(...):               [adjust_config(full)] → [validate] → [commit]

    Notes
    -----
    - Adjusters are pure/idempotent; calling adjust_config multiple times is safe.
    - Validators must not mutate the model.
    - Tabs/controllers should not write files; all persistence flows through commit().
    """
    def __init__(self, *, config_repo: ConfigRepository, bus: EventBus,
                 scope: AdjusterScope = AdjusterScope.EXPERIMENT,
                 schemas_dir: Path | None = None,
                 defaults_provider: Optional[DefaultsProvider] = None) -> None:
        super().__init__(bus)
        self.working: Dict[str, Dict[str, Any]] = {}  # name -> cfg dict
        self._rev = 0

        self._config_repo = config_repo
        self.scope = scope

        if schemas_dir is None:
            schemas_dir = CURRENT_SCHEMAS_DIR
        self._schemas_dir = schemas_dir

        if defaults_provider is None:
            defaults_provider = get_defaults_provider()
        self.defaults_provider: Optional[DefaultsProvider] = defaults_provider

        self._lock = RLock()

        self._section_validators = SectionValidators(self._schemas_dir)
        self._active_sections: set[str] = set()
        self._post_commit_hooks: list[Callable[[CommitRecord], None]] = []

    @property
    def _allowed_sections(self):
        if not self._active_sections:
            return set(ALTERNATIVES_REG.canonical_config_names)  # TODO: chekc if this is the right fallback
        return set(self._active_sections) | set(ALTERNATIVES_REG.canonical_global_config_names)

    @property
    def active_sections(self) -> frozenset[str]:
        """The local sections declared active (empty means no filtering)."""
        return frozenset(self._active_sections)

    def add_post_commit_hook(self, hook: Callable[[CommitRecord], None]) -> None:
        """
        Register a callable run after every committed change (submit / submit_patch),
        *before* CfgChanged is published. It is given the CommitRecord of that commit.

        Use it to keep state derived from the config (e.g. the workspace) up to date,
        so that every CfgChanged subscriber sees it consistent with the config, whatever
        the subscription order (nested submits included).
        """
        self._post_commit_hooks.append(hook)

    def _run_post_commit_hooks(self, commit_record: CommitRecord) -> None:
        for hook in list(self._post_commit_hooks):
            hook(commit_record)

    def _record_commit(self, channels_before: list[str], rename_map: Mapping[str, str]) -> CommitRecord:
        """To be called right after the commit: the channels of the config are the ones it wrote"""
        channels_after = self.current_channels
        return CommitRecord(channels_before=tuple(channels_before), channels_after=tuple(channels_after),
                            renames=committed_channel_renames(rename_map, channels_before, channels_after))

    @property
    def workspace_config_path(self):
        return self._config_repo.base_dir() / 'workspace.yml'

    @classmethod
    def from_folder(cls, folder: Path, known_names: Optional[Iterable[str]], *,
                    scope: AdjusterScope = AdjusterScope.EXPERIMENT,
                    auto_load: bool = False, **kwargs) -> "ConfigCoordinator":
        if not known_names:
            known_names = ALTERNATIVES_REG.canonical_config_names
        repo = ConfigRepository(base_dir=folder, known_names=known_names)
        bus = EventBus()
        coord = cls(config_repo=repo, bus=bus, schemas_dir=CURRENT_SCHEMAS_DIR)
        if auto_load:
            coord.load_all()
        return coord

    def set_defaults_provider(self, provider) -> None:
        """Wire/replace the defaults provider after construction."""
        self.defaults_provider = provider
        set_defaults_provider(provider)

    def set_active_sections(self, sections: Optional[Iterable[str]]) -> None:
        """
        Declare which *local* config sections are in play for this experiment.
        Global sections (machine, display, etc.) stay untouched.

        This method:
          - Prunes self.working to only keep active+global sections.
          - Updates the repo so future load_all()/copy_from_defaults()
            only touch those sections (plus globals).
        """
        if sections is None:  # None -> all active -> no filtering
            self._active_sections = set()
            return

        active = {ConfigHandler.normalise_cfg_name(s) for s in sections}
        self._active_sections = active

        repo_sections = list(active) + ALTERNATIVES_REG.canonical_global_config_names
        self._config_repo.set_sections(repo_sections)

        # Prune working config
        keep = set(repo_sections)
        with self._lock:
            self.working = {name: cfg for name, cfg in self.working.items()
                            if name in keep}
            self._rev += 1

    def seed_missing_from_defaults(self, *, tabs_only: bool = True) -> None:
        """
        For any known section missing in the working model, seed from DefaultsProvider.
        Call this after load_all() when creating a new experiment.
        The templates reservoirs of the defaults are left out (see ContainerSpecRegistry.without_template_roots).
        """
        if not self.defaults_provider:
            return
        known = self._config_repo.list_sections()

        if tabs_only:
            known = [n for n in known if ALTERNATIVES_REG.is_local_file(n)]

        with self.__edit_session(origin="seed_defaults", validate=False) as wkng:
            for sec in known:
                if sec not in wkng or not wkng[sec]:
                    dflt = self.defaults_provider.get(sec)
                    if dflt:
                        wkng[sec] = INSTANCE_SPECS_REGISTRY.without_template_roots(sec, dflt)

    def config_exists(self, name: str) -> bool:
        return self.path_for(name, must_exist=False).exists()

    def set_base_dir(self, base_dir: Path) -> None:
        new_base = Path(base_dir).expanduser().resolve()
        old_base = self._config_repo.base_dir()
        if old_base is not None and new_base == old_base:
            return
        self._config_repo.set_base_dir(new_base)
        self.reset_working()

    def reset_working(self) -> None:
        """Clear the in-memory working config."""
        with self._lock:
            self.working = {}
            self._rev = 0

    @property
    def base_dir(self) -> Path:
        return self._config_repo.base_dir()

    def path_for(self, name: str, *, must_exist: bool = False) -> Path:
        return self._config_repo.path_for(name, must_exist=must_exist)

    def config_exists_any(self, name: str) -> bool:
        return self._config_repo.exists_any(name)

    def ensure_present(self, name: str) -> Path | None:
        """Ensure the config file exists in the experiment, written from the defaults without their templates"""
        return self._config_repo.ensure_present(name, section_filter=INSTANCE_SPECS_REGISTRY.without_template_roots)

    def load(self, name: str) -> Dict[str, Any]:
        with self._lock:
            data = self._config_repo.load(name)
            self.working[name] = deepcopy(data)
            return deepcopy(data)

    def load_all(self, sections=None) -> Dict[str, Dict[str, Any]]:
        self.reset_working()
        with self._lock:
            if sections:
                self.set_active_sections(sections)
            data = self._config_repo.load_all()
            for name, cfg in data.items():
                self.working[name] = deepcopy(cfg)
            return deepcopy(data)

    def clone_from(self, template_dir: str | Path, dest_dir: str | Path) -> None:
        """
        Clone the entire working config from another coordinator.
        """
        self._config_repo.clone_from(Path(template_dir), Path(dest_dir))

    def copy_from_defaults(self, dest_dir: str | Path) -> None:
        """
        Write all default configs to dest_dir, without their templates reservoirs: the templates only
        live in the defaults (where the templates resolver reads them), never in an experiment.
        """
        self._config_repo.copy_from_defaults(Path(dest_dir), section_filter=INSTANCE_SPECS_REGISTRY.without_template_roots)

    @property
    def rev(self) -> int:
        with self._lock:
            return self._rev

    def view(self) -> Mapping[str, Mapping[str, Any]]:
        """Return an immutable view of the current working config."""
        with self._lock:
            return MappingProxyType(self.working)

    # WARNING: avoid calling in loops because of deepcopy
    def get_config_view(self, cfg_name: str = '') -> Dict[str, Any]:
        """
        Read-only dict for UI using a deep copy of the current working config
        (or a single section if cfg_name is given).
        """
        with self._lock:  # Avoid mid-edit reads
            if cfg_name:
                section = self.working.get(cfg_name)
                return deepcopy(section) if isinstance(section, dict) else {}
            else:
                return deepcopy(self.working)

    @property
    def current_channels(self):
        chans_cfg = self.working.get('sample', {}).get('channels', [])
        if isinstance(chans_cfg, dict):
            return list(chans_cfg.keys())
        else:
            return chans_cfg

    @contextmanager
    def __edit_session(self, *, origin: str = "", validate: bool = True):
        """
        Yield a mutable copy of the current working config
        on exit, validate+commit.
        This edit is thread-safe.

        Any edit to the gated configuration should be done
        through this context manager to ensure thread safety
        validation and proper event notification.

        Parameters
        ----------
        origin: str
            Optional string to identify the source of the change
            (e.g. "user", "import", "sync", ...).
            It will be part of the "config_changed" event payload.
        validate: bool
            If True (default), run all validators before committing.
            If validation fails, the working config remains unchanged.
        """
        with self._lock:
            working_copy = deepcopy(self.working)
            yield working_copy  # mutations happen here

            if validate:
                self.validate(working_copy)

            # commit (single source of truth)
            self.working = working_copy
            self._rev += 1

    def _merge_patch(self, working_cfg: Dict[str, Dict[str, Any]], patch: Dict[str, Any], *,
                     allowed_sections: Optional[set[str]] = None) -> None:
        """
        Merge a patch dict into the working config.
        If any top-level key matches a known file name, direct merge into that.
        Otherwise, merge into a special 'global' config (create if missing).

        Parameters
        ----------
        working_cfg
        patch
        """
        extra_sections = set(patch.keys()) - (allowed_sections or set()) - {'origin'}
        if extra_sections:
            warnings.warn(f'Applying patch with keys outside active sections: {extra_sections}',
                          RuntimeWarning, stacklevel=2)
        targeted = False
        for name, subpatch in patch.items():
            if name == 'origin':
                continue
            if allowed_sections is not None and name not in allowed_sections:
                continue
            if isinstance(subpatch, dict):
                if name not in working_cfg or not isinstance(working_cfg.get(name), dict):
                    working_cfg[name] = {}
                deep_merge(working_cfg[name], subpatch)
                targeted = True
        if not targeted:
            return  # No-op if no targeted sections found
            # raise NotImplementedError(f'Global patches are not supported in this version.')
            # Merge into a default 'global' config (create if missing)
            deep_merge(working_cfg.setdefault('global', {}), patch)

    # def apply_section(self, name: str, patch: Dict[str, Any]) -> None:
    #     """
    #     Apply a patch to a single named config section.
    #     """
    #     origin = patch.pop('origin', None)
    #     if not origin:
    #         origin = infer_origin_from_caller()
    #     with self.__edit_session(origin=origin, validate=False) as working_cfg:
    #         if name not in working_cfg:
    #             working_cfg[name] = {}
    #         if isinstance(working_cfg[name], dict):
    #             deep_merge(working_cfg[name], patch)
    #         else:
    #             raise ValueError(f'Cannot apply patch to non-dict config section \'{name}\'')

    def _extract_channel_renames(self, patch: dict) -> tuple[dict, dict[str, str]]:
        """
        Recursively remove any {"$rename": {"channels": {...}}} blocks from `patch`,
        and return (cleaned_patch, rename_map).
        """
        patch = deepcopy(patch)
        rename_map: dict[str, str] = {}

        def rec(node):
            if not isinstance(node, dict):
                return
            # pull and remove this level's $rename
            r = node.pop("$rename", None)
            if isinstance(r, dict):
                ch_map = r.get("channels")
                if isinstance(ch_map, dict):
                    rename_map.update({str(k): str(v) for k, v in ch_map.items() if k and v and k != v})
            # continue recursion
            for v in node.values():
                rec(v)

        rec(patch)
        return patch, rename_map

    def submit_patch(self, patch: dict, *, sample_manager: Optional[SampleManager],
                     origin: str | None = "ui", phase=Phase.PRE_VALIDATE) -> None:
        """
        Apply `patch` to a copy of the working config, then adjust_config (adjusters
        filtered by the patch's changed keys), validate, and commit. Every committed config went
        through the adjusters and the validators.

        The copy replaces the working config only once validated: if the adjusters or the validators
        refuse the patch, it raises and nothing changed (neither the working config nor the files),
        and nothing is published.
        """
        if not patch:
            return

        if origin:
            patch = dict(patch)  # copy to avoid mutating caller's dict
            patch['origin'] = origin

        channels_before = self.current_channels

        # Swap $rename directives to channels in patch and extract map for SampleManager
        clean_patch, rename_map = self._extract_channel_renames(patch)
        refuse_renames_onto_existing_channels(rename_map, channels_before)

        allowed = self._allowed_sections
        # The patched copy replaces the working config only once validated: a refused patch leaves no trace
        try:
            with self.__edit_session(origin=origin) as working_cfg:
                view_before_patch = deepcopy(working_cfg)
                self._merge_patch(working_cfg, clean_patch, allowed_sections=allowed)
                changed_keys = _patch_to_config_keys_sets(clean_patch)

                # Pass rename map to sample manager — the APPLY_RENAMES adjuster
                # reads ctx.sample_manager.renamed_channels
                if rename_map and sample_manager is not None:
                    sample_manager.set_renamed_channels(rename_map)

                # The runner merges each adjuster result into the view it is given (in place). Hand it a copy
                # so that only the returned patch, clipped to the active sections, reaches the working config.
                patch2 = self.adjust_config(sample_manager=sample_manager, phase=phase, view=deepcopy(working_cfg),
                                            active_sections=None, changed_keys=changed_keys,
                                            view_before_patch=view_before_patch)
                if patch2:
                    self._merge_patch(working_cfg, patch2, allowed_sections=allowed)
                    changed_keys.extend(_patch_to_config_keys_sets(patch2))
        finally:  # The renames are for the adjusters of this patch only, accepted or refused
            if rename_map and sample_manager is not None:
                sample_manager.clear_renamed_channels()

        changed_keys = sorted(set(changed_keys))  # Deduplicate

        changed_sections = {k[0] for k in changed_keys if k}
        self.commit(sections=list(changed_sections))
        commit_record = self._record_commit(channels_before, rename_map)
        self._run_post_commit_hooks(commit_record)

        self.publish(CfgChanged(changed_keys=tuple(".".join(k) for k in changed_keys)))
        self._publish_channel_facts(commit_record)

    def _publish_channel_facts(self, commit_record: CommitRecord) -> None:
        if commit_record.channels_changed:
            self.publish(ChannelsChanged(before=list(commit_record.channels_before),
                                         after=list(commit_record.channels_after)))
        for old, new in commit_record.renames.items():
            self.publish(ChannelRenamed(old=old, new=new))

    def submit(self, *, sample_manager: Optional[SampleManager] = None, phase=Phase.PRE_VALIDATE) -> None:
        """
        Run adjusters on the current working config (unfiltered), then validate and commit.
        Use this when you haven't just applied a new patch.
        """
        channels_before = self.current_channels

        # The adjusted copy replaces the working config only once validated: a refused config leaves no trace
        with self.__edit_session(origin='adjusters') as working_cfg:
            applied_patch = self.adjust_config(sample_manager=sample_manager, changed_keys=None, phase=phase,
                                               active_sections=None, view=deepcopy(working_cfg))
            if applied_patch:
                self._merge_patch(working_cfg, applied_patch, allowed_sections=self._adjusted_sections(None))
        self.commit()
        commit_record = self._record_commit(channels_before, {})
        self._run_post_commit_hooks(commit_record)

        self._publish_channel_facts(commit_record)

        changed_keys = _patch_to_config_keys_sets(applied_patch)
        self.publish(CfgChanged(changed_keys=tuple(".".join(k) for k in changed_keys)))

    def adjust_config(self, *, sample_manager: Optional[SampleManager] = None,
                      phase: Phase = Phase.PRE_VALIDATE,
                      active_sections: Optional[Iterable[str]] = None,
                      changed_keys: Optional[Iterable[ConfigKeys]] = None,
                      view: Optional[Mapping[str, Any]] = None,
                      view_before_patch: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
        """
        Run all config adjusters on the current working config,
        optionally filtered by changed_keys and pipelines.
        A global patch dict is returned; the caller merges it (inside an edit session).

        Parameters
        ----------
        sample_manager
        phase
        active_sections
        changed_keys
        view_before_patch
            The config before the patch which changed changed_keys (for the adjusters which follow a change
            of what they are derived from, see AdjustmentContext.view_before_patch).

        Returns
        -------

        """
        if view is None:
            view = self.get_config_view()

        active_sections = self._adjusted_sections(active_sections)

        ctx = self._build_context(sample_manager, view_before_patch=view_before_patch)

        patch = run_adjusters(view=view, ctx=ctx, phase=phase, active_sections=active_sections,
                              changed_keys=changed_keys)
        if patch:
            patch = {k: v for k, v in patch.items() if k in active_sections}
        return patch

    def _adjusted_sections(self, active_sections: Optional[Iterable[str]]) -> set[str]:
        """The sections the adjusters may write: the given (or the active) local sections, and the global ones."""
        sections = set(active_sections) if active_sections is not None else set(self._active_sections)
        return sections | set(ALTERNATIVES_REG.canonical_global_config_names)

    def _build_context(self, sample_manager, view_before_patch=None) -> AdjustmentContext:
        if self.scope == AdjusterScope.EXPERIMENT:
            if sample_manager is None:
                raise ValueError('EXPERIMENT coordinator requires a sample_manager')
            ctx = AdjustmentContext(
                scope=AdjusterScope.EXPERIMENT,
                sample_manager=sample_manager,
                group_base_dir=None,
                run_label=str(self.base_dir),
                view_before_patch=view_before_patch,
            )
        elif self.scope == AdjusterScope.GROUP:
            if sample_manager is not None:
                raise ValueError('GROUP coordinator must not receive a sample_manager')
            ctx = AdjustmentContext(
                scope=AdjusterScope.GROUP,
                sample_manager=None,
                group_base_dir=self.base_dir,  # group base dir == repo base dir
                run_label=str(self.base_dir),
            )
        else:
            raise ValueError(f'Unknown coordinator scope: {self.scope}')
        return ctx

    def validate(self, working_copy=None) -> None:
        """
        Ensure the current working config is valid.
        Raises if not valid.
        """
        if working_copy is None:
            working_copy = self.working
        validate_all(working_copy, sections_validators=self._section_validators,
                     schemas_dir=self._schemas_dir)  # Just in case

    def commit(self, sections: Optional[List[str]] = None) -> None:
        """
        Persist working configs to disk. If `sections` is given, only those
        sections are written; otherwise all are.
        """
        to_write = sections or list(self.working.keys())
        for name in to_write:
            cfg = self.working.get(name)
            if cfg is None:
                continue
            self._config_repo.commit(name, cfg)

    def snapshot_to(self, target_dir: Path | str) -> Path:
        """
        Write each working cfg into target_dir

        Parameters
        ----------
        target_dir: Path | str
            The target directory where to write the snapshot.
            Existing files will be overwritten.

        Returns
        -------
        The target directory path.
        """
        target_dir = Path(target_dir)
        target_dir.mkdir(parents=True, exist_ok=True)
        orig = self.base_dir
        try:
            self.set_base_dir(target_dir)
            self.commit()
        finally:
            self.set_base_dir(orig)
        return target_dir


def make_cfg_coordinator_factory(bus, scope=AdjusterScope.EXPERIMENT):
    def factory(base_dir, config_groups=None, scope=scope):
        cfg_repo = ConfigRepository(base_dir=base_dir)
        defaults_provider = get_defaults_provider()
        return ConfigCoordinator(config_repo=cfg_repo, bus=bus, scope=scope,
                                 schemas_dir=SCHEMAS_DIR, defaults_provider=defaults_provider)
    return factory
