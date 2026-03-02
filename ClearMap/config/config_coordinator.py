from __future__ import annotations

import warnings
from contextlib import contextmanager
from pathlib import Path
from threading import RLock
from types import MappingProxyType
from typing import Dict, Any, Optional, Iterable, Mapping, List, TYPE_CHECKING
from copy import deepcopy

from ClearMap.Utils.event_bus import EventBus, BusSubscriberMixin
from ClearMap.Utils.events import CfgChanged, ChannelRenamed, ChannelsChanged
from ClearMap.Utils.utilities import infer_origin_from_caller, deep_merge

from .config_adjusters.type_hints import AdjustmentContext
from .config_adjusters.adjusters_api import Phase, ConfigKeys, AdjusterScope
from .config_adjusters.engine import run_adjusters
from .config_handler import ALTERNATIVES_REG, ConfigHandler
from .config_repository import ConfigRepository
from .defaults_provider import DefaultsProvider, get_defaults_provider, SCHEMAS_DIR, set_defaults_provider
from .validators import validate_all, SectionValidators

if TYPE_CHECKING:
    from ClearMap.pipeline_orchestrators.sample_info_management import SampleManager


INSTALL_CFG_DIR = Path(__file__).parent
CURRENT_SCHEMAS_DIR = INSTALL_CFG_DIR / 'schemas' / 'v3.1'


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
    apply(patch):      merge patch into working model (no validation, no IO)
    adjust_config(...):  run adjusters to derive config; merge their patch (no validation, no IO)
    validate():        run validators; raise if invalid (no IO)
    commit():          write working config to disk atomically

    One-shots
    ---------
    submit_patch(patch, ...): apply → [adjust_config(filtered)] → [validate] → [commit]
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

    @property
    def _allowed_sections(self):
        if not self._active_sections:
            return set(ALTERNATIVES_REG.canonical_config_names)  # TODO: chekc if this is the right fallback
        return set(self._active_sections) | set(ALTERNATIVES_REG.canonical_global_config_names)

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
                        wkng[sec] = dflt

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
        return self._config_repo.ensure_present(name)

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
        Copy all default configs to dest_dir.
        """
        self._config_repo.copy_from_defaults(Path(dest_dir))

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

    def _apply(self, patch: Dict[str, Any]) -> None:  # WARNING: why no validate here?
        """
        Apply a patch possibly targeting multiple files.
        Policy: top-level keys select target configs if they exist in working;
        otherwise, apply into a special 'global' config.
        """
        origin = patch.pop('origin', None) or infer_origin_from_caller()

        allowed = self._allowed_sections
        extra = set(patch.keys()) - allowed - {'origin'}
        if extra:
            warnings.warn(f'Applying patch with keys outside active sections: {extra}', RuntimeWarning, stacklevel=2)
        with self.__edit_session(origin=origin, validate=False) as working_cfg:
            self._merge_patch(working_cfg, patch, allowed_sections=allowed)

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

    def submit_patch(self, patch: dict, *, sample_manager: Optional[SampleManager], do_run_adjusters: bool = True,
                     validate: bool = True, commit: bool = True, origin: str | None = "ui",
                     phase=Phase.PRE_VALIDATE) -> None:
        """
        Apply `patch` to the working config, then optionally adjust_config (adjusters
        filtered by the patch's changed keys), validate, and commit.

        Equivalent to:
            self.apply(patch, origin=origin)
            if run_adjusters: self.adjust_config(changed_keys=_patch_to_config_keys_sets(patch), sample_manager=sample_manager)
            if validate:    self.validate()
            if commit:      self.commit()
        """
        if not patch:
            return

        if not commit:
            warnings.warn(f'ConfigCoordinator.submit_patch called with commit=False; '
                          f'working model updated in-memory, but no events were emitted and'
                          f' nothing was written to disk. Call commit() later to persist '
                          f'and emit CfgChanged.', RuntimeWarning, stacklevel=2)
        if origin:
            patch = dict(patch)  # copy to avoid mutating caller's dict
            patch['origin'] = origin

        channels_before = self.current_channels

        # Swap $rename directives to channels in patch and extract map for SampleManager
        clean_patch, rename_map = self._extract_channel_renames(patch)

        allowed = self._allowed_sections
        with self.__edit_session(origin=origin, validate=False) as working_cfg:
            self._merge_patch(working_cfg, clean_patch, allowed_sections=allowed)
            changed_keys = _patch_to_config_keys_sets(clean_patch)
            if do_run_adjusters:
                patch2 = self.adjust_config(sample_manager=sample_manager, phase=phase, view=working_cfg,
                                            active_sections=None, changed_keys=changed_keys, apply=False)
                if patch2:
                    self._merge_patch(working_cfg, patch2, allowed_sections=allowed)
                    changed_keys.extend(_patch_to_config_keys_sets(patch2))

        changed_keys = sorted(set(changed_keys))  # Deduplicate

        if validate:
            self.validate()

        if commit:
            changed_sections = {k[0] for k in changed_keys if k}
            self.commit(sections=list(changed_sections))

            self.publish(CfgChanged(changed_keys=tuple(".".join(k) for k in changed_keys)))

            channels_after = self.current_channels
            if set(channels_before) != set(channels_after):
                self.publish(ChannelsChanged(before=channels_before, after=channels_after))

            for old, new in rename_map.items():
                self.publish(ChannelRenamed(old=old, new=new))

    def submit(self, *, sample_manager: Optional[SampleManager] = None, do_run_adjusters: bool = True,
               validate: bool = True, commit: bool = True, phase=Phase.PRE_VALIDATE) -> None:
        """
        Run adjusters on the current working config (unfiltered), then optionally
        validate and commit. Use this when you haven't just applied a new patch.
        """
        channels_before = self.current_channels

        applied_patch = {}
        if do_run_adjusters:
            applied_patch = self.adjust_config(sample_manager=sample_manager, changed_keys=None,
                                               phase=phase, active_sections=None, apply=True)
        if validate:
            self.validate()
        if commit:
            self.commit()

            channels_after = self.current_channels
            if set(channels_before) != set(channels_after):
                self.publish(ChannelsChanged(before=channels_before, after=channels_after))

            changed_keys = _patch_to_config_keys_sets(applied_patch)
            self.publish(CfgChanged(changed_keys=tuple(".".join(k) for k in changed_keys)))

    def adjust_config(self, *, sample_manager: Optional[SampleManager] = None,
                      phase: Phase = Phase.PRE_VALIDATE,
                      active_sections: Optional[Iterable[str]] = None,
                      changed_keys: Optional[Iterable[ConfigKeys]] = None,
                      view: Optional[Mapping[str, Any]] = None,
                      apply: bool=True) -> Dict[str, Any]:
        """
        Run all config adjusters on the current working config,
        optionally filtered by changed_keys and pipelines.
        A global patch dict is returned.
        If `apply` is True (default), the patch is merged into the working config.

        Parameters
        ----------
        sample_manager
        phase
        active_sections
        changed_keys
        apply

        Returns
        -------

        """
        if view is None:
            view = self.get_config_view()

        active_sections = set(active_sections) if active_sections is not None else set(self._active_sections)
        active_sections = active_sections | set(ALTERNATIVES_REG.canonical_global_config_names)

        ctx = self._build_context(sample_manager)

        patch = run_adjusters(view=view, ctx=ctx, phase=phase, active_sections=active_sections,
                              changed_keys=changed_keys)
        if patch:
            patch = {k: v for k, v in patch.items() if k in active_sections}
        if apply and patch:
            with self.__edit_session(origin="adjusters", validate=False) as working_cfg:
                self._merge_patch(working_cfg, patch, allowed_sections=active_sections)
        return patch

    def _build_context(self, sample_manager) -> AdjustmentContext:
        if self.scope == AdjusterScope.EXPERIMENT:
            if sample_manager is None:
                raise ValueError('EXPERIMENT coordinator requires a sample_manager')
            ctx = AdjustmentContext(
                scope=AdjusterScope.EXPERIMENT,
                sample_manager=sample_manager,
                group_base_dir=None,
                run_label=str(self.base_dir),
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


def make_cfg_coordinator_factory(bus):
    def factory(base_dir, config_groups=None):
        cfg_repo = ConfigRepository(base_dir=base_dir)
        defaults_provider = get_defaults_provider()
        return ConfigCoordinator(
            config_repo=cfg_repo,
            bus=bus,
            schemas_dir=SCHEMAS_DIR,
            defaults_provider=defaults_provider,
        )
    return factory
