from __future__ import annotations

from contextlib import contextmanager
from threading import RLock
from types import MappingProxyType
from typing import Dict, Any, Optional, Callable, Iterable, Mapping
from copy import deepcopy

from ClearMap.Utils.event_bus import EventBus


def _deep_merge(dst: Dict[str, Any], src: Dict[str, Any]) -> Dict[str, Any]:
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _deep_merge(dst[k], v)  # type: ignore[index]
        else:
            dst[k] = deepcopy(v)
    return dst


class ConfigCoordinator:
    """
    In-memory working model + pipeline:
      apply(patch) -> materialize() -> validate() -> commit_all(repo)
    """
    def __init__(self, *, validators=None,
                 bus: Optional[EventBus] = None,
                 materializers: Optional[Iterable[Callable[[Dict[str, Dict]], Dict[str, Any]]]] = None) -> None:
        self.working: Dict[str, Dict[str, Any]] = {}  # name -> cfg dict
        self._rev = 0
        self._validators = validators
        self._bus = bus
        self._materializers = list(materializers) if materializers else []

        self._lock = RLock()

    def set_event_bus(self, bus) -> None:
        """Wire/replace the event bus after construction."""
        self._bus = bus

    # RO interface
    @property
    def rev(self) -> int:
        return self._rev

    def view(self) -> Mapping[str, Mapping[str, Any]]:
        """Return an immutable view of the current working config."""
        return MappingProxyType(self.working)

    def get(self, name: str) -> Optional[Mapping[str, Any]]:
        sect = self.working.get(name)
        return MappingProxyType(sect) if isinstance(sect, dict) else None

    def set_working(self, name: str, cfg: Dict[str, Any]) -> None:
        self.working[name] = deepcopy(cfg)

    def get_config(self, name: str) -> Optional[Dict[str, Any]]:
        return self.working.get(name)

    def get_config_view(self, cfg_name: str = '') -> Dict[str, Any]:
        """
        Read-only dict for UI. We expose a {name -> cfg} mapping.
        (If you prefer a merged 'global' view, adapt params accordingly.)
        """
        if cfg_name:
            sect = self.working.get(cfg_name)
            return deepcopy(sect) if isinstance(sect, dict) else {}
        else:
            return deepcopy(self.working)

    @contextmanager
    def edit_session(self, *, origin: str = "", validate: bool = True):
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

            if validate and self._validators:
                self._validators.validate_all(working_copy)  # raise on error

            # commit (single source of truth)
            self.working = working_copy
            self._rev += 1
            if self._bus:
                self._bus.publish("config_changed", {"origin": origin, "rev": self._rev})

    def _merge_patch(self, working_cfg: Dict[str, Dict[str, Any]], patch: Dict[str, Any]) -> None:
        """
        Merge a patch dict into the working config.
        If any top-level key matches a known file name, direct merge into that.
        Otherwise, merge into a special 'global' config (create if missing).

        Parameters
        ----------
        working_cfg
        patch
        """
        targeted = False
        for name in list(self.working.keys()):
            if name in patch and isinstance(patch[name], dict):
                _deep_merge(working_cfg[name], patch[name])
                targeted = True
        if not targeted:
            # Merge into a default 'global' config (create if missing)
            _deep_merge(working_cfg.setdefault('global', {}), patch)

    def apply(self, patch: Dict[str, Any]) -> None:  # WARNING: why no validate here?
        """
        Apply a patch possibly targeting multiple files.
        Policy: top-level keys select target configs if they exist in working;
        otherwise, apply into a special 'global' config.
        """
        with self.edit_session(origin=patch['origin'], validate=False) as working_cfg:
            self._merge_patch(working_cfg, patch)

    def materialize(self) -> None:
        """
        Call pure materializers with the entire working set.
        Each materializer returns a dict patch (possibly multi-file); merge it.
        """
        with self.edit_session(origin="materialize", validate=False) as working_cfg:
            for mat in self._materializers:
                patch = mat(self.get_config_view())  # pass a copy
                if not patch:
                    continue
                self._merge_patch(working_cfg, patch)

    def validate(self) -> None:
        if self._validators:
            self._validators.validate_all(self.working)

    def commit_all(self, repo) -> None:
        for name, cfg in self.working.items():
            repo.commit(name, cfg)
