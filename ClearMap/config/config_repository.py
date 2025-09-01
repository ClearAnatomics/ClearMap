from __future__ import annotations

import shutil
from pathlib import Path
from typing import Dict, Any, Iterable

import configobj

from ClearMap.IO.FileUtils import atomic_replace
from config_handler import ConfigHandler


def _to_native_dict(obj) -> Any:
    """
    Recursively convert ConfigObj/Section (and other mapping types) to plain dict/lists.
    """
    if isinstance(obj, dict):
        return {k: _to_native_dict(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [ _to_native_dict(x) for x in obj ]
    return obj


class ConfigRepository:
    """
    File I/O + atomic commits, powered by ConfigLoader's resolution:
    - logical 'name' -> path resolution (w/ alternative file names and extensions)
    - load/commit per file; load_all/commit_all across known names
    """

    def __init__(self, *, base_dir: Path, known_names: Iterable[str]):  # FIXME: get known_names from ConfigLoader ?
        self._loader = ConfigHandler(base_dir)
        self._known_names = list(known_names)

    def set_base_dir(self, base_dir: Path) -> None:
        self._loader.src_dir = base_dir

    def base_dir(self) -> Path:
        return self._loader.src_dir

    def path_for(self, name: str, *, must_exist: bool = False) -> Path:
        """
        Resolve path for a logical config name, support alternative names and
        extensions ordered by preference (in ConfigLoader).
        """
        return Path(self._loader.get_cfg_path(name, must_exist=must_exist))

    @staticmethod
    def default_path_for(name: str, *, must_exist: bool = True) -> Path:
        """
        Resolve the packaged default for a logical config name.
        """
        return Path(ConfigHandler.get_default_path(name, must_exist=must_exist))

    def load(self, name: str) -> Dict[str, Any]:
        """
        Return a plain dict for this logical config. (or empty dict if missing).
        """
        try:
            cfg_obj = self._loader.get_cfg(name, must_exist=False)
        except FileNotFoundError:
            cfg_obj = None
        if cfg_obj is None:
            return {}
        return _to_native_dict(cfg_obj)

    def load_all(self) -> Dict[str, Dict[str, Any]]:
        return {name: self.load(name) for name in self._known_names}

    def commit(self, name: str, cfg: Dict[str, Any]) -> None:
        """
        Atomically write the given dict to the resolved path.
        Uses ConfigObj for .cfg/.ini; JSON/YAML adapters could be added similarly.  # FIXME: this should use ConfigLoader instead
        """
        path = self.path_for(name, must_exist=False)
        ext = path.suffix.lower()
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.parent.mkdir(parents=True, exist_ok=True)

        if ext in (".cfg", ".ini"):
            # Write via ConfigObj to preserve its formatting expectations
            cfg_obj = configobj.ConfigObj(encoding="UTF8", indent_type="    ", unrepr=True)  # FIXME: no direct ConfigObj here
            cfg_obj.filename = str(tmp)
            # Replace content
            cfg_obj.clear()
            for k, v in cfg.items():
                cfg_obj[k] = v
            cfg_obj.write()
        else:
            # Fallback: simple JSON (extend if you add YAML/JSON support)
            import json
            with tmp.open("w", encoding="utf-8") as f:
                json.dump(cfg, f, indent=2)

        atomic_replace(tmp, path)

    def clone_from(self, template_dir: Path, dest_dir: Path) -> None:
        """
        Copy known config files from a template experiment dir.
        """
        dest_dir = Path(dest_dir)
        dest_dir.mkdir(parents=True, exist_ok=True)
        for name in self._known_names:
            # Try to resolve an existing file in the template with same alternatives/ext rules
            loader = ConfigHandler(template_dir)
            try:
                src = loader.get_cfg_path(name, must_exist=True)
            except FileNotFoundError:
                continue
            dst = dest_dir / src.name
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)

    def copy_from_defaults(self, dest_dir: Path | str) -> None:
        """
        Copy packaged defaults for each known config into dest_dir.
        """
        dest_dir = Path(dest_dir)
        dest_dir.mkdir(parents=True, exist_ok=True)
        for name in self._known_names:
            try:
                default_src = self.default_path_for(name, must_exist=True)
            except FileNotFoundError:
                continue  # TODO: log missing default?
            dest_path = dest_dir / default_src.name
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(default_src, dest_path)

            # reset sample_id to 'undefined' in the copied sample config
            if name == "sample":
                cfg = self.load(name)
                cfg['sample_id'] = 'undefined'
                self.commit(name, cfg)

    def ensure_present(self, name: str) -> Path | None:  # FIXME: use this to refactor above code
        """
        Ensure a config file exists in the current experiment directory.
        If missing, copy from packaged defaults. Return the dest path or None
        if no default exists for that name.
        """
        dest = self.path_for(name, must_exist=False)
        if dest.exists():
            return dest
        try:
            src = self.default_path_for(name, must_exist=True)
        except FileNotFoundError:
            return None
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
        return dest
