import shutil
from pathlib import Path
from typing import Dict, Any, Iterable, Optional, List

from ClearMap.config.config_handler import ConfigHandler, ALTERNATIVES_REG


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

    def __init__(self, *, base_dir: Optional[Path] = None, known_names: Optional[Iterable[str]] = None,
                 config_groups: Optional[List[List[str]]] = None) -> None:
        if base_dir is None:
            base_dir = Path.cwd()
        if known_names is not None:
            pass  # keep
        elif known_names is None and config_groups is not None:
            known_names = [names[0] for names in config_groups]
        else:
            known_names = ALTERNATIVES_REG.canonical_config_names
        self._config_handler = ConfigHandler(base_dir)
        self._known_names = list(known_names)

    def list_sections(self):
        return list(self._known_names)

    def set_base_dir(self, base_dir: Path) -> None:
        self._config_handler.src_dir = base_dir

    def base_dir(self) -> Path:
        return self._config_handler.src_dir

    def path_for(self, name: str, *, must_exist: bool = False) -> Path:
        """
        Resolve path for a logical config name, support alternative names and
        extensions ordered by preference (in ConfigLoader).
        """
        # REFACTOR: check if this shouldn't be in ConfigHandler directly
        name = ConfigHandler.normalise_cfg_name(name)

        if must_exist:
            # "Where is the current source file?"
            if ConfigHandler.is_global(name):
                path = ConfigHandler.get_global_path(name, must_exist=True)
                return Path(path)
            elif ConfigHandler.is_local(name):
                return self._config_handler.get_cfg_path(name, must_exist=True)
            else:
                # Legacy/odd names: fall back to defaults location
                path = ConfigHandler.get_default_path(name, must_exist=True)
                return Path(path)
        else:
            # "Where should we write this config *now*?"
            return ConfigHandler.resolve_write_path(name, base_dir=self.base_dir())

    @staticmethod
    def default_path_for(name: str, *, must_exist: bool = True) -> Path:
        """
        Resolve the packaged default for a logical config name.
        """
        return Path(ConfigHandler.get_default_path(name, must_exist=must_exist))

    def exists_any(self, name: str) -> bool:
        """
        Return True if a config file for 'name' exists in the experiment folder,
        considering all alternative names/extensions and legacy layouts.
        """
        loader = ConfigHandler(self.base_dir())
        try:
            # will raise if nothing can be found under any alternative
            loader.get_cfg_path(name, must_exist=True)
            return True
        except FileNotFoundError:
            return False

    def load(self, name: str) -> Dict[str, Any]:
        """
        Return a plain dict for this logical config. (or empty dict if missing).
        """
        try:
            cfg = self._config_handler.get_cfg(name, must_exist=False)
        except FileNotFoundError:
            cfg = None
        if cfg is None:
            return {}
        return _to_native_dict(cfg)

    def load_all(self) -> Dict[str, Dict[str, Any]]:
        return {name: self.load(name) for name in self._known_names}

    def commit(self, name: str, cfg: Dict[str, Any]) -> None:
        """
        Atomically write the given dict to the resolved path.
        Uses ConfigHandler's dump() to dispatch to the right format.
        The write is atomic: first to a temp file, then rename.
        2nd step (rename) is atomic on most OS/FS.
        1st step (write to temp) is not atomic, but should not leave a
        partial file behind (unless disk full or similar).
        """
        path = self.path_for(name, must_exist=False)
        self._config_handler.dump(path=path, data=cfg)

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
            if name == 'sample':
                cfg = self.load(name)
                cfg["sample_id"] = 'undefined'
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
