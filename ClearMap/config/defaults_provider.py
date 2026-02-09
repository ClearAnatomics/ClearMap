import warnings
from copy import deepcopy
from pathlib import Path
from typing import Dict, Any, Optional, Iterable

from .config_handler import ConfigHandler, INSTALL_CFG_DIR, ALTERNATIVES_REG
from .validators.json_schema_utils import build_schema_registry_from_dir, load_yaml_schema, compile_validator


JsonDict = Dict[str, Any]

VERSION = 'v3.1'  # REFACTOR: use CLEARMAP_VERSION instead or straight ConfigHAndler for default dirs
SCHEMAS_DIR = INSTALL_CFG_DIR / 'schemas' / VERSION

DEFAULTS_PROVIDER = None

def get_defaults_provider():
    global DEFAULTS_PROVIDER
    if DEFAULTS_PROVIDER is not None:
        return DEFAULTS_PROVIDER
    else:
        DEFAULTS_PROVIDER = DefaultsProvider(schemas_dir=SCHEMAS_DIR)
        return DEFAULTS_PROVIDER

# WARNING: for tests only
def set_defaults_provider(provider):
    global DEFAULTS_PROVIDER
    old = DEFAULTS_PROVIDER
    DEFAULTS_PROVIDER = provider
    return old


def _relax_schema_for_defaults(schema: dict) -> dict:
    """
    Produce a permissive variant of a JSON Schema for validating *defaults files*:
    - remove all 'required' constraints recursively
    - keep types/shapes/ranges (so user-provided values are type-checked)
    """
    from copy import deepcopy as _dc
    s = _dc(schema)
    def walk(node: Any):
        if isinstance(node, dict):
            node.pop('required', None)
            for v in node.values():
                walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)
    walk(s)
    return s


class DefaultsProvider:
    """
    Per-section defaults with precedence:
      1) user YAML: ~/.clearmap/defaults/<section>.yml  (validated statically)
      2) built-in YAML: <repo>/config/defaults/<section>.yml
      3) code fallback: provided dicts in code_defaults
    """

    def __init__(self,*, schemas_dir: Optional[Path] = None,
                 sections: Optional[Iterable[str]] = None,
                 tabs_only: bool = False, relax_required_for_defaults: bool = True,
                 code_defaults: Optional[Dict[str, JsonDict]] = None,
                 builtin_dir: Optional[Path] = None, user_dir: Optional[Path] = None  # For tests
                 ) -> None:
        """

        Parameters
        ----------
        schemas_dir: Path | None
            Directory with JSON schemas for static validation
            If None, no static validation is performed.
        sections: Iterable[str] | None
            Sections to load schemas for. If None, all known sections are used.
        tabs_only: bool
            If True, only load schemas for sections that correspond to tabs (local files).
        relax_required_for_defaults: bool
            If True, 'required' constraints are removed from schemas for static validation
        code_defaults: Dict[str, JsonDict] | None
            Fallback defaults per section if no YAML file is found.
            Useful for tests.
        builtin_dir: Path | None
            For tests: override built-in defaults dir
        user_dir: Path | None
            For tests: override user defaults dir
        """
        self.schemas_dir = Path(schemas_dir) if schemas_dir else None
        self.code_defaults = code_defaults or {}

        all_sections = list(sections) if sections is not None else ALTERNATIVES_REG.canonical_config_names
        if tabs_only:
            all_sections = [s for s in all_sections if ALTERNATIVES_REG.is_local_file(s)]

        self._validators: Dict[str, Any] = {}
        if self.schemas_dir:
            self._registry = build_schema_registry_from_dir(self.schemas_dir)
            for name in all_sections:
                schema_path = self.schemas_dir / f'{name}.schema.yaml'
                if schema_path.exists():
                    schema = load_yaml_schema(schema_path)
                    schema_used = _relax_schema_for_defaults(schema) if relax_required_for_defaults else schema
                    self._validators[name] = compile_validator(schema_used, registry=self._registry)

        # For tests
        self.__builtin_dir = Path(builtin_dir).expanduser().resolve() if builtin_dir else None
        self.__user_dir = Path(user_dir).expanduser().resolve() if user_dir else None
        if self.__user_dir:
            self.__user_dir.mkdir(parents=True, exist_ok=True)

    def _validate_static(self, section: str, data: JsonDict) -> bool:
        v = self._validators.get(section)
        if not v:
            return True
        try:
            for _ in v.iter_errors(data):
                return False
            return True
        except Exception as err:  # WARNING: broad
            warnings.warn(f'Static validation error for defaults section "{section}": {err}')
            return False

    def __get_cfg(self, section: str, from_package: bool) -> Optional[dict]:
        base_dir = self.__builtin_dir if from_package else self.__user_dir
        if ConfigHandler.is_global(section):
            cfg_path = ConfigHandler.get_global_path(section, must_exist=False,
                                                     base_dir=base_dir)  # For tests if not None
        else:
            cfg_path = ConfigHandler.get_default_path(section, must_exist=False, from_package=from_package,
                                                      base_dir=base_dir)  # For tests if not None
        if not cfg_path.exists():
            return None
        cfg = ConfigHandler.get_cfg_from_path(cfg_path)
        return dict(cfg) if cfg is not None else None


    def get(self, section: str) -> JsonDict:
        """Deep-copied defaults for a section, honoring precedence & static validation."""
        # 1) user
        user_cfg = self.__get_cfg(section, from_package=False)
        if user_cfg is not None and self._validate_static(section, user_cfg):
            if not user_cfg:
                raise ValueError(f'Invalid user config for section "{section}"')
            return deepcopy(user_cfg)
        # 2) built-in
        default_cfg = self.get_default_config(section)
        if not default_cfg:
            raise ValueError(f'Invalid default config for section "{section}"')
        return deepcopy(default_cfg)

    def ensure_user_file(self, section: str, out_path: Optional[Path] = None) -> Path:   # FIXME: unused
        """
        Ensure ~/.clearmap/defaults/<section>.yml exists by copying built-in (or code fallback).
        """
        if not self.__user_dir:
            raise RuntimeError("No user_dir configured")
        dst = ConfigHandler.get_default_path(section, must_exist=False, from_package=False)
        if dst.exists():
            return dst
        else:
            dst.parent.mkdir(parents=True, exist_ok=True)
            src_data = self.get_default_config(section)
            ConfigHandler.dump(dst, src_data)  # atomic write (can convert if format differs)
            return dst

    def get_default_config(self, section: str) -> dict[str, Any]:
        default_cfg = self.__get_cfg(section, from_package=True)
        if default_cfg is not None:
            return default_cfg
        else:
            return dict(self.code_defaults.get(section, {}))
