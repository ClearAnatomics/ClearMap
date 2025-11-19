import re
from pathlib import Path
from typing import Any, Mapping, Dict, Iterable, Sequence

from jsonschema import FormatChecker

from ClearMap.IO.assets_constants import CONTENT_TYPE_TO_PIPELINE

from .json_schema_utils import load_yaml_schema, compile_validator, build_schema_registry_from_dir, \
    inject_image_content_types
from ..config_handler import ALTERNATIVES_REG


__all__ = ['SectionValidators', 'AggregatedValidationError', 'validate_sections_with_jsonschema']


class AggregatedValidationError(Exception):  # REFACTOR: maybe in exceptions module ??
    def __init__(self, messages: Iterable[str]):
        super().__init__("\n".join(messages))
        self.messages = list(messages)


FORMAT_CHECKER = FormatChecker()


@FORMAT_CHECKER.checks('extension_string')
def _check_extension_string(v: Any) -> bool:
    if not isinstance(v, str):
        return False
    return re.fullmatch(r"\.[A-Za-z0-9][A-Za-z0-9._-]*", v) is not None


def _path_str(path: Sequence[Any]) -> str:
    if not path:
        return '(root)'
    parts = []
    for p in path:
        parts.append(f'[{p}]' if isinstance(p, int) else str(p))
    s = '.'.join(parts)
    return s.replace('.[', '[')


def get_image_content_types() -> list[str]:
    try:
        return list(CONTENT_TYPE_TO_PIPELINE.keys())
    except Exception:
        return []


class SectionValidators:
    """
    Compiles & caches per-section JSON-Schema validators with shared typedefs and format checker.
    Private to the validators package; use via validate_sections_with_jsonschema or validate_all.
    """
    def __init__(self, schemas_dir: Path):
        self.schemas_dir = Path(schemas_dir)

        def _mutate(path: Path, schema: Dict[str, Any]) -> None:
            # Only touch typedefs; everything else is passed through.
            if path.name == 'typedefs.schema.yaml':
                inject_image_content_types(schema, content_types=get_image_content_types())
                # Keep a reference if we ever want to inspect typedefs.
                self.typedefs = schema

        self._registry = build_schema_registry_from_dir(self.schemas_dir, mutate=_mutate)
        self._compiled: Dict[str, Any] = {}

    def validator_for(self, section: str):
        if section not in self._compiled:
            schema = load_yaml_schema(self.schemas_dir / f'{section}.schema.yaml')
            self._compiled[section] = compile_validator(schema, registry=self._registry, format_checker=FORMAT_CHECKER)
        return self._compiled[section]


def validate_sections_with_jsonschema(config: Mapping[str, Mapping[str, Any]], sv: SectionValidators) -> None:
    """
    Validate each section of the config using JSON-Schema (Draft 2020-12).
    Uses the provided SectionValidators instance to load and cache schemas.

    Parameters
    ----------
    config
    sv

    Returns
    -------

    """

    messages: list[str] = []
    for section in ALTERNATIVES_REG.canonical_config_names:
        data = config.get(section)
        if data is None:
            continue
        v = sv.validator_for(section)

        # Not implemented yet but for future-proofing:
        # Optional custom messages via x-errorMessage in subschema
        for e in v.iter_errors(data):
            custom = None
            try:
                x = e.schema.get('x-errorMessage')
                if isinstance(x, dict):
                    custom = x.get(e.validator)
                elif isinstance(x, str):
                    custom = x
            except Exception:
                pass  # TODO: check if that will blow up at some point
            msg = custom or e.message
            where = _path_str(e.absolute_path)
            messages.append(f'[{section}] {where}: {msg}')
    if messages:
        raise AggregatedValidationError(messages)
