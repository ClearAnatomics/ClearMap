"""
ClearMap config validation — public API.
This module exposes stable entrypoints; implementations live in private modules.

Functions
---------
validate_all(config: Mapping[str, Mapping[str, Any]], *, schemas_dir: Path) ->
    Validate a full multisection config.

    1) JSON-Schema per section (Draft 2020-12), with typedefs and dynamic enums
    2) Semantic checks (cross-field/cross-section rules)

    Raises AggregatedValidationError with aggregated messages on failure.

validate_static(config: Mapping[str, Mapping[str, Any]], sv: SectionValidators) ->
    Validate each section of the config using JSON-Schema (Draft 2020-12).

    Uses the provided SectionValidators instance to load and cache schemas.

    Raises AggregatedValidationError with aggregated messages on failure.

run_semantic_checks(config: Mapping[str, Mapping[str, Any]], sv: SectionValidators) ->
"""
from pathlib import Path
from typing import Mapping, Any, Protocol, List

from ._schema import SectionValidators, AggregatedValidationError
from ._schema import validate_sections_with_jsonschema as validate_static
from ._semantic import run_semantic_checks


class ConfigValidator(Protocol):
    def __call__(self, working: Mapping[str, Any]) -> None: ...


_CUSTOM_VALIDATORS: List[ConfigValidator] = []


def register_validator(fn: ConfigValidator) -> None:
    _CUSTOM_VALIDATORS.append(fn)


def _run_custom_validators(config: Mapping[str, Any]) -> list[str]:
    msgs: list[str] = []
    for fn in _CUSTOM_VALIDATORS:
        try:
            fn(config)
        except Exception as e:
            msgs.append(str(e))
    return msgs


def validate_all(config: Mapping[str, Mapping[str, Any]], *,
                 schemas_dir: Path,
                 sections_validators: SectionValidators | None = None,
                 ) -> None:
    """
    Validate a full multisection config.
    This is typically a ConfigCoordinator's view.

    1) JSON-Schema per section (Draft 2020-12), with typedefs and dynamic enums
    2) Semantic checks (cross-field/cross-section rules)

    Raises AggregatedValidationError with aggregated messages on failure.
    """
    if sections_validators is None:
        if schemas_dir is None:
            raise ValueError("validate_all requires either schemas_dir or sv")
        else:
            sections_validators = SectionValidators(schemas_dir)
    validate_static(config, sections_validators)
    run_semantic_checks(config, sections_validators)   # sv available if a semantic check wants schema context
    extra = _run_custom_validators(config)   # legacy plug-ins
    if extra:
        raise AggregatedValidationError(extra)
