from __future__ import annotations

from itertools import chain
from pathlib import Path
from typing import Dict, Any, List, Callable, Optional
from urllib.parse import urljoin

import yaml
from jsonschema.validators import validator_for
from referencing import Registry, Resource

YAML_EXTENSIONS = ('.yml', '.yaml')

def load_yaml_schema(path: Path) -> Dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding='utf-8'))
    if not isinstance(data, dict):
        raise ValueError(f'Schema at "{path}" did not parse to a mapping.')
    data.setdefault('$id', path.resolve().as_uri())
    return data


def build_schema_registry_from_dir(schemas_dir: Path, mutate: Optional[Callable[[Path, Dict[str, Any]], None]] = None) -> Registry:
    reg = Registry()
    schemas_dir = Path(schemas_dir)

    for p in chain(*(schemas_dir.glob(f'*.schema{ext}') for ext in YAML_EXTENSIONS)):
        data = load_yaml_schema(p)

        if mutate is not None:
            mutate(p, data)
        res = Resource.from_contents(data)
        declared_id = data['$id']
        # by declared id
        reg = reg.with_resource(declared_id, res)
        # by file:// path
        reg = reg.with_resource(p.resolve().as_uri(), res)
        # sibling alias for ./typedefs.schema.yaml style refs
        # base = declared_id.rsplit('/', 1)[0] + '/'
        base = urljoin(declared_id, '.')  # parent folder w/ trailing slash
        reg = reg.with_resource(urljoin(base, p.name), res)

        for ext in YAML_EXTENSIONS:
            if not declared_id.endswith(f'.schema{ext}'):
                reg = reg.with_resource(f'{declared_id}.schema{ext}', res)
    return reg


# FIXME: set develpopment flag to machine.verbosity < INFO
def compile_validator(schema: dict, *, registry=None, format_checker=None, development=True):
    Validator = validator_for(schema)
    if development:
        Validator.check_schema(schema)
    return Validator(schema, registry=registry, format_checker=format_checker)


def inject_image_content_types(typedefs: Dict[str, Any], content_types: List[str]) -> None:
    """
    Inject dynamic enum values into typedefs, e.g. for contentType.
    So that they are read from single source of truth in code and
    not duplicated verbatim in the schema file.

    Parameters
    ----------
    typedefs: Dict[str, Any]
        The typedefs schema dictionary to modify in place.
    """
    defs = typedefs.get('$defs') or {}
    if content_types is not None and 'imageContentType' in defs:
        content_type_def = defs['imageContentType']
        content_type_def.clear()
        content_type_def['anyOf'] = [
            {
                'type': 'string',
                'enum': sorted(map(str, content_types))
             },
            {
                'type': 'null'
            }
        ]
