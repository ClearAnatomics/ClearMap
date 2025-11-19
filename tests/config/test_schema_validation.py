from copy import deepcopy
from pathlib import Path
import yaml
import pytest
from jsonschema import Draft202012Validator as Validator


from config.conftest import SECTIONS

SCHEMAS = {sec_name: f"{sec_name}.schema.yaml" for sec_name in SECTIONS}


def _load_yaml(p: Path):
    with p.open('r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def _load_schema(p: Path):
    data = _load_yaml(p)
    data.setdefault('$id', p.resolve().as_uri())  # base for relative id
    return data


@pytest.mark.parametrize('section', SECTIONS)
def test_example_yaml_validates_against_schema(src_dir: Path, schemas_dir: Path, schema_registry, section: str):
    schema = _load_schema(schemas_dir / f'{section}.schema.yaml')
    cfg = _load_yaml(src_dir / f'{section}.yml')

    Validator.check_schema(schema)
    validator = Validator(schema, registry=schema_registry)
    errors = list(validator.iter_errors(cfg))
    assert not errors, f'{section}.yml validation failed: {[e.message for e in errors]}'


def test_forbidden_channel_name_fails(src_dir: Path, schemas_dir: Path, schema_registry, tmp_repo: Path):
    schema = _load_schema(schemas_dir / 'sample.schema.yaml')
    bad = _load_yaml(tmp_repo / 'sample.yml')

    # Violates typedefs.channelName (only [A-Za-z0-9_])
    forbidden_name = 'bad-name-with-dashes'
    bad['channels'][forbidden_name] = bad['channels'].get('autofluorescence', {})
    validator = Validator(schema, registry=schema_registry)
    errors = list(validator.iter_errors(bad))
    assert errors, 'Expected validation errors for bad channel name'
    msgs = [e.message for e in errors]
    assert any('does not match' in m or 'pattern' in m for m in msgs)


def test_forbidden_extension_string_fails(src_dir: Path, schemas_dir: Path, schema_registry, tmp_repo: Path):
    schema = _load_schema(schemas_dir / 'sample.schema.yaml')
    validator = Validator(schema, registry=schema_registry)

    default = _load_yaml(tmp_repo / 'sample.yml')
    bad = deepcopy(default)

    default['default_tile_extension'] = ['.tif', '.npy']
    assert len(list(validator.iter_errors(default))) == 0

    # If your schema references typedefs.extensionString (^\.[A-Za-z0-9][A-Za-z0-9._-]*$),
    # this should fail:
    forbidden_extension = ['tif']  # missing dot
    bad['default_tile_extension'] = [forbidden_extension]
    errors = list(validator.iter_errors(bad))
    assert errors, 'Expected validation to fail for non-dot-prefixed extension'
