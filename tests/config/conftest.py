import shutil
from pathlib import Path
import pytest
import yaml

from ClearMap.config.validators.json_schema_utils import build_schema_registry_from_dir

VERSION = 'v3.1'

SECTIONS = ('sample', 'stitching', 'registration', 'cell_map', 'vasculature')

CONFIG_FILES = tuple(f"{name}.yml" for name in SECTIONS)


TESTS_DIR = Path(__file__).parent.parent
SRC_DIR = TESTS_DIR.parent / 'ClearMap' / 'config' / 'defaults' / VERSION


@pytest.fixture()
def sections() -> tuple[str, ...]:
    return SECTIONS


@pytest.fixture(scope='session')
def src_dir():
    assert SRC_DIR.exists(), f"Defaults directory not found: {SRC_DIR}"
    return SRC_DIR


@pytest.fixture(scope='session')
def schemas_dir(src_dir: Path) -> Path:
    return src_dir.parent.parent / 'schemas' / VERSION


@pytest.fixture(scope='session')
def load_yaml():
    def _load(path: Path):
        with open(path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    return _load


@pytest.fixture
def dump_yaml():
    def _dump(data, path: Path):
        with open(path, 'w', encoding='utf-8') as f:
            yaml.safe_dump(data, f, sort_keys=False)
    return _dump


# FIXME: implement this in real lib
@pytest.fixture(scope="session")
def schema_registry(schemas_dir, load_yaml):
    return build_schema_registry_from_dir(schemas_dir)


@pytest.fixture
def tmp_repo(tmp_path, src_dir):
    """
    Creates a temp repo that looks like a ClearMap experiment config folder,
    seeded with example YAMLs for safety.
    """
    for f_name in CONFIG_FILES:
        # Copy to temp folder
        shutil.copy(src_dir / f_name, tmp_path / f_name)
    return tmp_path
