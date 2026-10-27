import importlib.util
from pathlib import Path

import pytest

SETUP_SCRIPT = Path(__file__).resolve().parents[2] / 'setup.py'


@pytest.fixture(scope='session')
def setup_script():
    """``setup.py``, imported: importing it builds nothing (see ``test_setup_script_is_importable.py``)"""
    spec = importlib.util.spec_from_file_location('clearmap_setup_script', SETUP_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
