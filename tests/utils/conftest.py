import importlib.util
import subprocess
import sys
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


#: install_gui.sh creates its temporary environment with python, packaging, pyyaml and configobj only
PACKAGES_MISSING_FROM_THE_BOOTSTRAP_ENVIRONMENT = ('numpy', 'psutil')


@pytest.fixture
def import_in_bootstrap_environment():
    """
    A function importing a module in a new interpreter where numpy and psutil cannot be imported, like in the temporary
    environment of install_gui.sh: it returns the finished process (returncode, stderr).
    """
    def import_module(module_name):
        script = ('import importlib.abc, sys\n'
                  'class MissingPackages(importlib.abc.MetaPathFinder):\n'
                  '    def find_spec(self, name, path, target=None):\n'
                  f'        if name.split(".")[0] in {PACKAGES_MISSING_FROM_THE_BOOTSTRAP_ENVIRONMENT!r}:\n'
                  '            raise ModuleNotFoundError(f"{name} is not in the bootstrap environment", name=name)\n'
                  'sys.meta_path.insert(0, MissingPackages())\n'
                  f'import {module_name}\n')
        return subprocess.run([sys.executable, '-c', script], capture_output=True, text=True, check=False)
    return import_module
