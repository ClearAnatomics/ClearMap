"""
test_cuda_version_selection.py
==============================

Regression tests for the CUDA‑suffix and toolkit‑pin selection logic in
``install_utils.patch_env`` (pip mode, Linux).
"""

import sys
import types
from pathlib import Path

import pytest
import yaml
from packaging.version import Version
from ClearMap.Utils.install_utils import patch_env, PytorchVersionManager

# ---------------------------------------------------------------------------
# install_utils does ``from ClearMap import Settings`` at module level.
# Fake it out so the test file can be run without ClearMap installed.
# ---------------------------------------------------------------------------
if "ClearMap" not in sys.modules:
    _settings = types.ModuleType("ClearMap.Settings")
    _settings.clearmap_path = "/tmp"

    _cm = types.ModuleType("ClearMap")
    _cm.Settings = _settings

    sys.modules["ClearMap"] = _cm
    sys.modules["ClearMap.Settings"] = _settings


# ── fixtures ──────────────────────────────────────────────────────────────

@pytest.fixture
def env_yaml(tmp_path):
    """Write a minimal conda environment YAML; return *(src, dest)* paths."""
    cfg = {
        "name": "test_env",
        "channels": ["defaults", "nvidia", "pytorch"],
        "dependencies": [
            "python=3.10",
            "pytorch=2.1",
            "pytorch-cuda=11.8",
            "mkl",
            "cudatoolkit",
            "numpy",
        ],
    }
    src = tmp_path / "env_src.yml"
    dest = tmp_path / "env_dest.yml"
    src.write_text(yaml.dump(cfg))
    return src, dest


# ── helpers ───────────────────────────────────────────────────────────────

def _parse_pip_entries(dest_path):
    """Return every string in the pip sub-list of the written YAML."""
    with open(dest_path) as fh:
        cfg = yaml.safe_load(fh)
    for dep in cfg.get("dependencies", []):
        if isinstance(dep, dict) and "pip" in dep:
            return dep["pip"]
    return []


def _read_pip_cuda_suffix(dest_path):
    """Return the trailing ``cuXYZ`` token from the pip extra-index-url."""
    for entry in _parse_pip_entries(dest_path):
        if "--extra-index-url" in entry and "pytorch.org" in entry:
            return entry.rsplit("/", 1)[-1]
    return None


def _read_toolkit_pin(dest_path):
    """Return the ``cuda-toolkit==X.Y.*`` line, or None."""
    for entry in _parse_pip_entries(dest_path):
        if entry.startswith("cuda-toolkit=="):
            return entry
    return None


def _run_patch_env(src, dest, driver_cuda, monkeypatch):
    """Call ``patch_env`` with a faked driver CUDA version on Linux."""
    monkeypatch.setattr(
        PytorchVersionManager,
        "cuda_version",
        property(lambda self, _v=driver_cuda: Version(_v)),
    )
    monkeypatch.setattr("ClearMap.Utils.install_utils.platform.system", lambda: "Linux")

    patch_env(
        cfg_path=src,
        dest_path=dest,
        use_cuda_torch=True,
        pip_mode=True,
        use_spyder=False,
        tmp_dir="/tmp",
    )


# ── suffix selection ──────────────────────────────────────────────────────
#
# Viable versions hard-coded in patch_env:
#   11.8   12.4   12.6   12.8   13.0
#
# Correct rule: pick the LARGEST viable version v ≤ driver_cuda

@pytest.mark.parametrize(
    "driver_cuda, expected_suffix",
    [
        # exact matches
        ("11.8", "cu118"),
        ("12.4", "cu124"),
        ("12.6", "cu126"),
        ("12.8", "cu128"),
        ("13.0", "cu130"),
        # driver between two viable versions → must round DOWN
        ("12.0", "cu118"),
        ("12.3", "cu118"),
        ("12.5", "cu124"),
        ("12.7", "cu126"),
        ("12.9", "cu128"),
        # driver newer than every candidate → pick highest
        ("13.2", "cu130"),
    ],
    ids=lambda v: str(v),
)
def test_cuda_suffix_matches_driver(env_yaml, driver_cuda, expected_suffix, monkeypatch):
    """patch_env must choose the largest CUDA toolkit ≤ the driver version."""
    src, dest = env_yaml
    _run_patch_env(src, dest, driver_cuda, monkeypatch)

    suffix = _read_pip_cuda_suffix(dest)
    assert suffix == expected_suffix, (
        f"Driver CUDA {driver_cuda}: expected {expected_suffix!r}, got {suffix!r}"
    )


@pytest.mark.parametrize("driver_cuda", ["10.2", "11.0", "11.7"])
def test_driver_too_old_raises(env_yaml, driver_cuda, monkeypatch):
    """If the driver is older than every viable toolkit, raise ValueError."""
    src, dest = env_yaml
    with pytest.raises(ValueError, match="No matching CUDA version"):
        _run_patch_env(src, dest, driver_cuda, monkeypatch)


# ── toolkit pin ───────────────────────────────────────────────────────────

@pytest.mark.parametrize(
    "driver_cuda, expected_pin",
    [
        ("11.8", "cuda-toolkit==11.8.*"),
        ("12.4", "cuda-toolkit==12.4.*"),
        ("12.5", "cuda-toolkit==12.4.*"),   # rounded down to nearest viable
        ("12.8", "cuda-toolkit==12.8.*"),
        ("13.2", "cuda-toolkit==13.0.*"),   # capped at highest viable
    ],
    ids=lambda v: str(v),
)
def test_toolkit_pin_present(env_yaml, driver_cuda, expected_pin, monkeypatch):
    """The generated env must pin cuda-toolkit to the *selected* version,
    not the driver version, so pip can't pull a newer incompatible runtime."""
    src, dest = env_yaml
    _run_patch_env(src, dest, driver_cuda, monkeypatch)
    assert _read_toolkit_pin(dest) == expected_pin


def test_no_toolkit_pin_for_cpu(env_yaml, monkeypatch):
    """CPU-only installs must not contain any cuda-toolkit pin."""
    src, dest = env_yaml
    monkeypatch.setattr("ClearMap.Utils.install_utils.platform.system", lambda: "Linux")
    patch_env(
        cfg_path=src,
        dest_path=dest,
        use_cuda_torch=False,
        pip_mode=True,
        use_spyder=False,
        tmp_dir="/tmp",
    )
    assert _read_toolkit_pin(dest) is None

# ── packages pinned for Linux only ────────────────────────────────────────
#
# Qt 6.11.1 crashes on Linux only (fontconfig), so ClearMap3.yml pins qt6-main: the installer keeps that pin on Linux
# and lets the version go elsewhere, where it only narrows the solve (no PyQt6 for Qt 6.11.3 on Intel macOS and
# Windows). The package itself stays: pyqt6 requires it anyway.

QT_PACKAGE_NAME = 'qt6-main'
QT_PIN = f'{QT_PACKAGE_NAME}>=6.11.3'


@pytest.fixture
def env_yaml_with_qt_pin(tmp_path):
    cfg = {'name': 'test_env', 'channels': ['conda-forge'],
           'dependencies': ['python=3.12', 'pyqt6', QT_PIN, 'numpy']}
    src = tmp_path / 'env_src.yml'
    dest = tmp_path / 'env_dest.yml'
    src.write_text(yaml.dump(cfg))
    return src, dest


def _conda_dependencies(dest_path):
    with open(dest_path) as fh:
        return [dep for dep in yaml.safe_load(fh)['dependencies'] if isinstance(dep, str)]


def _patch_env_on(system_name, src, dest, monkeypatch):
    def system():
        return system_name

    def processor():
        return 'arm'

    monkeypatch.setattr('ClearMap.Utils.install_utils.platform.system', system)
    monkeypatch.setattr('ClearMap.Utils.install_utils.platform.processor', processor)
    patch_env(cfg_path=src, dest_path=dest, use_cuda_torch=False, pip_mode=True, use_spyder=False, tmp_dir='/tmp')


def test_the_qt_pin_against_the_linux_font_crash_is_kept_on_linux(env_yaml_with_qt_pin, monkeypatch):
    src, dest = env_yaml_with_qt_pin

    _patch_env_on('Linux', src, dest, monkeypatch)

    assert QT_PIN in _conda_dependencies(dest)


@pytest.mark.parametrize('system_name', ['Darwin', 'Windows'])
def test_the_qt_pin_against_the_linux_font_crash_goes_on_other_systems_but_the_package_stays(
        env_yaml_with_qt_pin, system_name, monkeypatch):
    src, dest = env_yaml_with_qt_pin

    _patch_env_on(system_name, src, dest, monkeypatch)

    dependencies = _conda_dependencies(dest)
    assert QT_PACKAGE_NAME in dependencies  # Listed without a version: only the pin is gone
    assert not [dep for dep in dependencies if dep.startswith(QT_PACKAGE_NAME) and dep != QT_PACKAGE_NAME]
    assert 'pyqt6' in dependencies


@pytest.mark.parametrize('system_name', ['Linux', 'Darwin', 'Windows'])
def test_qt6_main_is_not_installed_when_the_environment_file_selects_pyqt5(tmp_path, system_name, monkeypatch):
    cfg = {'name': 'test_env', 'channels': ['conda-forge'],
           'dependencies': ['python=3.12', 'pyqt=5', QT_PIN, 'numpy']}
    src = tmp_path / 'env_src.yml'
    dest = tmp_path / 'env_dest.yml'
    src.write_text(yaml.dump(cfg))

    _patch_env_on(system_name, src, dest, monkeypatch)

    dependencies = _conda_dependencies(dest)
    assert not [dep for dep in dependencies if dep.startswith(QT_PACKAGE_NAME)]
    assert 'pyqt=5' in dependencies
