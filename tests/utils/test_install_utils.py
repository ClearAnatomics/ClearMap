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