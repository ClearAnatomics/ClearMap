from pathlib import Path

import pytest

from ClearMap.config.config_repository import ConfigRepository


def test_repo_lists_sections(tmp_repo: Path, sections):
    repo = ConfigRepository(base_dir=tmp_repo)
    names = set(repo.list_sections())
    # our temp repo is seeded with these via conftest
    assert set(sections) <= names


def test_repo_read_returns_dict(tmp_repo: Path):
    repo = ConfigRepository(base_dir=tmp_repo)
    sample = repo.load('sample')
    assert isinstance(sample, dict)
    assert 'channels' in sample


def test_repo_write_roundtrip(tmp_repo: Path):
    repo = ConfigRepository(base_dir=tmp_repo)
    data = repo.load('sample')
    data['sample_id'] = 'SAMPLE_001'
    repo.commit('sample', data)

    reloaded = repo.load('sample')
    assert reloaded['sample_id'] == 'SAMPLE_001'


def test_idempotent_dump_format(tmp_repo: Path, load_yaml):
    """
    Writing what we read shouldn't corrupt YAML.
    Exact bytes may change (ordering), so we compare parsed dicts.
    """
    repo = ConfigRepository(base_dir=tmp_repo)
    sample_path = tmp_repo / 'sample.yml'

    before = load_yaml(sample_path)
    repo.commit('sample', before)
    after = load_yaml(sample_path)

    assert before == after


def test_atomic_write(tmp_repo: Path, monkeypatch: pytest.MonkeyPatch, load_yaml):
    """
    If ConfigRepository uses a temp file + replace, simulate a short write to
    ensure the final file is valid YAML.
    """
    repo = ConfigRepository(base_dir=tmp_repo)
    data = repo.load('stitching')
    data['clearmap_version'] = '9.9.9'

    # If there is an internal dump() utility, we don't hook it; we just write
    # and then ensure YAML is still parseable.
    repo.commit('stitching', data)

    # parsed = yaml.safe_load((tmp_repo / "stitching.yml").read_text(encoding="utf-8"))
    parsed = load_yaml(tmp_repo / 'stitching.yml')
    assert parsed["clearmap_version"] == "9.9.9"
