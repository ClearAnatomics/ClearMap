import pytest


@pytest.fixture()
def default_cfg():
    return {'cell_detection': {'min_radius': 2}}


@pytest.mark.parametrize('min_radius', [5, 6, 3])
def test_patch_cfg(default_cfg, min_radius):
    from ClearMap.config.config_handler import patch_cfg
    cfg = default_cfg.copy()
    cfg['cell_detection']['min_radius'] = min_radius
    patch_cfg(cfg, default_cfg)
    assert cfg['cell_detection']['min_radius'] == min_radius
