import os
import copy

import pytest

from ClearMap.config.config_loader import get_configobj_cfg
from ClearMap.config.update_config import deep_merge_dicts, remove_extra_keys


def get_cfg(cfg_name):
    cfg = get_configobj_cfg(os.path.join(os.path.dirname(os.path.abspath(__file__)), cfg_name))
    return dict(cfg)


@pytest.fixture
def old_alignment():
    return get_cfg('pre_2.1.0_alignment_params.cfg')


@pytest.fixture
def new_alignment():
    return get_cfg('2.1.0_alignment_params.cfg')


@pytest.fixture
def corrected_alignement():
    return get_cfg('pre_2.1.0_fixed_alignment_params.cfg')


def test_deep_merge_dicts(old_alignment, new_alignment, corrected_alignement):
    assert old_alignment != new_alignment
    merged = remove_extra_keys(deep_merge_dicts(copy.deepcopy(old_alignment), new_alignment), new_alignment)
    assert merged == corrected_alignement
