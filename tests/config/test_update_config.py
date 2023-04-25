import os
import copy

import pytest

from ClearMap.config.config_loader import get_configobj_cfg
from ClearMap.config.update_config import deep_merge_dicts, remove_extra_keys


@pytest.fixture
def old_alignment():
    cfg = get_configobj_cfg(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'pre_2.1.0_alignment_params.cfg'))
    return dict(cfg)


@pytest.fixture
def new_alignment():
    cfg = get_configobj_cfg(os.path.join(os.path.dirname(os.path.abspath(__file__)), '2.1.0_alignment_params.cfg'))
    return dict(cfg)


# @pytest.parametrize()
def test_deep_merge_dicts(old_alignment, new_alignment):
    assert old_alignment != new_alignment
    merged = remove_extra_keys(deep_merge_dicts(copy.deepcopy(old_alignment), new_alignment), new_alignment)
    loaded_merged = dict(get_configobj_cfg(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'pre_2.1.0_fixed_alignment_params.cfg')))
    assert merged == loaded_merged
