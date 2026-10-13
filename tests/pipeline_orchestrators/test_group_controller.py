"""
The group controller, on a real group folder (the config of the groups) and a real sample.
"""
import pytest

from ClearMap.Utils.event_bus import EventBus
from ClearMap.config.config_coordinator import make_cfg_coordinator_factory
from ClearMap.pipeline_orchestrators.experiment_controller import (AnalysisGroupController, ExperimentController,
                                                                   DEFAULT_GROUP_PIPELINE)


@pytest.fixture()
def fos_sample(experiment, channel_editor, tmp_path):
    """A sample with a nuclei channel and its density map"""
    channel_editor.add('fos', 'nuclei', with_path=True)
    (tmp_path / 'fos_density_counts.tif').write_bytes(b'')
    return tmp_path


def group_controller(group_dir, group_config: str) -> AnalysisGroupController:
    """The controller of the group folder whose group_analysis.yml is group_config"""
    group_dir.mkdir(exist_ok=True)
    (group_dir / 'group_analysis.yml').write_text(group_config)
    bus = EventBus()
    controller = AnalysisGroupController(make_cfg_coordinator_factory(bus), event_bus=bus,
                                         exp_controller_factory=ExperimentController)
    controller.set_group_base_dir(group_dir)
    return controller


def test_a_group_config_without_a_pipeline_uses_the_default_one(fos_sample, tmp_path_factory):
    config = f'version: 3.1\ngroups:\n  control:\n  - {fos_sample}\n'  # Written by hand: no pipeline
    controller = group_controller(tmp_path_factory.mktemp('groups'), config)

    analysis = controller.density_orchestrator

    assert analysis.pipeline == DEFAULT_GROUP_PIPELINE == 'CellMap'
    assert analysis.find_available_density_suffixes(groups=controller.groups) == ['counts']
    assert analysis.find_analysable_channels(density_suffix='counts', groups=controller.groups) == ['fos']


def test_the_pipeline_of_the_group_config_is_the_one_used(fos_sample, tmp_path_factory):
    config = f'version: 3.1\npipeline: TractMap\ngroups:\n  control:\n  - {fos_sample}\n'
    controller = group_controller(tmp_path_factory.mktemp('groups'), config)
    assert controller.density_orchestrator.pipeline == 'TractMap'
