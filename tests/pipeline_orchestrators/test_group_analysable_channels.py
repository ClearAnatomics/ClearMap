"""
The channels of the group analysis are those which have a density map for the suffix, under the name the
density maps are saved with: for the vessels, their compound channel (P0001: the density map looked for was the
one of a single vessels channel, which does not exist when there are several).
"""
from types import SimpleNamespace

import pytest

from ClearMap.Utils.exceptions import MissingRequirementException
from ClearMap.config.compound_keys import channel_file_name_token
from ClearMap.pipeline_orchestrators.generic_orchestrators import register_compound_channel
from ClearMap.pipeline_orchestrators.group_orchestrators import DensityGroupAnalysisOrchestrator

SUFFIX = 'branches'


def group_analysis(experiment, tmp_path, pipeline='TubeMap'):
    """The group analysis of one group of one sample (this experiment)"""
    controller = SimpleNamespace(groups={'control': [str(tmp_path)]},
                                 get_sample_manager=lambda _: experiment.sample_manager)
    return DensityGroupAnalysisOrchestrator(group_controller=controller, pipeline=pipeline)


def save_density(experiment, parts):
    """The voxelization of the vessels: the density map of their compound channel"""
    sample_manager = experiment.sample_manager
    channel = register_compound_channel(sample_manager, 'TubeMap', parts)
    path = sample_manager.get('density', channel=channel, suffix=SUFFIX).path
    path.write_bytes(b'')
    return path


@pytest.fixture()
def plotted(monkeypatch):
    """The paths given to the viewer"""
    import ClearMap.Visualization.Qt.Plot3d as plot_3d
    paths = []
    monkeypatch.setattr(plot_3d, 'plot', lambda sources, **kwargs: paths.append(list(sources)) or [])
    monkeypatch.setattr('ClearMap.Visualization.Qt.utils.link_dataviewers_cursors', lambda dvs: None)
    return paths


@pytest.fixture()
def two_vessel_channels(experiment, channel_editor):
    channel_editor.add('vessels', 'vessels', with_path=True)
    channel_editor.add('arteries', 'arteries', with_path=True)


@pytest.fixture()
def one_vessel_channel(experiment, channel_editor):
    channel_editor.add('vessels', 'vessels', with_path=True)


def test_the_vessels_are_one_channel_the_compound_of_their_channels(experiment, tmp_path, two_vessel_channels):
    save_density(experiment, ['vessels', 'arteries'])
    analysis = group_analysis(experiment, tmp_path)

    channels = analysis.find_analysable_channels(density_suffix=SUFFIX)

    assert channels == [('arteries', 'vessels')]  # Not vessels and arteries
    assert [channel_file_name_token(channel) for channel in channels] == ['arteries-vessels']


def test_a_channel_without_a_density_map_for_the_suffix_is_not_offered(experiment, tmp_path, two_vessel_channels):
    analysis = group_analysis(experiment, tmp_path)
    assert analysis.find_analysable_channels(density_suffix=SUFFIX) == []


def test_the_density_map_plotted_is_the_one_of_the_compound_channel(experiment, tmp_path, two_vessel_channels,
                                                                     plotted):
    density = save_density(experiment, ['vessels', 'arteries'])
    analysis = group_analysis(experiment, tmp_path)
    channel, = analysis.find_analysable_channels(density_suffix=SUFFIX)

    analysis.plot_density_maps([str(tmp_path)], channel=channel, density_suffix=SUFFIX)

    assert plotted == [[density]]
    assert density.name.endswith('arteries-vessels_density_branches.tif')


def test_a_single_vessels_channel_is_plotted_too(experiment, tmp_path, one_vessel_channel, plotted):
    density = save_density(experiment, ['vessels'])
    analysis = group_analysis(experiment, tmp_path)
    channel, = analysis.find_analysable_channels(density_suffix=SUFFIX)

    analysis.plot_density_maps([str(tmp_path)], channel=channel, density_suffix=SUFFIX)
    analysis.plot_density_maps([str(tmp_path)], channel=channel_file_name_token(channel), density_suffix=SUFFIX)

    assert plotted == [[density], [density]]  # The 1-tuple and its name share their file


def test_a_missing_density_map_names_the_file(experiment, tmp_path, two_vessel_channels):
    analysis = group_analysis(experiment, tmp_path)
    channel = register_compound_channel(experiment.sample_manager, 'TubeMap', ['vessels', 'arteries'])
    with pytest.raises(MissingRequirementException, match='arteries-vessels_density_branches'):
        analysis.plot_density_maps([str(tmp_path)], channel=channel, density_suffix=SUFFIX)


# ----------------------------------------------------------------------------------------------------------------
# The groups being edited in the tab are not in the config (where the controller reads them) until applied
# ----------------------------------------------------------------------------------------------------------------

@pytest.fixture()
def fos_sample(experiment, channel_editor, tmp_path):
    channel_editor.add('fos', 'nuclei', with_path=True)
    (tmp_path / 'fos_density_counts.tif').write_bytes(b'')
    controller = SimpleNamespace(groups={}, get_sample_manager=lambda _: experiment.sample_manager)  # Not applied yet
    return DensityGroupAnalysisOrchestrator(group_controller=controller, pipeline='CellMap'), {'control': [str(tmp_path)]}


def test_the_suffixes_and_channels_are_those_of_the_groups_of_the_tab(fos_sample):
    analysis, tab_groups = fos_sample

    assert analysis.find_available_density_suffixes() == []  # The groups of the controller
    assert analysis.find_available_density_suffixes(groups=tab_groups) == ['counts']
    assert analysis.find_analysable_channels(density_suffix=None, groups=tab_groups) == []  # No density without suffix
    assert analysis.find_analysable_channels(density_suffix='counts', groups=tab_groups) == ['fos']


def test_empty_groups_have_no_suffixes_nor_channels(fos_sample):
    analysis, _ = fos_sample
    for groups in ({}, {'control': []}):
        assert analysis.find_available_density_suffixes(groups=groups) == []
        assert analysis.find_analysable_channels(density_suffix='counts', groups=groups) == []
