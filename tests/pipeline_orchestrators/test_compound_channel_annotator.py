"""
The atlas annotators are those of the sample channels: the one of a compound channel (the TubeMap channels) is the one
of its representative part, its all-vessels channel (P0007: the p-value maps of the vessels looked up the annotator of
their compound channel, which has none).
"""
from types import SimpleNamespace

import pytest

from ClearMap.Utils.exceptions import ClearMapValueError, MissingRequirementException
from ClearMap.pipeline_orchestrators.generic_orchestrators import representative_channel
from ClearMap.pipeline_orchestrators.group_orchestrators import DensityGroupAnalysisOrchestrator
from ClearMap.pipeline_orchestrators.registration_orchestrator import RegistrationProcessor

VESSELS_ANNOTATOR = object()  # The annotator of the all-vessels channel


class FakeRegistrationProcessor:
    """The annotators of the sample channels, as the atlas setup makes them, with the real lookup"""
    annotator_of = RegistrationProcessor.annotator_of

    def __init__(self):
        self.annotators = {'vessels': VESSELS_ANNOTATOR}

    def setup_if_needed(self):
        pass


@pytest.fixture()
def vessels_and_arteries(experiment, channel_editor):
    channel_editor.add('arteries', 'arteries', with_path=True)
    channel_editor.add('vessels', 'vessels', with_path=True)
    return experiment.sample_manager


@pytest.mark.parametrize('channel, representative', [('arteries', 'arteries'), (('arteries', 'vessels'), 'vessels'),
                                                     (('vessels', 'arteries'), 'vessels'), (('vessels',), 'vessels'),
                                                     ('arteries-vessels', 'vessels')])  # The config and GUI form
def test_the_representative_of_a_tubemap_compound_is_its_all_vessels_part(vessels_and_arteries, channel,
                                                                          representative):
    assert representative_channel(vessels_and_arteries, 'TubeMap', channel) == representative


def test_a_tubemap_compound_without_an_all_vessels_part_has_no_representative(vessels_and_arteries):
    with pytest.raises(MissingRequirementException, match="'arteries',"):
        representative_channel(vessels_and_arteries, 'TubeMap', ('arteries',))


def test_the_compounds_of_a_pipeline_without_a_representative_data_type_have_none(vessels_and_arteries):
    with pytest.raises(ClearMapValueError, match='Colocalization'):
        representative_channel(vessels_and_arteries, 'Colocalization', ('arteries', 'vessels'))


def test_the_group_analysis_finds_the_annotator_of_the_compound_vessels_channel(vessels_and_arteries, tmp_path):
    # The user's case: plot the p-value maps of the vessels (one comparison: with the atlas)
    registration = FakeRegistrationProcessor()
    controller = SimpleNamespace(groups={'control': [str(tmp_path)]},
                                 get_sample_manager=lambda _: vessels_and_arteries,
                                 get_worker=lambda sample_dir, pipeline, channel=None, substep=None: registration)
    analysis = DensityGroupAnalysisOrchestrator(group_controller=controller, pipeline='TubeMap')

    assert analysis._get_annotator(tmp_path, channel=('arteries', 'vessels')) is VESSELS_ANNOTATOR
