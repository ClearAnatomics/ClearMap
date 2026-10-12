"""
The contract every orchestrator must respect (see the docstring of
:mod:`ClearMap.pipeline_orchestrators.generic_orchestrators`), checked on all of them, with the channels of the
sample complete or not:

- they can be built and set up (``setup_if_needed()``, without argument) whatever the state of the channels:
  an incomplete channel (no path yet) is hydrated, not processed;
- ``setup_complete`` and ``sample_ready`` are derived from the sample, never stored;
- an orchestrator is ready when the channels it requires are in the workspace, no matter the others.

A new orchestrator is added to :data:`WORKER_SPECS`.
"""
import importlib
import inspect
from typing import NamedTuple, Optional

import pytest

from ClearMap.pipeline_orchestrators.generic_orchestrators import PipelineOrchestrator

ORCHESTRATOR_MODULES = ('cell_map', 'colocalization', 'registration_orchestrator', 'stitching_orchestrator',
                        'tract_map', 'tube_map')

NUCLEI_CHANNELS = ('nuclei_a', 'nuclei_b')
SAMPLE_CHANNELS = {'nuclei_a': 'nuclei', 'nuclei_b': 'nuclei', 'myelin': 'myelin', 'vessels': 'vessels'}


class WorkerSpec(NamedTuple):
    """How to get a worker from the experiment controller, and whether it needs the channels to be complete"""
    pipeline: str
    substep: Optional[str]
    channel: object  # The key of the worker: None (global), a channel, or a pair
    needs_channels: bool  # Whether it requires channels of the sample (False: it works with the ones there are)


WORKER_SPECS = (
    WorkerSpec('stitching', None, None, needs_channels=False),
    WorkerSpec('registration', None, None, needs_channels=False),
    WorkerSpec('cell_map', None, 'nuclei_a', needs_channels=True),
    WorkerSpec('tract_map', None, 'myelin', needs_channels=True),
    WorkerSpec('colocalization', None, NUCLEI_CHANNELS, needs_channels=True),
    WorkerSpec('vasculature', 'binary', None, needs_channels=True),
    WorkerSpec('vasculature', 'graph', None, needs_channels=True),
)


def _spec_id(spec: WorkerSpec) -> str:
    return '-'.join(part for part in (spec.pipeline, spec.substep) if part)


@pytest.fixture(params=['incomplete', 'complete'])
def sample_state(request, channel_editor) -> str:
    """The channels of the sample have a data type; they also have a path (complete) or not (incomplete)"""
    complete = request.param == 'complete'
    for name, data_type in SAMPLE_CHANNELS.items():
        channel_editor.add(name, data_type, with_path=complete)
    return request.param


@pytest.fixture(params=WORKER_SPECS, ids=_spec_id)
def worker_spec(request) -> WorkerSpec:
    return request.param


@pytest.fixture()
def worker(experiment, sample_state, worker_spec):
    return experiment.get_worker(worker_spec.pipeline, worker_spec.channel, substep=worker_spec.substep)


def test_worker_can_be_built_and_set_up_whatever_the_channels(worker):
    worker.setup_if_needed()  # No argument, it is what the tabs call on every worker


def test_worker_is_ready_when_the_channels_it_requires_are_in_the_workspace(worker, worker_spec, sample_state):
    worker.setup_if_needed()
    expected = sample_state == 'complete' or not worker_spec.needs_channels

    assert worker.sample_ready is expected


def test_setup_complete_is_derived_not_stored(worker):
    with pytest.raises(AttributeError):
        worker.setup_complete = True


@pytest.mark.parametrize('spec', [spec for spec in WORKER_SPECS if spec.needs_channels], ids=_spec_id)
def test_setup_complete_follows_the_sample(spec, experiment, channel_editor):
    """A channel the user completes later (path set) makes the worker that requires it ready, no stale flag"""
    for name, data_type in SAMPLE_CHANNELS.items():
        channel_editor.add(name, data_type)
    worker = experiment.get_worker(spec.pipeline, spec.channel, substep=spec.substep)
    worker.setup_if_needed()
    assert not worker.sample_ready
    assert not worker.setup_complete

    for name in SAMPLE_CHANNELS:
        channel_editor.set_path(name)
    worker.setup_if_needed()

    assert worker.sample_ready
    assert worker.setup_complete


def _all_subclasses(cls):
    for subclass in cls.__subclasses__():
        yield subclass
        yield from _all_subclasses(subclass)


def _concrete_orchestrators() -> list[type]:
    for module_name in ORCHESTRATOR_MODULES:
        importlib.import_module(f'ClearMap.pipeline_orchestrators.{module_name}')
    return sorted({cls for cls in _all_subclasses(PipelineOrchestrator) if 'pipeline' in vars(cls)},
                  key=lambda cls: cls.__name__)


@pytest.mark.parametrize('orchestrator_class', _concrete_orchestrators(), ids=lambda cls: cls.__name__)
def test_concrete_orchestrator_says_which_sample_channels_it_requires(orchestrator_class):
    base_property = PipelineOrchestrator.required_sample_channels

    assert inspect.getattr_static(orchestrator_class, 'required_sample_channels') is not base_property


def test_every_concrete_orchestrator_is_in_the_contract_tests():
    in_contract = {spec.pipeline for spec in WORKER_SPECS}
    pipelines = {cls.pipeline.lower() for cls in _concrete_orchestrators()}
    # TubeMap workers are the 'vasculature' pipeline of the controller
    pipelines = {'vasculature' if pipeline == 'tubemap' else pipeline for pipeline in pipelines}

    assert pipelines <= in_contract | {'cellmap', 'tractmap'}, pipelines - in_contract
