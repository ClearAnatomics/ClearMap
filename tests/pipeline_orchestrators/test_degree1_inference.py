"""
The worker that classifies the degree 1 vertices of the vascular graph (tip cells), headless.

The model, the GPU and the graph and image assets are stand-ins (they need torch, an NVIDIA GPU and graph_tool):
what is tested is what the worker asks of them, what it gives them (from the config) and what it does with
the result. The classification itself is tested in tests/image_processing/machine_learning.
"""
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import ClearMap.pipeline_orchestrators.degree1_inference_processor as processor_module
from ClearMap.Analysis.graphs.graph_gt import Graph
from ClearMap.Analysis.graphs.tip_cells import TIP_CELL_SCORE_PROPERTY_NAME, TIP_CELL_PREDICTION_PROPERTY_NAME
from ClearMap.IO.assets_constants import TIP_CELLS_SCORED_SUB_TYPE
from ClearMap.Utils.exceptions import (CudaNotAvailableError, MissingRequirementException, ClearMapVRamException)

GRAPH_STEP = 'annotated'
COORDINATES = np.array([[10, 10, 10], [20, 20, 20], [30, 30, 30]])
DEGREES = np.array([1, 3, 1])


class FakeGraph:
    """What the worker uses of ClearMap.Analysis.graphs.graph_gt.Graph (which needs graph_tool)"""
    def __init__(self):
        self.properties = {'coordinates': COORDINATES}

    def vertex_property(self, name):
        return self.properties[name]

    def vertex_indices(self):
        return np.arange(len(COORDINATES))

    def vertex_degrees(self):
        return DEGREES

    def vertex_properties_table(self, *args, **kwargs):  # The real one, on this stand-in
        return Graph.vertex_properties_table(self, *args, **kwargs)

    def define_vertex_property(self, name, source):
        self.properties[name] = np.asarray(source)


class FakeAsset:
    def __init__(self, path: Path, *, exists: bool = True, content=None):
        self.path = path
        self.exists = exists
        self.content = content
        self.written = None

    def read(self):
        return self.content

    def open_ro(self):
        return self.content

    def write(self, data):
        self.written = data


class FakeSteps:
    def __init__(self, last_step):
        self.last_step = last_step


class FakeInference:
    """The module of the model: records how it is called, returns the scores it is given"""
    MODEL_WEIGHTS_URL = ''

    def __init__(self, weights: Path, scores: pd.DataFrame):
        self.weights = weights
        self.scores = scores
        self.calls = []
        self.batches = 0  # Reported to the progress callbacks

    def get_default_model_path(self) -> Path:
        return self.weights

    def run_inference(self, *, image, vertices_df, model_weights, **kwargs):
        self.calls.append(SimpleNamespace(image=image, vertices_df=vertices_df, model_weights=model_weights, **kwargs))
        kwargs['on_start'](self.batches)
        for _ in range(self.batches):
            kwargs['on_batch_done']()
        return self.scores


class VRamStarvedInference(FakeInference):
    def run_inference(self, *, image, vertices_df, model_weights, **kwargs):
        raise ClearMapVRamException('Insufficient VRAM')


class RecordingWatcher:
    def __init__(self):
        self.substeps, self.increments = [], []

    def prepare_for_substep(self, counter_size, pattern, title):
        self.substeps.append((counter_size, pattern, title))

    def increment(self, val):
        self.increments.append(val)


def _scores(*scores: float) -> pd.DataFrame:
    """The scores of the degree 1 vertices (0 and 2)"""
    return pd.DataFrame({'vertex_id': [0, 2], TIP_CELL_SCORE_PROPERTY_NAME: list(scores)})


@pytest.fixture()
def worker(experiment, channel_editor, monkeypatch):
    channel_editor.add('vessels', 'vessels', with_path=True)
    channel_editor.add('arteries', 'arteries', with_path=True)
    worker = experiment.get_worker('vasculature', substep='degree1')
    worker.setup_if_needed()
    monkeypatch.setattr(processor_module, 'cuda_unavailable_reason', _cuda_is_there)
    monkeypatch.setattr(worker, 'steps', FakeSteps(GRAPH_STEP))
    return worker


def _cuda_is_there():
    return None


@pytest.fixture()
def assets(worker, tmp_path, monkeypatch) -> dict:
    """The assets the worker asks the workspace for, by (type, sub type)"""
    assets = {('graph', GRAPH_STEP): FakeAsset(tmp_path / 'graph_annotated.gt', content=FakeGraph()),
              ('stitched', None): FakeAsset(tmp_path / 'stitched.npy', content=np.zeros((2, 2, 2))),
              ('graph', TIP_CELLS_SCORED_SUB_TYPE): FakeAsset(tmp_path / 'graph_tip_cells_scored.gt', exists=False)}

    def get_fake_asset(asset_type, channel=None, asset_sub_type=None, **kwargs):
        requests.append((asset_type, channel, asset_sub_type))
        return assets[(asset_type, asset_sub_type)]

    requests = []
    assets['requests'] = requests
    monkeypatch.setattr(worker, 'get', get_fake_asset)
    return assets


@pytest.fixture()
def inference(tmp_path, monkeypatch) -> FakeInference:
    weights = tmp_path / 'weights.pth'
    weights.write_bytes(b'')
    fake = FakeInference(weights, _scores(0.85, 0.2))
    monkeypatch.setattr(processor_module, '_inference_utils', fake_getter(fake))
    return fake


def fake_getter(fake):
    def get_inference_utils():
        return fake
    return get_inference_utils


# ----------------------------------------------------------------------------------------------------------------
# What it does with the result
# ----------------------------------------------------------------------------------------------------------------

def test_the_scored_graph_is_a_new_asset_and_the_graph_it_comes_from_is_not_written(worker, assets, inference):
    worker.run()

    scored = assets[('graph', TIP_CELLS_SCORED_SUB_TYPE)].written
    assert scored is assets[('graph', GRAPH_STEP)].content
    assert scored.properties[TIP_CELL_SCORE_PROPERTY_NAME].tolist() == pytest.approx([0.85, -1, 0.2])
    assert assets[('graph', GRAPH_STEP)].written is None


def test_the_threshold_comes_from_the_config(worker, assets, inference, experiment):
    worker.run()
    default_predictions = assets[('graph', TIP_CELLS_SCORED_SUB_TYPE)].written.properties[TIP_CELL_PREDICTION_PROPERTY_NAME]
    assert default_predictions.tolist() == [1, 0, 0]  # 0.85 is above the default of 0.8

    experiment.apply_ui_patch({'vasculature': {'degree1_inference': {'threshold': 0.9}}})
    assets[('graph', GRAPH_STEP)].content = FakeGraph()
    worker.run()

    predictions = assets[('graph', TIP_CELLS_SCORED_SUB_TYPE)].written.properties[TIP_CELL_PREDICTION_PROPERTY_NAME]
    assert predictions.tolist() == [0, 0, 0]


# ----------------------------------------------------------------------------------------------------------------
# What it gives the model
# ----------------------------------------------------------------------------------------------------------------

def test_the_graph_is_the_one_of_the_compound_channel_and_the_image_the_one_of_the_vessels(worker, assets, inference):
    worker.run()

    assert ('graph', ('arteries', 'vessels'), GRAPH_STEP) in assets['requests']
    assert ('stitched', 'vessels', None) in assets['requests']  # Not the arteries: the one all the parts share


def test_the_model_gets_the_image_the_vertices_the_weights_and_the_performance_config(worker, assets, inference,
                                                                                      experiment):
    experiment.apply_ui_patch({'vasculature': {'performance': {'degree1_inference': {'batch_size': 4,
                                                                                    'n_workers': 2}}}})
    worker.run()

    call, = inference.calls
    assert call.image is assets[('stitched', None)].content
    assert call.vertices_df['vertex_id'].tolist() == [0, 1, 2]
    assert call.vertices_df['degree'].tolist() == DEGREES.tolist()
    assert call.model_weights == inference.weights
    assert (call.batch_size, call.n_workers) == (4, 2)


def test_progress_is_forwarded_to_the_watcher(worker, assets, inference):
    watcher = RecordingWatcher()
    worker.set_progress_watcher(watcher)
    inference.batches = 3

    worker.run()

    assert watcher.substeps == [(3, None, processor_module.PROGRESS_TITLE)]
    assert watcher.increments == [1, 1, 1]


# ----------------------------------------------------------------------------------------------------------------
# What it refuses
# ----------------------------------------------------------------------------------------------------------------

def test_without_a_cuda_gpu_it_raises_before_anything_else(worker, assets, monkeypatch):
    monkeypatch.setattr(processor_module, 'cuda_unavailable_reason', _no_gpu)
    monkeypatch.setattr(processor_module, '_inference_utils', _must_not_be_called)

    with pytest.raises(CudaNotAvailableError, match='no GPU here'):
        worker.run()


def _no_gpu():
    return 'no GPU here'


def _must_not_be_called():
    raise AssertionError('The model must not be loaded without a GPU')


def test_without_a_graph_it_raises(worker, assets, inference, monkeypatch):
    monkeypatch.setattr(worker, 'steps', FakeSteps(None))

    with pytest.raises(MissingRequirementException, match='build it first'):
        worker.run()


def test_without_the_graph_file_it_raises(worker, assets, inference):
    assets[('graph', GRAPH_STEP)].exists = False

    with pytest.raises(MissingRequirementException, match='graph_annotated.gt'):
        worker.run()


def test_without_the_stitched_image_it_raises(worker, assets, inference):
    assets[('stitched', None)].exists = False

    with pytest.raises(MissingRequirementException, match='stitched image'):
        worker.run()


def test_without_the_weights_it_raises_and_says_where_they_should_be(worker, assets, inference):
    inference.weights.unlink()

    with pytest.raises(MissingRequirementException, match='weights.pth'):
        worker.run()


def test_insufficient_vram_is_the_caller_s_to_handle_and_nothing_is_written(worker, assets, tmp_path, monkeypatch):
    weights = tmp_path / 'weights.pth'
    weights.write_bytes(b'')
    monkeypatch.setattr(processor_module, '_inference_utils',
                        fake_getter(VRamStarvedInference(weights, _scores(0.5, 0.5))))

    with pytest.raises(ClearMapVRamException):
        worker.run()

    assert assets[('graph', TIP_CELLS_SCORED_SUB_TYPE)].written is None
