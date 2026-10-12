"""
The classification of the degree 1 vertices from patches of the image, with a model that is a stand-in
(its weights are not needed): what is tested is what is given to the model and what comes back.
Needs torch (it is skipped otherwise); runs on the CPU.
"""
import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip('torch')
pytest.importorskip('torchvision')

import ClearMap.ImageProcessing.machine_learning.vertices_classification.degree1_inference_utils as inference_utils
from ClearMap.Analysis.graphs.tip_cells import TIP_CELL_SCORE_PROPERTY_NAME
from ClearMap.Utils.exceptions import ClearMapVRamException

IMAGE_SHAPE = (80, 80, 80)
BRIGHT_VERTEX, DARK_VERTEX, DARK_VERTEX_2 = 1, 0, 4
VERTICES = pd.DataFrame({
    'vertex_id': [0, 1, 2, 3, 4],
    'x': [40, 60, 40, 5, 20],  # Vertex 3 is too close to the border for a patch
    'y': [40, 60, 40, 40, 20],
    'z': [40, 60, 40, 40, 20],
    'degree': [1, 1, 3, 1, 1]})  # Vertex 2 is not a degree 1
BATCH_SIZE = 2


class _BrightnessModel(torch.nn.Module):
    """The score is high for a bright patch"""
    def forward(self, patches):
        mean = patches.mean(dim=(1, 2, 3, 4))
        return torch.stack([-mean, mean], dim=1)


class _OutOfMemoryModel(torch.nn.Module):
    def forward(self, patches):
        raise torch.cuda.OutOfMemoryError('CUDA out of memory')


def _load_brightness_model(weights_path):
    return _BrightnessModel()


def _load_out_of_memory_model(weights_path):
    return _OutOfMemoryModel()


@pytest.fixture(autouse=True)
def cpu_and_stand_in_model(monkeypatch):
    monkeypatch.setattr(inference_utils, 'DEVICE', 'cpu')
    monkeypatch.setattr(inference_utils, 'load_model', _load_brightness_model)


@pytest.fixture()
def image():
    image = np.zeros(IMAGE_SHAPE, dtype=np.float32)
    half = np.array(inference_utils.PATCH_SHAPE) // 2
    low, high = 60 - half, 60 + half
    image[low[0]:high[0], low[1]:high[1], low[2]:high[2]] = 5  # Around the bright vertex
    return image


def _run(image, **kwargs):
    return inference_utils.run_inference(image=image, vertices_df=VERTICES, model_weights='unused_weights',
                                         batch_size=BATCH_SIZE, **kwargs)


def test_only_the_degree_1_vertices_with_a_patch_in_the_image_are_classified(image):
    scores = _run(image).set_index('vertex_id')[TIP_CELL_SCORE_PROPERTY_NAME]

    assert scores.index.tolist() == [DARK_VERTEX, BRIGHT_VERTEX, DARK_VERTEX_2]


def test_the_score_follows_the_patch_of_its_vertex(image):
    scores = _run(image).set_index('vertex_id')[TIP_CELL_SCORE_PROPERTY_NAME]

    assert scores[BRIGHT_VERTEX] > 0.99
    assert scores[DARK_VERTEX] == pytest.approx(0.5)


def test_progress_is_reported_per_batch(image):
    calls = []

    class Recorder:
        @staticmethod
        def on_start(n_batches):
            calls.append(('start', n_batches))

        @staticmethod
        def on_batch_done():
            calls.append(('batch',))

    _run(image, on_start=Recorder.on_start, on_batch_done=Recorder.on_batch_done)

    assert calls == [('start', 2), ('batch',), ('batch',)]  # 3 vertices by 2


def test_out_of_memory_becomes_a_vram_exception_naming_the_batch_size(image, monkeypatch):
    monkeypatch.setattr(inference_utils, 'load_model', _load_out_of_memory_model)

    with pytest.raises(ClearMapVRamException, match=f'batches of {BATCH_SIZE} patches'):
        _run(image)
