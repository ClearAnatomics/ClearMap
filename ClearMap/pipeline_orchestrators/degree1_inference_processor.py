"""
degree1_inference_processor
===========================

Classification of the degree 1 vertices of the vascular graph: tip cells (vessels that grow or retract)
or ends of a bad tracing (interrupted vessels, to be reconnected later).

:class:`Degree1InferenceProcessor` is the third worker of the TubeMap pipeline, after the
:class:`~ClearMap.pipeline_orchestrators.tube_map.BinaryVesselProcessor` and
:class:`~ClearMap.pipeline_orchestrators.tube_map.VesselGraphProcessor`. It takes the most advanced
graph that exists, extracts a patch of the vessels image around each of its degree 1 vertices, and has a
model score them. The graph with the score and prediction of its vertices
(see :mod:`ClearMap.Analysis.graphs.tip_cells`) is a new asset (:data:`TIP_CELLS_SCORED_SUB_TYPE`):
the graphs it is computed from are left as they are.

The model runs on an NVIDIA GPU: check :func:`~ClearMap.Utils.utilities.cuda_unavailable_reason`
before offering the step.

Typical usage::

    tip_cells = exp_ctrl.get_worker('vasculature', substep='degree1')
    tip_cells.run()
"""
from __future__ import annotations

from typing import Optional

from ClearMap.Analysis.graphs.tip_cells import add_tip_cell_properties
from ClearMap.IO.assets_constants import TIP_CELLS_SCORED_SUB_TYPE
from ClearMap.IO.workspace2 import Workspace2
from ClearMap.Utils.events import WorkspaceChannelsUpdated
from ClearMap.Utils.exceptions import CudaNotAvailableError, MissingRequirementException
from ClearMap.Utils.utilities import cuda_unavailable_reason
from ClearMap.config.config_coordinator import ConfigCoordinator

from .sample_info_management import SampleManager
from .tube_map import TubeMapOrchestrator, VesselGraphProcessorSteps

PROGRESS_TITLE = 'Classifying the degree 1 vertices'


def _inference_utils():
    """The module of the model (it imports torch, so only when it is needed)"""
    from ClearMap.ImageProcessing.machine_learning.vertices_classification import degree1_inference_utils
    return degree1_inference_utils


class Degree1InferenceProcessor(TubeMapOrchestrator):
    """
    Scores the degree 1 vertices of the vascular graph of the TubeMap compound channel (see the module).

    Attributes
    ----------
    steps : VesselGraphProcessorSteps
        The graphs of the pipeline, to find the most advanced one that exists.
    """
    def __init__(self, sample_manager: Optional[SampleManager] = None,
                 config_coordinator: Optional[ConfigCoordinator] = None):
        super().__init__(config_coordinator)
        self.sample_manager: Optional[SampleManager] = sample_manager
        self.workspace: Optional[Workspace2] = None  # Set in setup
        self.steps: VesselGraphProcessorSteps = VesselGraphProcessorSteps(self.workspace)

        self.setup(sample_manager)
        self.subscribe(WorkspaceChannelsUpdated, self._on_workspace_channels_updated)

    def _on_workspace_channels_updated(self, evt: WorkspaceChannelsUpdated) -> None:
        if self.sample_ready:
            self.setup()

    def setup(self, sample_manager: Optional[SampleManager] = None) -> None:
        self.sample_manager = sample_manager if sample_manager is not None else self.sample_manager
        self._setup_done = False
        if self.sample_ready:
            self.workspace = self.sample_manager.workspace
            self._follow_sample_compound_channel()
            self.steps.workspace = self.workspace
            self.steps.channel = self.compound_channel
            self._setup_done = True

    # ################################ INPUTS ####################################

    @staticmethod
    def ensure_cuda_available() -> None:
        """
        Raises
        ------
        CudaNotAvailableError
            If the model cannot run on this machine (see cuda_unavailable_reason).
        """
        reason = cuda_unavailable_reason()
        if reason is not None:
            raise CudaNotAvailableError(f'The degree 1 vertices cannot be classified: {reason}.')

    @property
    def graph_step(self) -> str:
        """The most advanced step of the graph that exists: the one that is classified"""
        step = self.steps.last_step
        if step is None:
            raise MissingRequirementException(f'No graph found for {self.compound_channel}: build it first '
                                              f'to classify its degree 1 vertices.')
        return step

    def _get_existing_asset(self, asset_type: str, channel, what: str, **kwargs):
        asset = self.get(asset_type, channel=channel, **kwargs)
        if not asset.exists:
            raise MissingRequirementException(f'The {what} is missing: {asset.path}')
        return asset

    # ############################### PROCESSING #################################

    def run(self) -> None:
        """
        Score the degree 1 vertices of the most advanced graph and save the result as a new graph.

        Raises
        ------
        CudaNotAvailableError
            If there is no usable GPU.
        MissingRequirementException
            If the graph, the image of the vessels or the weights of the model are missing.
        ClearMapVRamException
            If the GPU does not have enough memory for the batch size
            (``performance.degree1_inference.batch_size``).
        """
        self.ensure_cuda_available()
        inference = _inference_utils()

        weights = inference.get_default_model_path()
        if not weights.exists():
            where = f' Download them from {inference.MODEL_WEIGHTS_URL}.' if inference.MODEL_WEIGHTS_URL else ''
            raise MissingRequirementException(f'The weights of the model are missing: {weights}.{where}')

        step = self.graph_step
        graph_asset = self._get_existing_asset('graph', self.compound_channel, f'graph ({step})', asset_sub_type=step)
        image_asset = self._get_existing_asset('stitched', self.representative_channel, 'stitched image of the vessels')

        graph = graph_asset.read()
        coords_table = graph.vertex_properties_table(['coordinates'], with_degrees=True)
        scores = inference.run_inference(image=image_asset.open_ro(), vertices_df=coords_table, model_weights=weights,
                                         batch_size=self.perf_config['batch_size'],
                                         n_workers=self.perf_config['n_workers'],
                                         on_start=self._prepare_progress, on_batch_done=self._on_batch_done)

        add_tip_cell_properties(graph, scores, threshold=self.config['degree1_inference']['threshold'])
        self.get('graph', channel=self.compound_channel, asset_sub_type=TIP_CELLS_SCORED_SUB_TYPE).write(graph)

    @property
    def perf_config(self) -> dict:
        return self.config['performance']['degree1_inference']

    def _prepare_progress(self, n_batches: int) -> None:
        self.prepare_watcher_for_substep(n_batches, None, PROGRESS_TITLE)

    def _on_batch_done(self) -> None:
        self.update_watcher_progress(1)
