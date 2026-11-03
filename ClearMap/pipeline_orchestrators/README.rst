pipeline_orchestrators
======================

Pipeline worker classes that implement each ClearMap processing pipeline.
They are the backend used by the GUI tabs and can also be used independently
for scripted or headless workflows.

Workers read their parameters from YAML config files managed by
:mod:`ClearMap.config.config_coordinator` and write all outputs as
:class:`~ClearMap.IO.workspace_asset.Asset` objects through a
:class:`~ClearMap.IO.workspace2.Workspace2`.


Composition
===========

generic_orchestrators
    Abstract base classes shared by all workers:
    ``OrchestratorBase`` (config access, asset retrieval, event bus),
    ``PipelineOrchestrator`` (progress watcher, stop/cancel),
    ``ChannelPipelineOrchestrator`` (per-channel config scoping),
    ``CompoundChannelPipelineOrchestrator`` (multi-channel pipelines),
    ``ProcessorSteps`` (disk-level step tracking),
    and ``GroupOrchestratorBase`` (multi-experiment analyses).

sample_info_management
    ``SampleManager`` — the root object for a single experiment.
    Owns the sample config, keeps the workspace in sync with channel
    definitions, and provides channel queries used by all other workers.
    Also provides ``build_sample_manager()`` as the recommended entry
    point for scripted use.

experiment_controller
    ``ExperimentController`` — wires together the config coordinator,
    sample manager, and all pipeline workers for one experiment.
    Manages worker lifecycle (lazy creation, reconciliation on channel
    changes) and is the single entry point for UI patches.
    ``AnalysisGroupController`` — manages multiple
    ``ExperimentController`` instances for group-level analyses.

stitching_orchestrator
    ``StitchingProcessor`` — rigid and wobbly stitching via
    :mod:`~ClearMap.Alignment.Stitching`, tile conversion, and layout
    management.

registration_orchestrator
    ``RegistrationProcessor`` — resampling, Elastix-based atlas
    registration, and atlas annotation of sample data.
    Also exposes ``RegistrationStatus`` and annotator helpers used by
    downstream workers that need atlas-space coordinates.

cell_map
    ``CellDetector`` — cell detection, filtering, atlas alignment,
    voxelization, and density-map export for fluorescence channels.

tube_map
    ``BinaryVesselProcessor`` — multi-step vessel binarization
    (threshold, smooth, fill, deep fill) with configurable step order.
    ``VesselGraphProcessor`` — skeletonization, graph construction,
    cleaning, reduction, atlas registration, and artery/vein tracing.

tract_map
    ``TractMapProcessor`` — binarization, coordinate extraction,
    atlas-space transform, labeling, and voxelization for
    myelinated-tract channels.

colocalization
    ``ColocalizationProcessor`` — nearest-neighbor colocalization
    analysis between two fluorescence channels.

group_orchestrators
    ``DensityGroupAnalysisOrchestrator`` — group-level density map
    statistics (p-values, effect sizes) across multiple experiments.

batch_process
    ``BatchProcessor`` — runs a full single-sample pipeline
    (stitch → register → detect/analyse) sequentially across a list
    of experiment folders.  Useful as a scripting template for
    unattended batch runs.

utils
    ``init_sample_manager_and_processors()`` — convenience factory used
    by the new-API scripts to initialise all standard workers for an
    experiment directory in one call.