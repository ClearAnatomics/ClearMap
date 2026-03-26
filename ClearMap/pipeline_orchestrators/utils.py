from ClearMap.pipeline_orchestrators.registration_orchestrator import RegistrationProcessor
from ClearMap.pipeline_orchestrators.sample_info_management import SampleManager, build_sample_manager
from ClearMap.pipeline_orchestrators.stitching_orchestrator import StitchingProcessor


def init_sample_manager_and_processors(folder='', configs=None):
    """
    Bootstrap a SampleManager and core pipeline processors for a given experiment folder.

    This is a convenience factory for headless / script usage (no GUI, no ExperimentController).
    It builds a SampleManager from the folder's config files, then instantiates and sets up
    the StitchingProcessor and RegistrationProcessor so they are ready for immediate use.

    .. deprecated:: 3.1
        The *configs* parameter is deprecated and will be removed in a future version.
        Provide a *folder* path instead.

    Parameters
    ----------
    folder : str | Path
        Path to the experiment directory containing the sample and pipeline config files.
        Must be non-empty.
    configs : Any, optional
        Deprecated. Previously accepted pre-loaded config objects.
        Passing a non-None value raises a DeprecationWarning.

    Returns
    -------
    dict[str, SampleManager | StitchingProcessor | RegistrationProcessor]
        A dictionary with the following keys:

        - ``'sample_manager'`` : SampleManager
            Fully initialised sample manager with workspace and channel metadata.
        - ``'stitcher'`` : StitchingProcessor
            Stitching processor, set up and ready to run stitching steps.
        - ``'registration_processor'`` : RegistrationProcessor
            Registration processor, set up and ready to run alignment steps.

    Raises
    ------
    ValueError
        If *folder* is empty or falsy.
    DeprecationWarning
        If *configs* is not None (legacy call-site).
    """
    if not folder:
        if configs is not None:
            raise DeprecationWarning('Config-based initialization is deprecated, please provide a folder')
        raise ValueError('A folder must be provided to initialize the sample manager and processors')

    sample_manager = build_sample_manager(src_dir=folder)
    cfg_coordinator = sample_manager.cfg_coordinator

    stitcher = StitchingProcessor(sample_manager, cfg_coordinator)
    stitcher.setup()

    registration_processor = RegistrationProcessor(sample_manager, cfg_coordinator)
    registration_processor.setup()

    return {
        'sample_manager': sample_manager,
        'stitcher': stitcher,
        'registration_processor': registration_processor
    }
