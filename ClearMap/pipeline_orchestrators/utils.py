from ClearMap.pipeline_orchestrators.registration_orchestrator import RegistrationProcessor
from ClearMap.pipeline_orchestrators.sample_info_management import SampleManager
from ClearMap.pipeline_orchestrators.stitching_orchestrator import StitchingProcessor


def init_sample_manager_and_processors(folder='', configs=None):
    sample_manager = SampleManager()
    if folder:
        sample_manager.setup(src_dir=folder)
    elif configs:
        raise NotImplementedError('Config-based initialization has been removed, please provide a folder')

    stitcher = StitchingProcessor(sample_manager)  # FIXME: pass cfg_coordinator
    stitcher.setup()

    registration_processor = RegistrationProcessor(sample_manager)  # FIXME: pass cfg_coordinator
    registration_processor.setup()

    return {'sample_manager': sample_manager, 'stitcher': stitcher, 'registration_processor': registration_processor}
