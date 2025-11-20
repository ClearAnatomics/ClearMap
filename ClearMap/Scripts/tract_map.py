from ClearMap.config.config_coordinator import make_cfg_coordinator_factory

from ClearMap.Utils.event_bus import EventBus

from ClearMap.pipeline_orchestrators.sample_info_management import SampleManager
from ClearMap.pipeline_orchestrators.stitching_orchestrator import StitchingProcessor
from ClearMap.pipeline_orchestrators.registration_orchestrator import RegistrationProcessor
from ClearMap.pipeline_orchestrators.tract_map import TractMapProcessor
from ClearMap.Scripts.align_new_api import stitch, register, plot_registration_results


def main(src_directory):
    bus = EventBus()
    cfg_coordinator_factory = make_cfg_coordinator_factory(bus)
    cfg_coordinator = cfg_coordinator_factory(src_directory)

    sample_manager = SampleManager(cfg_coordinator)
    sample_manager.setup(src_dir=src_directory)

    stitcher = StitchingProcessor(sample_manager, cfg_coordinator=cfg_coordinator)
    stitcher.setup()
    registration_processor = RegistrationProcessor(sample_manager, cfg_coordinator=cfg_coordinator)
    registration_processor.setup()

    stitch(stitcher)
    stitcher.plot_stitching_results(mode='overlay')

    register(registration_processor)
    plot_registration_results(registration_processor, sample_manager.alignment_reference_channel)

    tract_map_config = sample_manager.cfg_coordinator.get_config_view('tract_map')
    if 'template' in tract_map_config.keys() or not tract_map_config['channels'].values():
        print('Channels not yet configured in tract_map_params.cfg. Aborting.')
        return

    for channel in tract_map_config['channels']:
        tract_processor = TractMapProcessor(sample_manager, config_coordinator=cfg_coordinator, channel=channel,
                                            registration_processor=registration_processor)

        print('Starting Tract mapping')
        tract_processor.mask_to_coordinates(as_memmap=True)
        tract_processor.parallel_transform()
        tract_processor.label()
        tract_processor.voxelize()
        tract_processor.export_df()
        print('Mapping finished')


if __name__ == '__main__':
    # main(sys.argv[1])
    prefix = ''
    source_directories = [f'{prefix}{i}' for i in (1, 3, 8, 9, 10, 11, 12, 13, 14, 15, 16)]
    for src in source_directories:
        main(src)

