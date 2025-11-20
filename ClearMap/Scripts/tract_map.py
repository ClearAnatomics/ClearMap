import sys

from ClearMap.processors.sample_preparation import SampleManager, StitchingProcessor, RegistrationProcessor
from ClearMap.Scripts.align_new_api import stitch, register, plot_registration_results
from ClearMap.processors.tract_map import TractMapProcessor


def main(src_directory):
    sample_manager = SampleManager()
    sample_manager.setup(src_dir=src_directory)

    stitcher = StitchingProcessor(sample_manager)
    stitcher.setup()
    registration_processor = RegistrationProcessor(sample_manager)
    registration_processor.setup()

    stitch(stitcher)
    stitcher.plot_stitching_results(mode='overlay')

    register(registration_processor)
    plot_registration_results(registration_processor, sample_manager.alignment_reference_channel)

    tract_map_config = sample_manager.config_loader.get_cfg('tract_map')
    if 'example' in tract_map_config:
        print('Channels not yet configured in tract_map_params.cfg. Aborting.')
        return

    for channel in tract_map_config.keys():
        tract_processor = TractMapProcessor(sample_manager, channel=channel,
                                          registration_processor=registration_processor)

        print('Starting Tract mapping')
        tract_processor.reload_config()

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

