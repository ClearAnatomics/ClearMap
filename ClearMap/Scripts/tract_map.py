from ClearMap.pipeline_orchestrators.tract_map import TractMapProcessor
from ClearMap.pipeline_orchestrators.utils import init_sample_manager_and_processors

from ClearMap.Scripts.align_new_api import stitch, register, plot_registration_results


def main(src_directory):
    orchestrators = init_sample_manager_and_processors(src_directory)
    sample_manager = orchestrators['sample_manager']
    stitcher = orchestrators['stitcher']
    registration_processor = orchestrators['registration_processor']
    cfg_coordinator = sample_manager.cfg_coordinator

    stitch(stitcher)
    # stitcher.plot_stitching_results(mode='overlay')

    register(registration_processor)
    plot_registration_results(registration_processor, sample_manager.alignment_reference_channel)

    for channel in sample_manager.get_channels_by_pipeline('TractMap', as_list=True):
        tract_processor = TractMapProcessor(sample_manager, config_coordinator=cfg_coordinator,
                                            channel=channel,
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
