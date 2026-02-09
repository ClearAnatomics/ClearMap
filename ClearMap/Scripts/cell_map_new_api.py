"""
This script replaces the old CellMap.py script which is now deprecated

To run the analysis, create and edit
the sample_params.cfg, alignment_params.cfg and cell_map_params.cfg files
in the data folder and call this script with the folder as single argument
optionally, provide the atlas base name as second argument
"""
import sys

from ClearMap.processors.sample_preparation import SampleManager, StitchingProcessor,  RegistrationProcessor
from ClearMap.Scripts.align_new_api import plot_registration_results, register, stitch
from ClearMap.processors.cell_map import CellDetector


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

    cell_detection_config = sample_manager.config_loader.get_cfg('cell_map')['channels']
    if 'example' in cell_detection_config:
        print('Channels not yet configured in cell_map_params.cfg. Aborting.')
        return

    for channel in cell_detection_config.keys():
        cell_detector = CellDetector(sample_manager, channel=channel, registration_processor=registration_processor)
        # TEST CELL DETECTION
        # slicing = (
        #    slice(*cell_detector.processing_config['test_set_slicing']['dim_0']),
        #    slice(*cell_detector.processing_config['test_set_slicing']['dim_1']),
        #    slice(*cell_detector.processing_config['test_set_slicing']['dim_2'])
        # )
        # cell_detector.create_test_dataset(slicing=[......])
        # print('Cell detection preview')
        # cell_detector.run_cell_detection(tuning=True)
        # dvs = cell_detector.preview_cell_detection(arrange=True, sync=True)
        # link_dataviewers_cursors(dvs, RedCross)

        print('Starting cell detection')
        cell_detector.run_cell_detection(tuning=False)
        cell_detector.post_process_cells()
        cell_detector.plot_voxelized_counts(arrange=True)
        print('Cell detection done')

        cell_detector.plot_cells_3d_scatter_w_atlas_colors()


if __name__ == '__main__':
    main(sys.argv[1])
