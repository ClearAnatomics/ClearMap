"""
This script replaces the old CellMap.py script which is now deprecated

To run the analysis, create and edit
the sample_params.cfg, alignment_params.cfg and cell_map_params.cfg files
in the data folder and call this script with the folder as single argument
optionally, provide the atlas base name as second argument
"""
import sys

from ClearMap.pipeline_orchestrators.utils import init_sample_manager_and_processors
from ClearMap.pipeline_orchestrators.cell_map import CellDetector

from ClearMap.Scripts.align_new_api import plot_registration_results, register, stitch


def main(src_directory):
    orchestrators = init_sample_manager_and_processors(src_directory)
    sample_manager = orchestrators['sample_manager']
    stitcher = orchestrators['stitcher']
    registration_processor = orchestrators['registration_processor']

    stitch(stitcher)
    stitcher.plot_stitching_results(mode='overlay')

    register(registration_processor)
    plot_registration_results(registration_processor, sample_manager.alignment_reference_channel)

    for channel in sample_manager.get_channels_by_pipeline('CellMap', as_list=True):
        cell_detector = CellDetector(sample_manager, config_coordinator=sample_manager.cfg_coordinator,
                                     channel=channel, registration_processor=registration_processor)
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
        cell_detector.voxelize()
        cell_detector.plot_voxelized_counts(arrange=True)
        print('Cell detection done')

        cell_detector.plot_cells_3d_scatter_w_atlas_colors()


if __name__ == '__main__':
    main(sys.argv[1])
