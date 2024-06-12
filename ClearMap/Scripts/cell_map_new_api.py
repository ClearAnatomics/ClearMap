"""
This script replaces the old CellMap.py script which is now deprecated

To run the analysis, create and edit
the sample_params.cfg, alignment_params.cfg and cell_map_params.cfg files
in the data folder and call this script with the folder as single argument
optionally, provide the atlas base name as second argument
"""
import sys

from ClearMap.Scripts.align_new_api import plot_registration_results, register, convert_stitched, stitch
from ClearMap.config.config_loader import ConfigLoader, get_configs
from ClearMap.processors.cell_map import CellDetector
from ClearMap.processors.sample_preparation import PreProcessor


def main(src_directory, atlas_base_name='ABA_25um_2017'):
    cfg_loader = ConfigLoader(src_directory)
    configs = get_configs(cfg_loader.get_cfg_path('sample'), cfg_loader.get_cfg_path('processing'))
    pre_proc = PreProcessor()
    pre_proc.setup(configs)

    stitch(pre_proc)
    # if all(pre_proc.processing_config['stitching']['preview'].values)
    pre_proc.plot_stitching_results()

    convert_stitched(pre_proc)

    register(atlas_base_name, pre_proc)
    plot_registration_results(pre_proc)

    cell_detector = CellDetector(pre_proc)

    # TEST CELL DETECTION
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
    if len(sys.argv) > 2:
        main(sys.argv[1], sys.argv[2])
    else:
        main(sys.argv[1])
