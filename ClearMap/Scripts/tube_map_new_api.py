"""
This script replaces the old CellMap.py script which is now deprecated

To run the analysis, create and edit
the sample_params.cfg, alignment_params.cfg and cell_map_params.cfg files
in the data folder and call this script with the folder as single argument
optionally, provide the atlas base name as second argument
"""
import sys

from ClearMap.Scripts.align_new_api import stitch, convert_stitched, register, plot_registration_results
from ClearMap.config.config_loader import ConfigLoader, get_configs
from ClearMap.processors.sample_preparation import PreProcessor
from ClearMap.processors.tube_map import BinaryVesselProcessor, VesselGraphProcessor


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

    binary_vessel_processor = BinaryVesselProcessor(pre_proc)

    binary_vessel_processor.setup(pre_proc)
    binary_vessel_processor.binarize()
    binary_vessel_processor.plot_binarization_result(arrange=True)
    binary_vessel_processor.plot_vessel_filling_results(arrange=True)
    binary_vessel_processor.combine_binary()
    binary_vessel_processor.plot_combined(arrange=True)

    vessel_graph_processor = VesselGraphProcessor(pre_proc)
    vessel_graph_processor.setup(pre_proc)
    vessel_graph_processor.pre_process()
    # FIXME: slice
    vessel_graph_processor.post_process()
    vessel_graph_processor.voxelize()
    vessel_graph_processor.plot_voxelization(None)


if __name__ == '__main__':
    if len(sys.argv) > 2:
        main(sys.argv[1], sys.argv[2])
    else:
        main(sys.argv[1])
