"""
This script replaces the old CellMap.py script which is now deprecated

To run the analysis, create and edit
the sample_params.cfg, alignment_params.cfg and cell_map_params.cfg files
in the data folder and call this script with the folder as single argument
optionally, provide the atlas base name as second argument
"""
import sys
import os
import shutil
os.environ['TMP'] = '/data/maxime.boyer/1_tmp'

# sys.path.insert(0, '/home/maxime.boyer/Documents/1_Projects/1_ClearMap/ClearMap/')

from ClearMap.Scripts.align_new_api import stitch, register, plot_registration_results
from ClearMap.processors.sample_preparation import SampleManager, StitchingProcessor, RegistrationProcessor
from ClearMap.processors.tube_map import BinaryVesselProcessor, VesselGraphProcessor


def main(src_directory):
    # shutil.copy(
    #     '/network/iss/renier/projects/vasculature/pregnancy/processed/230411-vasculature-timepoint_idisco/230411-1/1_stitched.npy',
    #     src_directory)
    sample_manager = SampleManager()
    sample_manager.setup(src_dir=src_directory)

    # stitcher = StitchingProcessor(sample_manager)
    # stitcher.setup()
    registration_processor = RegistrationProcessor(sample_manager)
    registration_processor.setup()

    # stitch(stitcher)
    # stitcher.plot_stitching_results(mode='overlay')

    # register(registration_processor)
    # plot_registration_results(registration_processor, sample_manager.alignment_reference_channel)

    binary_vessel_processor = BinaryVesselProcessor(sample_manager)

    for channel in sample_manager.get_channels_by_pipeline('TubeMap', as_list=True):  # ["vasc"]
        binary_vessel_processor.binarize_channel(channel)
        binary_vessel_processor.smooth_channel(channel)
        binary_vessel_processor.fill_channel(channel)
        #binary_vessel_processor.deep_fill_channel(channel)

    binary_vessel_processor.combine_binary()
    # binary_vessel_processor.plot_combined(arrange=True)

    vessel_graph_processor = VesselGraphProcessor(sample_manager, registration_processor)
    vessel_graph_processor.pre_process()
    # TODO: slice
    vessel_graph_processor.post_process()
    vessel_graph_processor.voxelize()
    # vessel_graph_processor.plot_voxelization(None)


if __name__ == '__main__':
    main(sys.argv[1])
