# -*- coding: utf-8 -*-
"""
batch_process
=============

The processor for batch processing a group of samples.
This can be used from the GUI, from the CLI or interactively from the python interpreter
"""

import sys

from ClearMap.processors.tube_map import BinaryVesselProcessor, VesselGraphProcessor
from tqdm import tqdm

from ClearMap.Utils.utilities import backup_file
from ClearMap.processors.cell_map import CellDetector
from ClearMap.processors.sample_preparation import PreProcessor
from ClearMap.config.config_loader import get_configs, ConfigLoader

__author__ = 'Charly Rousseau <charly.rousseau@icm-institute.org>'
__license__ = 'GPLv3 - GNU General Public License v3 (see LICENSE.txt)'
__copyright__ = 'Copyright © 2022 by Charly Rousseau'
__webpage__ = 'https://idisco.info'
__download__ = 'https://www.github.com/ChristophKirst/ClearMap2'


def process_sample(configs, align=False, cells=False, vasc=False):
    patch_pipeline_name(configs, cells, vasc)

    pre_proc = PreProcessor()
    pre_proc.setup(configs)
    pre_proc.setup_atlases()
    if align:
        pre_proc.run()
    if cells:
        cell_detector = CellDetector(pre_proc)

        cell_detector.processing_config.reload()
        cell_detector.atlas_align()
        cell_detector.export_collapsed_stats()
        cell_detector.voxelize()
        # cell_detector.run_cell_detection()
        #cell_detector.post_process_cells()
        # backup_file(cell_detector.workspace.filename('cells'))
        # cell_detector.atlas_align()
        # backup_file(cell_detector.workspace.filename('cells', extension='csv'))
        # # cell_detector.export_as_csv()
    if vasc:
        binary_vessel_processor = BinaryVesselProcessor(pre_proc)
        binary_vessel_processor.binarize()
        binary_vessel_processor.fill_vessels()
        binary_vessel_processor.combine_binary()

        vessel_graph_processor = VesselGraphProcessor(pre_proc)
        vessel_graph_processor.pre_process()
        vessel_graph_processor.post_process()


def patch_pipeline_name(configs, cells, vasc):
    configs[2]['pipeline_name'] = 'CellMap' if cells else 'TubeMap'
    if cells and vasc:
        configs[2]['pipeline_name'] = 'Both'


def process_folders(folders, align=False, cells=False, vasc=False):
    for folder in tqdm(folders, desc='Processing sample ', unit='brain'):
        cfg_loader = ConfigLoader(folder)
        configs = get_configs(cfg_loader.get_cfg_path('sample'), cfg_loader.get_cfg_path('processing'))
        process_sample(configs, align=align, cells=cells, vasc=vasc)


def main(samples_file):
    with open(samples_file, 'r') as infile:
        folders = infile.readlines()
    folders = [f.strip() for f in folders if not f.startswith('#')]
    process_folders(folders)


def convert_to_cm_2_1(folder, atlas_base_name='ABA_25um'):
    cfg_loader = ConfigLoader(folder)
    configs = get_configs(cfg_loader.get_cfg_path('sample'), cfg_loader.get_cfg_path('processing'))
    pre_proc = PreProcessor()
    pre_proc.unpack_atlas(atlas_base_name)
    pre_proc.setup(configs)
    pre_proc.setup_atlases()
    cell_detector = CellDetector(pre_proc)
    cell_detector.convert_cm2_to_cm2_1_fmt()


if __name__ == '__main__':
    main(sys.argv[1])
