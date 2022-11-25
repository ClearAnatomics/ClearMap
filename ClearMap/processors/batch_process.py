# -*- coding: utf-8 -*-
"""
batch_process
=============

The processor for batch processing a group of samples.
This can be used from the GUI, from the CLI or interactively from the python interpreter
"""
import multiprocessing
import os
import sys

import numpy as np
from ClearMap import Settings
from skimage.transform import rescale
from tqdm import tqdm

from ClearMap.Alignment import Annotation as annotation
from ClearMap.config.atlas import ATLAS_NAMES_MAP, STRUCTURE_TREE_NAMES_MAP
from ClearMap.processors.tube_map import BinaryVesselProcessor, VesselGraphProcessor
from ClearMap.Utils.utilities import backup_file
from ClearMap.processors.cell_map import CellDetector
from ClearMap.processors.sample_preparation import PreProcessor
from ClearMap.config.config_loader import get_configs, ConfigLoader
from ClearMap.IO import IO as clearmap_io

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
    voxelize_folders(folders)


def voxelize_sample(configs, align=False, cells=False, vasc=False):
    patch_pipeline_name(configs, cells, vasc)

    pre_proc = PreProcessor()
    pre_proc.setup(configs)
    pre_proc.setup_atlases()
    if align:
        pre_proc.run()
    if cells:
        cell_detector = CellDetector(pre_proc)

        cell_detector.processing_config.reload()
        # cell_detector.atlas_align()
        # cell_detector.export_collapsed_stats()
        cell_detector.processing_config['voxelization']['radii'] = (10, 10, 10)
        cell_detector.processing_config.write()
        cell_detector.voxelize()


def voxelize_folders(folders, align=False, cells=True, vasc=False):
    for folder in tqdm(folders, desc='Processing sample ', unit='brain'):
        cfg_loader = ConfigLoader(folder)
        configs = get_configs(cfg_loader.get_cfg_path('sample'), cfg_loader.get_cfg_path('processing'))
        voxelize_sample(configs, align=align, cells=cells, vasc=vasc)


def init_preprocessor(folder, atlas_base_name=None):
    cfg_loader = ConfigLoader(folder)
    configs = get_configs(cfg_loader.get_cfg_path('sample'), cfg_loader.get_cfg_path('processing'))
    pre_proc = PreProcessor()
    if atlas_base_name is None:
        atlas_id = configs[2]['registration']['atlas']['id']
        atlas_base_name = ATLAS_NAMES_MAP[atlas_id]['base_name']
    json_file = os.path.join(Settings.atlas_folder, STRUCTURE_TREE_NAMES_MAP[configs[2]['registration']['atlas']['structure_tree_id']])
    pre_proc.unpack_atlas(atlas_base_name)
    pre_proc.setup(configs)
    pre_proc.setup_atlases()
    annotation.initialize(annotation_file=pre_proc.annotation_file_path, label_file=json_file)
    return pre_proc


def convert_to_cm_2_1(folder, atlas_base_name='ABA_25um'):
    pre_proc = init_preprocessor(folder, atlas_base_name)
    pre_proc.setup_atlases()
    cell_detector = CellDetector(pre_proc)
    cell_detector.convert_cm2_to_cm2_1_fmt()


def realign(folder, atlas_base_name='ABA_25um'):
    pre_proc = init_preprocessor(folder, atlas_base_name)
    pre_proc.setup_atlases()
    cell_detector = CellDetector(pre_proc)
    cell_detector.atlas_align()


def rescale_img(f_path, scaling_factor):
    print(f'Rescaling {f_path} by {scaling_factor}')
    img = clearmap_io.read(f_path)
    rescaled_img = rescale(img, scaling_factor, preserve_range=True, anti_aliasing=True)
    clearmap_io.write(f_path, rescaled_img)


def rescale_raw(folder, atlas_base_name='ABA_25um', dest_resolution=(3, 3, 6)):
    """
    Used to rescale to create e.g. test samples that can be ran quickly

    Parameters
    ----------
    folder str:
        The experiment folder with the
    atlas_base_name str:
        The base name of the atlas that serves as a file prefix for the atlas files
    dest_resolution tuple:
        The desired resolution

    Returns
    -------

    """
    pre_proc = init_preprocessor(folder, atlas_base_name)
    print(f'Processing {pre_proc.workspace.file_list("raw")}')

    scaling_factor = tuple(np.array(pre_proc.sample_config['resolutions']['raw']) / np.array(dest_resolution))

    p = multiprocessing.Pool()
    for f_path in pre_proc.workspace.file_list('raw'):
        p.apply_async(rescale_img, [f_path, scaling_factor])

    p.close()
    p.join()

    pre_proc.sample_config['resolutions']['raw'] = list(dest_resolution)
    pre_proc.sample_config.write()
    print('DONE')


if __name__ == '__main__':
    main(sys.argv[1])
