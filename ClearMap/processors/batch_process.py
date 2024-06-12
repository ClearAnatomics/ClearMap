# -*- coding: utf-8 -*-
"""
batch_process
=============

The processor set of scripts to batch process a group of samples.
This can be used from the GUI, from the CLI or interactively from the python interpreter
"""
import functools
import multiprocessing
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from skimage.transform import rescale
from tqdm import tqdm

try:
    from ClearMap.processors.tube_map import BinaryVesselProcessor, VesselGraphProcessor
    graph_gt = True
except ImportError:
    graph_gt = False
from ClearMap.processors.cell_map import CellDetector
from ClearMap.processors.sample_preparation import PreProcessor, init_preprocessor
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

        cell_detector.run_cell_detection()
        cell_detector.post_process_cells()
        cell_detector.voxelize()
    if vasc:
        binary_vessel_processor = BinaryVesselProcessor(pre_proc)
        binary_vessel_processor.binarize()
        binary_vessel_processor.combine_binary()

        vessel_graph_processor = VesselGraphProcessor(pre_proc)
        vessel_graph_processor.pre_process()
        vessel_graph_processor.post_process()


def patch_pipeline_name(configs, cells, vasc):
    configs[2]['pipeline_name'] = 'CellMap' if cells else 'TubeMap'
    if cells and vasc:
        configs[2]['pipeline_name'] = 'Both'


class BatchProcessor:
    def __init__(self, progress_watcher, results_folder=None, params=None):
        self.results_folder = results_folder
        self.progress_watcher = progress_watcher

        self.params = params

    def process_folders(self):
        paths = [p for ps in self.params.get_all_paths() for p in ps]  # flatten list
        for folder in tqdm(paths, desc='Processing sample ', unit='brain'):
            cfg_loader = ConfigLoader(folder)
            configs = get_configs(cfg_loader.get_cfg_path('sample'), cfg_loader.get_cfg_path('processing'))
            process_sample(configs,  align=self.params.align, cells=self.params.count_cells,
                           vasc=self.params.run_vaculature)
        self.progress_watcher.finish()


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


def convert_to_cm_2_1(folder, atlas_base_name='ABA_25um'):
    pre_proc = init_preprocessor(folder, atlas_base_name)
    pre_proc.setup_atlases()
    cell_detector = CellDetector(pre_proc)
    cell_detector.convert_cm2_to_cm2_1_fmt()


def realign(folder, atlas_base_name='ABA_25um'):
    pre_proc = init_preprocessor(folder, atlas_base_name)
    pre_proc.setup_atlases()
    cell_detector = CellDetector(pre_proc)
    cell_detector.filter_cells()
    cell_detector.atlas_align()


def rescale_img(f_path, scaling_factor):
    print(f'Rescaling {f_path} by {scaling_factor}')
    img = clearmap_io.read(f_path)
    rescaled_img = rescale(img, scaling_factor, preserve_range=True, anti_aliasing=True)
    clearmap_io.write(f_path, rescaled_img)


def rescale_channel(folder, atlas_base_name=None, dest_resolution=(3, 3, 6), n_cpus=None, channel='raw', ext='.tif',
                    chunk_size=1):
    """
    Used to rescale to create e.g. test samples that can be ran quickly

    Parameters
    ----------
    folder str:
        The experiment folder with the tiles
    atlas_base_name str:
        The base name of the atlas that serves as a file prefix for the atlas files
    dest_resolution tuple:
        The desired resolution

    Returns
    -------

    """
    n_cpus = multiprocessing.cpu_count() - 2 if n_cpus is None else n_cpus

    pre_proc = init_preprocessor(folder, atlas_base_name)
    file_list = pre_proc.workspace.file_list(channel, prefix=pre_proc.prefix, extension=ext)
    print(f'Processing {file_list}')

    scaling_factors = np.array(pre_proc.sample_config['resolutions'][channel]) / np.array(dest_resolution)
    print(scaling_factors)
    rescale_f = functools.partial(rescale_img, scaling_factor=tuple(scaling_factors))

    if n_cpus*chunk_size == 1:
        for f_path in file_list:
            rescale_f(f_path)
    else:
        with ThreadPoolExecutor(n_cpus) as executor:
            executor.map(rescale_f, file_list, chunksize=chunk_size)

    pre_proc.sample_config['resolutions'][channel] = list(dest_resolution)
    pre_proc.sample_config.write()
    print('DONE')


def batch_crop(folder, crop_x=0, crop_y=0, suffix='_cropped'):
    from ClearMap.processors.sample_preparation import init_preprocessor
    preproc = init_preprocessor(folder)

    shape = clearmap_io.shape(preproc.workspace.file_list('raw')[-1])
    for f_path in preproc.workspace.file_list('raw'):
        img = clearmap_io.read(f_path)
        cropped_img = img[crop_x:shape[0] - crop_x, crop_y:shape[1] - crop_y, :]
        cropped_path = f_path.replace('.ome.tif', f'{suffix}.ome.tif')
        clearmap_io.write(cropped_path, cropped_img)


if __name__ == '__main__':
    main(sys.argv[1])
