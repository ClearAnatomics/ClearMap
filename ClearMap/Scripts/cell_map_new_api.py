"""
cell_map_new_api
================

Headless entry point for the CellMap pipeline (cell detection and density mapping).

This script replaces the deprecated :mod:`ClearMap.Scripts.CellMap`.
It initialises a sample from the experiment directory, runs stitching and
atlas registration, then detects cells, filters them, aligns coordinates to
atlas space, and produces voxelized density maps for every CellMap channel.

Usage
-----

.. code-block:: bash

    conda activate ClearMap3.1
    python -m ClearMap.Scripts.cell_map_new_api /path/to/experiment

The experiment directory must contain a ``sample.yml`` config file (created
by the GUI or by copying from ``~/.clearmap/defaults/``).  All pipeline
parameters are read from the YAML files in that directory; no editing of
this script is required.

Steps performed
---------------
1. **Stitching** — assembles tiles into a single volume for each channel.
2. **Registration** — resamples the autofluorescence channel and aligns it
   to the atlas via Elastix.
3. **Cell detection** (per CellMap channel) —

   a. :meth:`~ClearMap.pipeline_orchestrators.cell_map.CellDetector.run_cell_detection`
      — background subtraction, maxima detection, shape-based watershed.
   b. :meth:`~ClearMap.pipeline_orchestrators.cell_map.CellDetector.post_process_cells`
      — intensity/size filtering and atlas-space coordinate alignment.
   c. :meth:`~ClearMap.pipeline_orchestrators.cell_map.CellDetector.voxelize`
      — rasterise cell positions into a density volume.
   d. Plots the density map and a 3-D scatter of cells coloured by atlas region.

Outputs (written to the experiment directory via the workspace)
---------------------------------------------------------------
* ``cells_raw.npy`` — raw detected cell table per channel
* ``cells_filtered.npy`` — filtered cell table
* ``cells.feather`` — atlas-annotated cell table (coordinates + region labels)
* ``cells_stats.csv`` — per-structure cell counts and average sizes
* ``density_counts.tif`` — voxelized cell density in atlas space

See also
--------
:class:`~ClearMap.pipeline_orchestrators.cell_map.CellDetector` :
    The worker class that implements each detection step.
:func:`~ClearMap.pipeline_orchestrators.utils.init_sample_manager_and_processors` :
    Convenience factory used to initialise all standard workers.
:doc:`/cellmap` :
    Full CellMap pipeline documentation.
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
