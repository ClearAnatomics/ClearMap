"""
tube_map_new_api
================

Headless entry point for the TubeMap pipeline (vasculature binarization,
graph construction, and annotation).

This script replaces the deprecated :mod:`ClearMap.Scripts.TubeMap`.
It initialises a sample from the experiment directory, runs stitching and
atlas registration, then binarizes each vessel channel, merges the binary
masks, and builds an annotated vasculature graph.

Usage
-----

.. code-block:: bash

    conda activate ClearMap3.1
    python -m ClearMap.Scripts.tube_map_new_api /path/to/experiment

The experiment directory must contain a ``sample.yml`` config file (created
by the GUI or by copying from ``~/.clearmap/defaults/``).  All pipeline
parameters are read from the YAML files in that directory; no editing of
this script is required.

Steps performed
---------------
1. **Stitching** — assembles tiles into a single volume for each channel.
2. **Registration** — resamples the autofluorescence channel and aligns it
   to the atlas via Elastix.
3. **Binarization** (per TubeMap channel) —

   a. :meth:`~ClearMap.pipeline_orchestrators.tube_map.BinaryVesselProcessor.binarize_channel`
      — multi-path thresholding.
   b. :meth:`~ClearMap.pipeline_orchestrators.tube_map.BinaryVesselProcessor.smooth_channel`
      — topology-preserving binary smoothing.
   c. :meth:`~ClearMap.pipeline_orchestrators.tube_map.BinaryVesselProcessor.fill_channel`
      — parallel 3-D binary hole filling.
   d. :meth:`~ClearMap.pipeline_orchestrators.tube_map.BinaryVesselProcessor.deep_fill_channel`
      — CNN-based hollow-tube filling (requires GPU with ≥ 24 GB VRAM).

4. :meth:`~ClearMap.pipeline_orchestrators.tube_map.BinaryVesselProcessor.combine_binary`
   — logical-OR merge of all channel masks into a single combined binary.
5. **Graph construction** —

   a. :meth:`~ClearMap.pipeline_orchestrators.tube_map.VesselGraphProcessor.pre_process`
      — skeletonize → build raw graph → clean → reduce → register to atlas.
   b. :meth:`~ClearMap.pipeline_orchestrators.tube_map.VesselGraphProcessor.post_process`
      — iterative artery/vein tracing and capillary removal (if artery channel present).
   c. :meth:`~ClearMap.pipeline_orchestrators.tube_map.VesselGraphProcessor.voxelize`
      — rasterise graph vertices into a branch density volume.

Outputs (written to the experiment directory via the workspace)
---------------------------------------------------------------
* ``binary.npy`` / ``binary_smoothed.npy`` / … — intermediate binary masks
* ``binary_combined.npy`` — merged vessel mask
* ``skeleton.npy`` — skeletonized binary
* ``graph_raw.gt`` / ``graph_cleaned.gt`` / ``graph_reduced.gt`` / ``graph_annotated.gt``
  — graph at each construction stage
* ``density_branches.tif`` — voxelized vessel density in atlas space
* ``vertices.feather`` — vertex table with coordinates, radii, and atlas labels

See also
--------
:class:`~ClearMap.pipeline_orchestrators.tube_map.BinaryVesselProcessor` :
    Binarization worker.
:class:`~ClearMap.pipeline_orchestrators.tube_map.VesselGraphProcessor` :
    Graph construction and annotation worker.
:func:`~ClearMap.pipeline_orchestrators.utils.init_sample_manager_and_processors` :
    Convenience factory used to initialise all standard workers.
:doc:`/tubemap` :
    Full TubeMap pipeline documentation.
"""
import sys

from ClearMap.pipeline_orchestrators.utils import init_sample_manager_and_processors
from ClearMap.pipeline_orchestrators.tube_map import BinaryVesselProcessor, VesselGraphProcessor

from ClearMap.Scripts.align_new_api import stitch, register, plot_registration_results


def main(src_directory):
    orchestrators = init_sample_manager_and_processors(src_directory)
    sample_manager = orchestrators['sample_manager']
    stitcher = orchestrators['stitcher']
    registration_processor = orchestrators['registration_processor']

    stitch(stitcher)
    stitcher.plot_stitching_results(mode='overlay')

    register(registration_processor)
    plot_registration_results(registration_processor, sample_manager.alignment_reference_channel)

    binary_vessel_processor = BinaryVesselProcessor(sample_manager,
                                                    config_coordinator=sample_manager.cfg_coordinator)

    for channel in sample_manager.get_channels_by_pipeline('TubeMap', as_list=True):
        binary_vessel_processor.binarize_channel(channel)
        binary_vessel_processor.smooth_channel(channel)
        binary_vessel_processor.fill_channel(channel)
        binary_vessel_processor.deep_fill_channel(channel)

    binary_vessel_processor.combine_binary()
    # binary_vessel_processor.plot_combined(arrange=True)

    vessel_graph_processor = VesselGraphProcessor(sample_manager, config_coordinator=sample_manager.cfg_coordinator,
                                                  registration_processor=registration_processor)
    vessel_graph_processor.pre_process()
    # TODO: slice
    vessel_graph_processor.post_process()
    vessel_graph_processor.voxelize()
    # vessel_graph_processor.plot_voxelization(None)


if __name__ == '__main__':
    main(sys.argv[1])
