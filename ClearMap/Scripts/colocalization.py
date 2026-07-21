"""
colocalization
==============

Script for the Colocalization pipeline.

Quantifies spatial overlap between cells detected in two or more fluorescence channels.
Produces per-pair colocalization reports and density maps.

Prerequisites
-------------
Channels must be configured as ``cells`` or ``nuclei`` in the sample config.
Cell detection must either be run by this script (default) or have been previously run
with colocalization compatibility (``save_shape=True`` and ``atlas_align()`` completed).

Usage::

    python -m ClearMap.Scripts.colocalization /path/to/experiment

    # Skip cell detection if already run in colocalization-compatible mode:
    python -m ClearMap.Scripts.colocalization /path/to/experiment --skip-cell-map

"""
import argparse

from ClearMap.pipeline_orchestrators.cell_map import CellDetector
from ClearMap.pipeline_orchestrators.colocalization import ColocalizationProcessor
from ClearMap.pipeline_orchestrators.utils import init_sample_manager_and_processors
from ClearMap.Scripts.align_new_api import stitch, register, plot_registration_results


def check_colocalization_prerequisites(sample_manager, registration_processor):
    """
    Check whether CellMap was run in colocalization-compatible mode for all detection channels.

    A channel is colocalization-compatible when:

    1. The shape detection file exists (``cells_shape``), either bool or uint labeled —
       both work because colocalization re-runs ``ndi.label`` internally per block.
    2. The aligned cells Feather table exists (i.e. ``atlas_align()`` was run).

    Parameters
    ----------
    sample_manager : SampleManager
    registration_processor : RegistrationProcessor

    Returns
    -------
    dict[str, tuple[bool, str]]
        Mapping ``channel -> (ok, message)``.
    """
    results = {}
    for channel in sample_manager.channels_to_detect:
        detector = CellDetector(sample_manager,
                                config_coordinator=sample_manager.cfg_coordinator,
                                channel=channel,
                                registration_processor=registration_processor)
        shape_asset = detector.get('cells', channel=channel, asset_sub_type='shape')
        cells_feather = detector.get_path('cells', channel=channel, extension='.feather')

        if not shape_asset.exists:
            results[channel] = (False, 'Shape file missing — re-run CellMap with save_shape=True')
        elif not cells_feather.exists:
            results[channel] = (False, 'Aligned cells table (.feather) missing — run atlas_align() first')
        else:
            results[channel] = (True, f'OK (shape dtype={shape_asset.dtype()})')
    return results


def run_cell_map_colocalization_compatible(sample_manager, registration_processor):
    """
    Run CellMap for all detection channels with colocalization compatibility.

    Saves the shape detection volume (required by ``ColocalizationProcessor``) and the
    aligned cell Feather table (required for coordinate filtering).

    Parameters
    ----------
    sample_manager : SampleManager
    registration_processor : RegistrationProcessor
    """
    for channel in sample_manager.channels_to_detect:
        print(f'Running CellMap (colocalization-compatible) for channel: {channel}')
        detector = CellDetector(sample_manager,
                                config_coordinator=sample_manager.cfg_coordinator,
                                channel=channel,
                                registration_processor=registration_processor)
        detector.run_cell_detection(save_shape=True)
        detector.filter_cells()
        detector.atlas_align()
        detector.voxelize()
        detector.export_collapsed_stats()
        print(f'CellMap finished for channel: {channel}')


def main(src_directory, run_cell_map=True):
    """
    Run the full colocalization pipeline for a single experiment.

    Parameters
    ----------
    src_directory : str or Path
        Root experiment folder.
    run_cell_map : bool
        If True (default), run CellMap first with colocalization compatibility.
        Set to False if CellMap was already run with ``save_shape=True`` and
        ``atlas_align()`` completed.
    """
    orchestrators = init_sample_manager_and_processors(src_directory)
    sample_manager = orchestrators['sample_manager']
    stitcher = orchestrators['stitcher']
    registration_processor = orchestrators['registration_processor']

    if not sample_manager.is_colocalization_compatible:
        print(f'Skipping {src_directory}: colocalization requires at least two channels '
              f'configured as cells/nuclei.')
        return

    stitch(stitcher)
    register(registration_processor)
    plot_registration_results(registration_processor, sample_manager.alignment_reference_channel)

    if run_cell_map:
        run_cell_map_colocalization_compatible(sample_manager, registration_processor)
    else:
        prerequisites = check_colocalization_prerequisites(sample_manager, registration_processor)
        failed = {ch: msg for ch, (ok, msg) in prerequisites.items() if not ok}
        if failed:
            print('Cannot run colocalization — prerequisites not met:')
            for ch, msg in failed.items():
                print(f'  {ch}: {msg}')
            return

    for channel_a, channel_b in sample_manager.colocalization_pairs():
        print(f'Colocalizing {channel_a} — {channel_b}')
        processor = ColocalizationProcessor(sample_manager=sample_manager,
                                            config_coordinator=sample_manager.cfg_coordinator,
                                            channels=[channel_a, channel_b],
                                            registration_processor=registration_processor)
        if not processor.setup_finalised:
            print(f'  Setup failed for {channel_a}–{channel_b}. '
                  f'Check prerequisites with check_colocalization_prerequisites(). Skipping.')
            continue

        processor.compute_colocalization(channel_a, channel_b)
        processor.voxelize_filtered_table(channel_a, channel_b)
        print(f'  Finished: {channel_a}–{channel_b}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run the ClearMap colocalization pipeline.')
    parser.add_argument('src_directory', help='Root experiment folder')
    parser.add_argument('--skip-cell-map', action='store_true',
                        help='Skip CellMap (use if already run with save_shape=True)')
    args = parser.parse_args()
    main(args.src_directory, run_cell_map=not args.skip_cell_map)
