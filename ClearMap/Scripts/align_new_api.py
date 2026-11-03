import sys

from ClearMap.pipeline_orchestrators.registration_orchestrator import RegistrationProcessor
from ClearMap.pipeline_orchestrators.utils import init_sample_manager_and_processors


def plot_registration_results(aligner: RegistrationProcessor, channel: str, composite: bool = False):
    # Imported here, not at the top: a headless job (cluster) stitches and registers without Qt installed
    from ClearMap.Visualization.Qt.utils import link_dataviewers_cursors
    dvs, titles = aligner.plot_registration_results(channel=channel, composite=composite)
    if not composite:
        link_dataviewers_cursors(dvs)


def main(src_directory, show_plots=True):
    """Stitch, then register to the atlas, the sample of the experiment folder `src_directory`."""
    orchestrators = init_sample_manager_and_processors(src_directory)
    sample_manager = orchestrators['sample_manager']
    stitcher = orchestrators['stitcher']
    registration_processor = orchestrators['registration_processor']

    stitcher.stitch()
    if show_plots:
        stitcher.plot_stitching_results(mode='overlay')

    registration_processor.register()
    if show_plots:
        plot_registration_results(registration_processor, sample_manager.alignment_reference_channel)


if __name__ == '__main__':
    main(sys.argv[1])
