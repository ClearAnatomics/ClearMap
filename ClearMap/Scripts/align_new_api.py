from ClearMap.Visualization.Qt.utils import link_dataviewers_cursors
from ClearMap.pipeline_orchestrators.stitching_orchestrator import StitchingProcessor
from ClearMap.pipeline_orchestrators.registration_orchestrator import RegistrationProcessor


def plot_registration_results(aligner: RegistrationProcessor, channel: str, composite: bool = False):
    dvs, titles = aligner.plot_registration_results(channel=channel, composite=composite)
    if not composite:
        link_dataviewers_cursors(dvs)


def register(aligner: RegistrationProcessor):
    print('Registering')
    aligner.setup_atlases()
    for channel in aligner.channels_to_resample():
        aligner.resample_channel(channel)
    print('Aligning')
    aligner.align()
    print('Registered')


def stitch(stitcher: StitchingProcessor):  # FIXME: part of stitcher object
    sample_manager = stitcher.sample_manager
    for channel in sample_manager.channels:
        if not sample_manager.is_tiled(channel):
            stitcher.copy_or_stack(channel)

    for stitching_tree in stitcher.get_stitching_order(strict=True).values():
        for channel in stitching_tree:
            config = stitcher.config['channels'][channel]
            if not config['run']:
                continue
            if config['use_npy'] and not sample_manager.has_npy(channel):
                    stitcher.convert_tiles()
            if channel == config['layout_channel']:  # If self reference -> compute layout and stitch
                stitcher.align_channel_rigid(channel, _force=True)
                stitcher.stitch_channel_wobbly(channel, _force=True)  # TODO: check if force
            else:  # If not self reference -> apply layout and stitch
                stitcher._stitch_layout_wobbly(channel)
