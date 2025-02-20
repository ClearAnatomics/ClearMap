import copy
import os

from ClearMap.IO import IO as clearmap_io
from ClearMap.IO.MHD import read as mhd_read
from ClearMap.Visualization.Qt import Plot3d as q_plot_3d
from ClearMap.Visualization.Qt.utils import link_dataviewers_cursors
from ClearMap.processors.sample_preparation import StitchingProcessor, RegistrationProcessor


def plot_registration_results(aligner: RegistrationProcessor, channel: str, composite: bool = False):
    img_paths = [
        aligner.get_fixed_image(channel).path,
        aligner.get_aligned_image(channel),
    ]
    if not all([p.exists() for p in img_paths]):
        raise ValueError(f'Missing requirements {img_paths}')
    # image_sources = copy.deepcopy(list(img_paths))
    # for i, im_path in enumerate(image_sources):
    #     if im_path.endswith('.mhd'):
    #         image_sources[i] = mhd_read(im_path)
    titles = [img.parent.stem if 'aligned_to' in str(img) else img.stem for img in img_paths]
    if composite:
        img_paths = [img_paths, ]
    dvs = q_plot_3d.plot(img_paths, title=titles, arrange=True, sync=True, lut='white')
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

    for channel in stitcher.get_stitching_order():
        config = stitcher.config['channels'][channel]
        if not config['run']:
            continue
        if config['use_npy'] and not sample_manager.has_npy(channel):
                stitcher.convert_tiles()
        if channel == config['layout_channel']:
            stitcher.stitch_channel_rigid(channel, _force=True)
            stitcher.stitch_channel_wobbly(channel, _force=True)  # TODO: check if force
        else:
            stitcher._stitch_layout_wobbly(channel)
