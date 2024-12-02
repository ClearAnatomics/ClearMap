import os
import sys

from matplotlib import pyplot as plt

sys.path.insert(0, os.path.abspath('.'))

from ClearMap.IO.metadata import define_auto_stitching_params
from ClearMap.processors.sample_preparation import init_preprocessor
import ClearMap.Alignment.Stitching.StitchingRigid as stitching_rigid


def stitch_and_plot_layout(channel):
    pre_proc = init_preprocessor('/data/test/')
    stitching_cfg = pre_proc.processing_config['stitching'][channel]['rigid']
    overlaps, projection_thickness = define_auto_stitching_params(
        pre_proc.workspace.source('raw', channel=channel).file_list[0], stitching_cfg)
    layout = pre_proc.get_wobbly_layout(overlaps)
    return pre_proc.overlay_layout_plane(layout.copy())


def plot_all_layouts(folder):
    pre_proc = init_preprocessor(folder)
    for postfix in ('aligned_axis', 'aligned', 'placed'):
        layout = stitching_rigid.load_layout(pre_proc.filename('layout', postfix=postfix))
        overlay = pre_proc.overlay_layout_plane(layout)
        plt.imshow(overlay)
        plt.show()


if __name__ == '__main__':
    plot_all_layouts('/data/sample_folder')
