import os
import sys

from matplotlib import pyplot as plt

sys.path.insert(0, os.path.abspath('.'))

from ClearMap.processors.sample_preparation import SampleManager, StitchingProcessor
import ClearMap.Alignment.Stitching.StitchingRigid as stitching_rigid



def plot_all_layouts(folder):
    sample_manager = SampleManager()
    sample_manager.setup(folder)
    stitcher = StitchingProcessor(sample_manager)
    for postfix in ('aligned_axis', 'aligned', 'placed'):
        layout = stitching_rigid.load_layout(sample_manager.get_path('layout', postfix=postfix))
        overlay = stitcher.overlay_layout_plane(layout)
        plt.imshow(overlay)
        plt.show()


if __name__ == '__main__':
    plot_all_layouts('/data/sample_folder')
