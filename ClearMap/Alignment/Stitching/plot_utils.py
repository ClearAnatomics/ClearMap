import os
import sys

from matplotlib import pyplot as plt

sys.path.insert(0, os.path.abspath('.'))

from ClearMap.pipeline_orchestrators.utils import init_sample_manager_and_processors
import ClearMap.Alignment.Stitching.StitchingRigid as stitching_rigid



def plot_all_layouts(folder):
    orchestrators = init_sample_manager_and_processors(folder)
    sample_manager = orchestrators['sample_manager']
    stitcher = orchestrators['stitcher']
    for postfix in ('aligned_axis', 'aligned', 'placed'):
        layout = stitching_rigid.load_layout(sample_manager.get_path('layout', asset_sub_type=postfix))
        overlay = stitcher.overlay_layout_plane(layout)
        plt.imshow(overlay)
        plt.show()


if __name__ == '__main__':
    plot_all_layouts('/data/sample_folder')
