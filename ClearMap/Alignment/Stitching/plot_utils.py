import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath('.'))

from matplotlib import pyplot as plt

from ClearMap.IO import IO as clearmap_io
from ClearMap.IO.metadata import define_auto_stitching_params
from ClearMap.processors.sample_preparation import init_preprocessor
import ClearMap.Alignment.Stitching.StitchingRigid as stitching_rigid


def overlay_layout_plane(layout):
    """Overlays the sources to check their placement.

    Arguments
    ---------
    layout : Layout class
      The layout with the sources to overlay.

    Returns
    -------
    image : array
      A color image.
    """
    sources = layout.sources

    dest_shape = tuple(layout.extent[:-1])
    full_lower = layout.lower
    middle_z = round(dest_shape[-1] / 2)

    cyan_image = np.zeros(dest_shape, dtype=int)
    magenta_image = np.zeros(dest_shape, dtype=int)
    # construct full image
    for s in sources:
        l = s.lower
        u = s.upper
        tile = clearmap_io.read(s.location)[:, :, middle_z]  # So as not to load the data into the list for memory efficiency
        current_slicing = tuple(slice(ll - fl, uu - fl) for ll, uu, fl in zip(l, u, full_lower))[:2]

        is_odd = sum(s.tile_position) % 2
        if is_odd:  # Alternate colors
            layer = cyan_image
        else:
            layer = magenta_image

        layer[current_slicing] = tile  # FIXME:colorise
    blank = np.zeros(dest_shape, dtype=cyan_image.dtype)

    high_intensity = (cyan_image.mean() + 4 * cyan_image.std())
    cyan_image = cyan_image / high_intensity * 128
    cyan_image = np.dstack((blank, cyan_image, cyan_image))  # To Cyan RGB

    high_intensity = (magenta_image.mean() + 4 * magenta_image.std())
    magenta_image = magenta_image / high_intensity * 128
    magenta_image = np.dstack((magenta_image, blank, magenta_image))  # To Magenta RGB

    output_image = cyan_image + magenta_image  # TODO: normalise
    output_image = output_image.clip(0, 255).astype(np.uint8)

    return output_image


def stitch_and_plot_layout():
    pre_proc = init_preprocessor('/data/test/')
    stitching_cfg = pre_proc.processing_config['stitching']
    overlaps, projection_thickness = define_auto_stitching_params(pre_proc.workspace.source('raw').file_list[0],
                                                                  stitching_cfg)
    layout = pre_proc.get_wobbly_layout(overlaps)
    return overlay_layout_plane(layout.copy())


def plot_all_layouts(folder):
    pre_proc = init_preprocessor(folder)
    for postfix in ('aligned_axis', 'aligned', 'placed'):
        layout = stitching_rigid.load_layout(pre_proc.filename('layout', postfix=postfix))
        overlay = overlay_layout_plane(layout)
        plt.imshow(overlay)
        plt.show()


if __name__ == '__main__':
    plot_all_layouts('/data/sample_folder')
