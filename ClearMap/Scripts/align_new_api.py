import copy
import os

from ClearMap.IO import IO as clearmap_io
from ClearMap.IO.MHD import read as mhd_read
from ClearMap.Visualization.Qt import Plot3d as q_plot_3d


def plot_registration_results(pre_proc):
    img_paths = (pre_proc.filename('resampled', postfix='autofluorescence'), pre_proc.aligned_autofluo_path)
    image_sources = copy.deepcopy(list(img_paths))
    for i, im_path in enumerate(image_sources):
        if im_path.endswith('.mhd'):
            image_sources[i] = mhd_read(im_path)
    titles = [os.path.basename(img) for img in img_paths]
    q_plot_3d.plot([image_sources, ], title=' '.join(titles), arrange=True, sync=True, lut='white')


def register(atlas_base_name, pre_proc):
    print('Registering')
    pre_proc.unpack_atlas(atlas_base_name)
    pre_proc.setup_atlases()
    if not pre_proc.processing_config['registration']['resampling']['skip']:
        print('Resampling for registering')
        pre_proc.resample_for_registration(force=True)
    print('Aligning')
    pre_proc.align()
    print('Registered')


def convert_stitched(pre_proc):
    if not pre_proc.processing_config['stitching']['output_conversion']['skip']:
        fmt = pre_proc.processing_config['stitching']['output_conversion']['format']
        print(f'Converting stitched image to {fmt}')
        pre_proc.convert_to_image_format()


def stitch(pre_proc):
    tags = pre_proc.workspace.expression('raw', prefix=pre_proc.prefix).tags
    if tags is not None:
        axes = [tag.name for tag in tags]
    if tags is None or axes == ['Z']:  # BYPASS stitching, just copy or stack
        clearmap_io.convert(pre_proc.filename('raw'), pre_proc.filename('stitched'))
    else:  # assume tiling
        if not pre_proc.processing_config['stitching']['rigid']['skip']:
            pre_proc.stitch_rigid(force=True)
            print('Stitched rigid')
        if not pre_proc.processing_config['stitching']['wobbly']['skip']:
            if pre_proc.was_stitched_rigid:
                pre_proc.stitch_wobbly(force=pre_proc.processing_config['stitching']['rigid']['skip'])
                print('Stitched wobbly')
            else:
                print('Could not run wobbly stitching <br>without rigid stitching first')
