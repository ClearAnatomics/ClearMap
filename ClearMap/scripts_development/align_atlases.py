from ClearMap.Environment import *  #analysis:ignore
import sys
import numpy as np
from ClearMap.IO.MHD import mhd_read
from skimage.exposure import match_histograms,rescale_intensity

def get_workspace(script_path, match_str):
     with open(script_path, 'r') as script:
         code_lines = script.readlines()

     start_idx = [i for i, ln in enumerate(code_lines) if match_str in ln][0]
     code_str = ''.join(code_lines[:start_idx+1])  # line included
     print(code_str)
     locs = {}
     exec(code_str, globals(), locs)
     return locs

#P4
# directory='/data/elisa.delaunoit/210811_dev_p4/220105-test-ib4-p4/'
# sub_dir='220105-4t'

#P5
# directory='/data/elisa.delaunoit/210811_dev_p5/'
# sub_dir='210811-mbp'
# template_new='/data/elisa.delaunoit/210811_dev_p5/normed_max_atlas_auto_new.tif'
#P3
# directory='/data/elisa.delaunoit/210811_dev_p3/'
# sub_dir='210817-3d'
#P1
# directory='/data/elisa.delaunoit/210811_dev_p1/'
# sub_dir='210811-1b'
# template_new='/home/elisa.delaunoit/Documents/sophie/atlases/P1/normed_max_atlas_auto.tif'
# P7
directory='/data/elisa.delaunoit/210811_dev_p14/'
sub_dir='210811-14c'
template_new='/home/elisa.delaunoit/Documents/sophie/atlases/P14/normed_max_atlas_auto.tif'
# P9
# directory='/data/elisa.delaunoit/p10/'
# sub_dir='13'
# template_new='/data/elisa.delaunoit/p10/normed_max_atlas_auto.tif'



script_path = directory + sub_dir +'/'+sub_dir[-3:] + '-TubeMap.py'
vars_dict = get_workspace(script_path, 'align_reference_bspline_file = ')
ws = vars_dict['ws']
ws.info()

# templateP4=vars_dict['reference_file']
# annotation_P4=vars_dict['annotation_file']

# annotation_ref='/home/elisa.delaunoit/Documents/sophie/atlases/P3/ano_full_P3_sagital.tif'
# template_ref='/data/elisa.delaunoit/210811_dev_p3/normed_max_atlas_auto_new_354.tif'

# annotation_ref='/home/elisa.delaunoit/Documents/sophie/atlases/P5/ano_P_good.tif'
# template_ref='/home/elisa.delaunoit/Documents/sophie/atlases/P5/normed_max_atlas_auto_new.tif'
#
# annotation_ref='/home/elisa.delaunoit/Documents/sophie/atlases/P7/ano_P7.tif'
# template_ref='/home/elisa.delaunoit/Documents/sophie/atlases/P7/normed_max_atlas_auto_new.tif'
#
annotation_ref='/home/elisa.delaunoit/Documents/sophie/atlases/P10/ano_P10.tif'
template_ref='/home/elisa.delaunoit/Documents/sophie/atlases/P10/normed_max_atlas_auto.tif'


# bspline alignement of template ADMBA on autofluo aligned good P7
align_channels_parameter = {
    # moving and reference images
    "moving_image": template_ref,
    "fixed_image": template_new,

    # elastix parameter files for alignment
    "affine_parameter_file": vars_dict['align_channels_affine_file'],
    "bspline_parameter_file": '/data/elisa.delaunoit/210811_dev_p1/210811-1b/align_bspline.txt',

    # directory of the alig'/home/nicolas.renier/Documents/ClearMap_Ressources/Par0000affine.txt',nment result
    "result_directory": directory + sub_dir +'/'+ 'new_atlas_alignement_P10'
    };
align_channels_parameter
elx.align(**align_channels_parameter);

# annotation_ref='/home/elisa.delaunoit/programs/clearmap_gui/ClearMap/Resources/Atlas/ABA_25um_annotation.tif'
annotation_new = elx.transform(
        annotation_ref, sink=[],
        transform_directory=directory + sub_dir +'/'+ 'new_atlas_alignement_P10',
        result_directory= directory + sub_dir +'/'+ 'new_anotation_P10'
        )


P7_annotation = elx.transform(
        annotation_ref, sink=[],
        transform_directory=directory + sub_dir +'/'+ 'elastix_P4_to_P7aligned',
        result_directory= directory + sub_dir +'/'+ 'new_anotation2'
        )



### align P4 to atlases



align_channels_parameter = {
        # moving and reference images
        "moving_image": '/data/elisa.delaunoit/210811_dev_p7/210817-7a/7a_resampled_autofluorescence.tif',
        "fixed_image": '/home/elisa.delaunoit/Documents/sophie/atlases/P7/normed_max_atlas_auto_new.tif',

        # elastix parameter files for alignment
        "affine_parameter_file": '/home/elisa.delaunoit/programs/ClearMap2/ClearMap/Resources/Alignment/align_affine.txt',
        "bspline_parameter_file": '/home/elisa.delaunoit/programs/ClearMap2/ClearMap/Resources/Alignment/align_bspline.txt',

        # directory of the alig'/home/nicolas.renier/Documents/ClearMap_Ressources/Par0000affine.txt',nment result
        "result_directory": '/data/elisa.delaunoit/210811_dev_p7/210817-7a/7a_elastix_auto_to_corrected_self'
        };
align_channels_parameter
elx.align(**align_channels_parameter);







workdirs=['/home/elisa.delaunoit/Documents/sophie/atlases/P1',
          '/home/elisa.delaunoit/Documents/sophie/atlases/P3',
          '/home/elisa.delaunoit/Documents/sophie/atlases/P5',
          '/home/elisa.delaunoit/Documents/sophie/atlases/P7']
templateP4='/data/elisa.delaunoit/ADMBA/P4/template_halfbrain__-1_2_3__slice_None_None_None__slice_None_None_None__slice_0_246_None__.tif'

for i , wd in enumerate(workdirs):
    reference_file=workdirs[i]+'/normed_max_atlas_auto_new.tif'
    annotation_file=workdirs[i]+'/ano_P5_good.tif'
    resample_shape=

    # affine alignement of refP7 on ADMBA
    align_channels_parameter = {
        # moving and reference images
        "moving_image": templateP4,
        "fixed_image": refTP,

        # elastix parameter files for alignment
        "affine_parameter_file": '/home/elisa.delaunoit/programs/ClearMap2/ClearMap/Resources/Alignment/align_affine.txt',
        "bspline_parameter_file": '/home/elisa.delaunoit/programs/ClearMap2/ClearMap/Resources/Alignment/align_bspline.txt',

        # directory of the alig'/home/nicolas.renier/Documents/ClearMap_Ressources/Par0000affine.txt',nment result
        "result_directory": workdirs[i] +'/'+ 'elastix_P4_to_timepoint'
        };
    align_channels_parameter
    elx.align(**align_channels_parameter);



