

from ClearMap.Environment import *  # analysis:ignore
resources_directory = settings.resources_path
align_channels_affine_file = io.join(resources_directory, 'Alignment/align_affine.txt')
align_reference_affine_file = io.join(resources_directory, 'Alignment/align_affine.txt')
align_reference_bspline_file = io.join(resources_directory, 'Alignment/align_bspline.txt')

import cv2 as cv
import scipy.ndimage as ndimage
template=io.read('/data_SSD_2to/ATLASES/FINAL/P21/base/p21_new_template_coronal_symetric.tif')

io.write('/data_SSD_2to/ATLASES/FINAL/P21/base/brain_masked_997.tif', (template>=15000))

brain_masked=io.read('/data_SSD_2to/ATLASES/FINAL/P21/base/brain_masked_997.tif')
# struct2 = ndimage.generate_binary_structure(3, 3, 3)
brain_masked=ndimage.binary_dilation(brain_masked, iterations=3)
brain_masked=ndimage.binary_erosion(brain_masked, iterations=3)

brain_masked=brain_masked.astype('uint16')*997
annotation=io.read('/data_SSD_2to/ATLASES/FINAL/P21/p21_new_annotation_masked_NEW.tif')

# annotation=annotation-32768

brain_masked[np.asarray(annotation!=0).nonzero()]=annotation[np.asarray(annotation!=0).nonzero()]
# p3d.plot(brain_masked)
#
#
# labels=[1, 2, 3, 5, 6, 7, 35, 64, 83, 170, 178, 186, 255, 262, 301, 351, 398, 429, 449, 475, 483, 603, 661, 662, 733, 776, 797, 908, 1099]
#
# for l in labels:
#     print(l)
#     brain_masked[np.asarray(brain_masked==l).nonzero()]=997
# p3d.plot(brain_masked)


brain_masked[np.asarray(brain_masked==385).nonzero()]=669
# p3d.plot(brain_masked)

brain_masked[np.asarray(brain_masked==81).nonzero()]=73
# p3d.plot(brain_masked)

brain_masked[np.asarray(brain_masked==10969).nonzero()]=997
# p3d.plot(brain_masked)

# io.write('/data_SSD_2to/ATLASES/FINAL/P21/p21_new annotation_masked_NEW.nrrd', brain_masked)

brain_masked[np.asarray(brain_masked==10969).nonzero()]=997
io.write('/data_SSD_2to/ATLASES/FINAL/P21/p21_new annotation_masked_NEW.tif', brain_masked)


new_annotation='/data_SSD_2to/ATLASES/NICOLAS/P3/P3_full_annotation_sagital.tif'
template='/data_SSD_2to/ATLASES/NICOLAS/P3/P3_full_template_sagital.nrrd'
target_template='/data_SSD_2to/ATLASES/FINAL/P5/P5_full_template_sagital.tif'


align_channels_parameter = {
    # moving and reference images
    "moving_image": template,
    "fixed_image": target_template,

    # elastix parameter files for alignment
    "affine_parameter_file": align_reference_affine_file,
    "bspline_parameter_file": align_reference_bspline_file,

    # directory of the alig'/home/nicolas.renier/Documents/ClearMap_Ressources/Par0000affine.txt',nment result
    "result_directory": '/data_SSD_2to/ATLASES/NICOLAS/P5/' + 'elastix_auto_03_to_05'
}
elx.align(**align_channels_parameter);


# new_annotation='/data_SSD_2to/ATLASES/NICOLAS/P12/Combined_Stacks_annotation_toalignonp9.tif'
aligned_nnotation = elx.transform(
    new_annotation,
    sink=[],
    transform_directory='/data_SSD_2to/ATLASES/NICOLAS/P5/'+ 'elastix_auto_03_to_05',
    result_directory= '/data_SSD_2to/ATLASES/NICOLAS/P5/'+ 'res_dir'
)
