import pandas as pd

align_channels_affine_file = io.join(resources_directory, 'Alignment/align_affine.txt')
align_reference_affine_file = io.join(resources_directory, 'Alignment/align_affine.txt')
align_reference_bspline_file = io.join(resources_directory, 'Alignment/align_bspline.txt')



p9=io.read('/data_2to/alignement/atlases/alignement_P9/autoP9.tif')
p9=np.clip(p9, 498, None)
io.write('/data_2to/alignement/atlases/alignement_P9/autoP9_clip.tif', p9)

p7=io.read('/data_2to/alignement/atlases/alignement_P9/autoP7_upsampled.tif')
p9=np.clip(p7, 2686, None)
io.write('/data_2to/alignement/atlases/alignement_P9/autoP7_upsampled_clip.tif', p7)

align_channels_parameter = {
    # moving and reference images
    "fixed_image": '/data_2to/alignement/atlases/alignement_P9/result6R.tif',
    "moving_image": '/data_2to/alignement/atlases/alignement_P9/result3L_upsampled.tif',

    # elastix parameter files for alignment
    "affine_parameter_file": align_reference_affine_file,
    "bspline_parameter_file": align_reference_bspline_file,

    # directory of the alig'/home/nicolas.renier/Documents/ClearMap_Ressources/Par0000affine.txt',nment result
    "result_directory": '/data_2to/alignement/atlases/alignement_P9'+'/'+ 'elastix_auto_to_chosen_auto_full'
}
elx.align(**align_channels_parameter);

annotation_file='/data_2to/alignement/atlases/alignement_P9/anoP7_upsampled.tif'
aligned_nnotation = elx.transform(
    annotation_file,
    transform_directory='/data_2to/alignement/atlases/alignement_P9'+'/'+ 'elastix_auto_to_chosen_auto_full',
    result_directory= '/data_2to/alignement/atlases/alignement_P9'+'/'+ 'res_atlas'
)