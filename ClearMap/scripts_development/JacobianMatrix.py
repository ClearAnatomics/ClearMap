
from ClearMap.Environment import *  # analysis:ignore
import numpy as np
resources_directory = settings.resources_path
import ClearMap.IO.IO as io
import os


#
#
# align_channels_affine_file = io.join(resources_directory, 'Alignment/align_affine.txt')
# align_reference_affine_file = io.join(resources_directory, 'Alignment/align_affine.txt')
# align_reference_bspline_file = io.join(resources_directory, 'Alignment/align_bspline.txt')
#

align_channels_affine_file = io.join(resources_directory, 'Alignment/align_affine_landmarks.txt')
align_reference_affine_file = io.join(resources_directory, 'Alignment/align_affine_landmarks.txt')
align_reference_bspline_file = io.join(resources_directory, 'Alignment/align_bspline_landmarks.txt')

moving='/data_SSD_2to/ATLASES/FINAL/P9/P9_full_template_sagital.tif'
fixed='/data_SSD_2to/ATLASES/FINAL/P12/sym_ref_sagital.tif'


directory='/data_SSD_2to/ATLASES/Jacobian/p9_to_p12'

image_list=[moving,fixed]

landmarks_file_paths = [directory  +'/'+ 'elastix_auto_to_chosen_auto_allMRI_landmarks/moving_landmarkd_pts.txt',
                        directory  +'/'+ 'elastix_auto_to_chosen_auto_allMRI_landmarks/fixed_landmarkd_pts.txt']
dvs = p3d.plot(image_list,  arange=True, sync=False,
               lut=None)





landmark_selector = LandmarksSelectorDialog('', params=None)
landmark_selector.data_viewers = dvs
for i in range(2):
    scatter = pg.ScatterPlotItem()
    dvs[i].enable_mouse_clicks()
    dvs[i].view.addItem(scatter)
    dvs[i].scatter = scatter
    coords = [landmark_selector.fixed_coords(), landmark_selector.moving_coords()][i]  # FIXME: check order (A to B)
    dvs[i].scatter_coords = Scatter3D(coords, colors=np.array(landmark_selector.colors), half_slice_thickness=5)
    callback = [landmark_selector.set_fixed_coords, landmark_selector.set_moving_coords][i]
    dvs[i].mouse_clicked.connect(callback)

landmark_selector.dlg.buttonBox.accepted.connect(writecoords)




align_channels_parameter = {
    # moving and reference images
    "moving_image": moving,
    "fixed_image": fixed,
    # elastix parameter files for alignment
    "affine_parameter_file": align_reference_affine_file,
    "bspline_parameter_file": align_reference_bspline_file,
    # directory of the alig'/home/nicolas.renier/Documents/ClearMap_Ressources/Par0000affine.txt',nment result
    "result_directory": directory+'/'+ 'transfo'
}


align_channels_parameter.update(moving_landmarks_path=landmarks_file_paths[0],
                                 fixed_landmarks_path=landmarks_file_paths[1],
                                 result_directory= directory+'/'+ 'transfo_landmarks')

# patch_elastix_parameter_files([align_reference_affine_file, align_reference_bspline_file])
for k, v in align_channels_parameter.items():
    if not v:
        raise ValueError('Registration missing parameter "{}"'.format(k))
elx.align(**align_channels_parameter);

elx.jacobian_det(transform_directory = align_channels_parameter[ "result_directory"],
                 result_directory = directory+'/'+ 'jacobian_landmarks')