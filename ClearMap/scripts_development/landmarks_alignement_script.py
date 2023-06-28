import os

import numpy as np
import pyqtgraph as pg

from ClearMap.IO.elastix_config import ElastixParser
from ClearMap.gui.widgets import LandmarksSelectorDialog, Scatter3D
from ClearMap.Environment import *  # analysis:ignore

resources_directory = settings.resources_path
align_channels_affine_file = io.join(resources_directory, 'Alignment/align_affine_landmarks.txt')
align_reference_affine_file = io.join(resources_directory, 'Alignment/align_affine_landmarks.txt')
align_reference_bspline_file = io.join(resources_directory, 'Alignment/align_bspline_landmarks.txt')

def patch_elastix_parameter_files(elastix_files):
    for f_path in elastix_files:
        cfg = ElastixParser(f_path)
        cfg['Registration'] = ['MultiMetricMultiResolutionRegistration']
        cfg['Metric'] = ["AdvancedMattesMutualInformation", "CorrespondingPointsEuclideanDistanceMetric"]
        cfg.write()


def restore_elastix_parameter_files(elastix_files):
    for f_path in elastix_files:
        cfg = ElastixParser(f_path)
        cfg['Registration'] = ['MultiResolutionRegistration']
        cfg['Metric'] = ["AdvancedMattesMutualInformation"]
        cfg.write()



def writecoords():
    global landmark_selector
    global landmarks_file_paths


    markers = [mrkr for mrkr in landmark_selector.coords if all(mrkr)]
    for i, f_path in enumerate(landmarks_file_paths):
        if not os.path.exists(os.path.dirname(f_path)):
            os.mkdir(os.path.dirname(f_path))
        with open(f_path, 'w') as landmarks_file:
            landmarks_file.write(f'point\n{len(markers)}\n')
            for marker in markers:
                x, y, z = marker[i]
                landmarks_file.write(f'{x} {y} {z}\n')
    landmark_selector.dlg.close()
    landmark_selector = None


directory='/data_SSD_2to/IRM/P14_T2/3'

resample_parameter_auto = {
    "source_resolution": (100,100, 100),
    "sink_resolution": ( 25, 25, 25),
    "processes": None,
    "verbose": True,
};

res.resample(directory+'/1.tif', sink=directory+'/resampled_allMRI_oriented_sagital_cropped.tif',
             **resample_parameter_auto)






image_list=[directory  +'/'+'resampled_allMRI_oriented_sagital_cropped.tif',
            '/data_SSD_2to/ATLASES/gamma/P14_full_template_sagital_scaled.tif']
landmarks_file_paths = [directory  +'/'+ 'elastix_auto_to_chosen_auto_allMRI_landmarks_new/fixed_landmarkd_pts.txt',
                        directory  +'/'+ 'elastix_auto_to_chosen_auto_allMRI_landmarks_new/moving_landmarkd_pts.txt']
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




align_reference_parameter={
    # moving and reference images
    "fixed_image": image_list[0],
    "moving_image": image_list[1],

    # elastix parameter files for alignment
    "affine_parameter_file": align_reference_affine_file,
    "bspline_parameter_file": align_reference_bspline_file,

    # directory of the alig'/home/nicolas.renier/Documents/ClearMap_Ressources/Par0000affine.txt',nment result
    "result_directory": directory  +'/'+ 'elastix_auto_to_chosen_auto_allMRI_landmarks_new'
}


align_reference_parameter.update(moving_landmarks_path=landmarks_file_paths[1],
                                 fixed_landmarks_path=landmarks_file_paths[0])

# patch_elastix_parameter_files([align_reference_affine_file, align_reference_bspline_file])
for k, v in align_reference_parameter.items():
    if not v:
        raise ValueError('Registration missing parameter "{}"'.format(k))
elx.align(**align_reference_parameter)
# restore_elastix_parameter_files([align_reference_affine_file, align_reference_bspline_file])



############## in case secong alignement is needed

directory='/data_SSD_2to/IRM/P14_T2/3'

image_list=[directory  +'/'+'resampled_allMRI_oriented_sagital_cropped.tif',
            directory  +'/'+'resampled_allMRI_oriented_sagital_cropped_affine_aligned.tif']
landmarks_file_paths = [directory  +'/'+ 'elastix_auto_to_chosen_auto_allMRI_bspline_landmarks/fixed_landmarkd_pts_bspline.txt',
                        directory  +'/'+ 'elastix_auto_to_chosen_auto_allMRI_bspline_landmarks/moving_landmarkd_pts_bspline.txt']
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



align_reference_parameter={
    # moving and reference images
    "fixed_image": image_list[0],
    "moving_image": image_list[1],

    # elastix parameter files for alignment
    "affine_parameter_file": align_reference_affine_file,
    "bspline_parameter_file": align_reference_bspline_file,

    # directory of the alig'/home/nicolas.renier/Documents/ClearMap_Ressources/Par0000affine.txt',nment result
    "result_directory": directory  +'/'+ 'elastix_auto_to_chosen_auto_allMRI_bspline_landmarks'
}


align_reference_parameter.update(moving_landmarks_path=landmarks_file_paths[1],
                                 fixed_landmarks_path=landmarks_file_paths[0])

# patch_elastix_parameter_files([align_reference_affine_file, align_reference_bspline_file])
for k, v in align_reference_parameter.items():
    if not v:
        raise ValueError('Registration missing parameter "{}"'.format(k))
elx.align(**align_reference_parameter)







directory = '/data_SSD_2to/IRM/P12_T2/1'
#aligning the atlas
annotation_file='/data_SSD_2to/ATLASES/FINAL/P12/sym_ano_sagital.tif'
aligned_nnotation = elx.transform(
    annotation_file, sink=[],
    transform_directory=directory  +'/'+ 'elastix_auto_to_chosen_auto_allMRI_landmarks',
    result_directory= directory +'/'+ 'ano_to_MRI'
)
# io.write(directory +'/'+ 'ano_to_MRI_aligned_anotation_0.tif',aligned_nnotation)

aligned_nnotation = elx.transform(
    directory +'/'+ 'ano_to_MRI_0/result.tif',
    sink=[],
    transform_directory=directory  +'/'+ 'elastix_auto_to_chosen_auto_allMRI_bspline_landmarks',
    result_directory= directory +'/'+ 'ano_to_MRI'
)
io.write(directory +'/'+ 'ano_to_MRI_aligned_anotation_1.tif',aligned_nnotation)







#aligning the MRI

directory = '/data_SSD_2to/IRM/P07_T2/2'
resample_parameter_auto = {
    "source_resolution": (100,100, 100),
    "sink_resolution": ( 25, 25, 25),
    "processes": None,
    "verbose": True,
};

res.resample(directory+'/1.tif', sink=directory+'/resampled_allMRI_oriented_sagital_cropped.tif',
             **resample_parameter_auto)



directory='/data_SSD_2to/IRM/P12_T2/3'
reference='/data_SSD_2to/IRM/P12_T2/1/resampled_allMRI_oriented_sagital_cropped.tif'

align_reference_affine_file = io.join(resources_directory, 'Alignment/align_affine.txt')
align_reference_bspline_file = io.join(resources_directory, 'Alignment/align_bspline.txt')

align_reference_parameter={
    # moving and reference images
    "fixed_image": directory+'/resampled_allMRI_oriented_sagital_cropped.tif',
    "moving_image": reference,

    # elastix parameter files for alignment
    "affine_parameter_file": align_reference_affine_file,
    "bspline_parameter_file": align_reference_bspline_file,

    # directory of the alig'/home/nicolas.renier/Documents/ClearMap_Ressources/Par0000affine.txt',nment result
    "result_directory": directory  +'/'+ 'elastix_MRIref_to_MRI'
}

elx.align(**align_reference_parameter)

directory='/data_SSD_2to/IRM/P12_T2/3'
annotation_file='/data_SSD_2to/IRM/P12_T2/1/ano_to_MRI/result.tif'
elx.transform(
    annotation_file, sink=[],
    transform_directory=directory  +'/'+ 'elastix_MRIref_to_MRI',
    result_directory= directory +'/'+ 'ano_to_MRI'
)