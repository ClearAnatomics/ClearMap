import pandas as pd
import os

import numpy as np
import pyqtgraph as pg

from ClearMap.IO.elastix_config import ElastixParser
from ClearMap.gui.widgets import LandmarksSelectorDialog, Scatter3D
from ClearMap.Environment import *  # analysis:ignore

resources_directory = settings.resources_path

align_channels_affine_file = io.join(resources_directory, 'Alignment/align_affine.txt')
align_reference_affine_file = io.join(resources_directory, 'Alignment/align_affine.txt')
align_reference_bspline_file = io.join(resources_directory, 'Alignment/align_bspline.txt')

directory='/data_SSD_2to/FUS'
arteries='/data_SSD_2to/FUS/2_resampled_arteries.tif'
autofluo='/data_SSD_2to/FUS/2_resampled_autofluorescence10.tif'
annotation='/data_SSD_2to/FUS/2_resampled_annotation-10-10-10.tif'
template='/data_SSD_2to/FUS/2_template_10_10_10.tif'

align_channels_parameter = {
    # moving and reference images
    "moving_image": arteries,
    "fixed_image": autofluo,

    # elastix parameter files for alignment
    "affine_parameter_file": align_reference_affine_file,
    "bspline_parameter_file": align_reference_bspline_file,

    # directory of the alig'/home/nicolas.renier/Documents/ClearMap_Ressources/Par0000affine.txt',nment result
    "result_directory": directory  +'/'+ 'art_to_fluo'
}
elx.align(**align_channels_parameter);


align_channels_parameter = {
    # moving and reference images
    "moving_image": autofluo,
    "fixed_image": template,

    # elastix parameter files for alignment
    "affine_parameter_file": align_reference_affine_file,
    "bspline_parameter_file": align_reference_bspline_file,

    # directory of the alig'/home/nicolas.renier/Documents/ClearMap_Ressources/Par0000affine.txt',nment result
    "result_directory": directory  +'/'+ 'fluo_to_temp'
}
elx.align(**align_channels_parameter);

aligned_nnotation = elx.transform(
    '/data_SSD_2to/FUS/art_to_fluo/result.1.tif',
    sink=[],
    transform_directory=directory  +'/'+ 'fluo_to_temp',
    result_directory= directory   +'/'+ 'art_to_temp'
)



############# FUS to iDISCO alignement

directory='/data_SSD_2to/FUS'

# testdisco=io.read('/data_SSD_2to/FUS/iDISCOaligned_32bits.tif')
# testr=io.read('/data_SSD_2to/FUS/DensityOnAllen_25_32bits.tif')
# image_list=[testdisco, testr]
image_list=[ '/data_SSD_2to/FUS/iDISCOaligned_32bits.tif','/data_SSD_2to/FUS/DensityOnAllen_25_32bits.tif']

landmarks_file_paths = [directory  +'/'+ 'fus_2_disco/fixed_landmarkd_pts_bspline.txt',
                        directory  +'/'+ 'fus_2_disco/moving_landmarkd_pts_bspline.txt']
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

image_list=[ '/data_SSD_2to/FUS/iDISCOaligned_32bits.tif','/data_SSD_2to/FUS/DensityOnAllen_25_32bits.tif']

align_reference_parameter={
    # moving and reference images
    "fixed_image": image_list[0],
    "moving_image": image_list[1],

    # elastix parameter files for alignment
    "affine_parameter_file": align_reference_affine_file,
    "bspline_parameter_file": align_reference_bspline_file,

    # directory of the alig'/home/nicolas.renier/Documents/ClearMap_Ressources/Par0000affine.txt',nment result
    "result_directory": directory  +'/'+ 'fus_2_disco'
}


align_reference_parameter.update(moving_landmarks_path=landmarks_file_paths[1],
                                 fixed_landmarks_path=landmarks_file_paths[0])

# patch_elastix_parameter_files([align_reference_affine_file, align_reference_bspline_file])
for k, v in align_reference_parameter.items():
    if not v:
        raise ValueError('Registration missing parameter "{}"'.format(k))
elx.align(**align_reference_parameter)
