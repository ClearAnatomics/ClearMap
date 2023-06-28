
resources_directory = settings.resources_path
align_channels_affine_file = io.join(resources_directory, 'Alignment/align_affine_landmarks.txt')
align_reference_affine_file = io.join(resources_directory, 'Alignment/align_affine_landmarks.txt')
align_reference_bspline_file = io.join(resources_directory, 'Alignment/align_bspline_landmarks.txt')



image_list=['/data_SSD_2to/ATLASES/symetries/P5/temp.tif',
            '/data_SSD_2to/ATLASES/symetries/P5/temp.tif']
landmarks_file_paths = ['/data_SSD_2to/ATLASES/symetries/P5/test/fixed_landmarkd_pts.txt',
                        '/data_SSD_2to/ATLASES/symetries/P5/test/moving_landmarkd_pts.txt']
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
    "result_directory": '/data_SSD_2to/ATLASES/symetries/P5/test'
}


align_reference_parameter.update(moving_landmarks_path=landmarks_file_paths[1],
                                 fixed_landmarks_path=landmarks_file_paths[0])

# patch_elastix_parameter_files([align_reference_affine_file, align_reference_bspline_file])
for k, v in align_reference_parameter.items():
    if not v:
        raise ValueError('Registration missing parameter "{}"'.format(k))
elx.align(**align_reference_parameter)




aligned_nnotation = elx.transform(
    '/data_SSD_2to/ATLASES/symetries/P1/ano.tif',
    sink=[],
    transform_directory='/data_SSD_2to/ATLASES/symetries/P1/test',
    result_directory= '/data_SSD_2to/ATLASES/symetries/P1/test_ano'
)
# io.write('/data_SSD_2to/ATLASES/symetries/P5/test',aligned_nnotation)
