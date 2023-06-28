ref_file='/data/elisa.delaunoit/gordus/C1-21_09_16_Sp_488 ChAT - 5_cropped.tif'
file='/data/elisa.delaunoit/gordus/new_brains/Brain4 - 22.4.21 Ud_AstA 488 - Synapsin rabbit 555 - 25x OIL_cropped.tif'
directory='/data/elisa.delaunoit/gordus/new_brains/'


resample_parameter_auto = {
    "source_resolution" : (0.6668,0.6668,2.5),
    "sink_resolution"   : (5,5,5),
    "processes" : None,
    "verbose" : True,
    };

res.resample(ref_file, sink=ref_file[:-4]+'downsampled.tif', **resample_parameter_auto)
# "source_resolution" : (0.2965,0.2965,1.5),
resample_parameter_auto = {
    "source_resolution": (0.6668, 0.6668, 1.269),
    "sink_resolution"   : (5,5,5),
    "processes" : None,
    "verbose" : True,
    };

res.resample(file, sink=file[:-4]+'downsampled.tif', **resample_parameter_auto)

# bspline alignement of template ADMBA on autofluo aligned good P7
align_channels_parameter = {
    # moving and reference images
    "moving_image": file[:-4]+'downsampled.tif',
    "fixed_image": ref_file[:-4]+'downsampled.tif',

    # elastix parameter files for alignment
    "affine_parameter_file": align_channels_affine_file,
    "bspline_parameter_file": '/data/elisa.delaunoit/210811_dev_p1/210811-1b/align_bspline.txt',

    # directory of the alig'/home/nicolas.renier/Documents/ClearMap_Ressources/Par0000affine.txt',nment result
    "result_directory": directory +'/'+ 'elastix_aligned'
    };
align_channels_parameter
elx.align(**align_channels_parameter);

