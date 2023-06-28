
from ClearMap.Environment import *  # analysis:ignore
import numpy as np
resources_directory = settings.resources_path
import ClearMap.IO.IO as io





directory = '/media/sophie.skriabine/sophie/auto-p14-556/5'

expression_raw = '230302_auto_11-52-21/11-52-21vasc_UltraII[<Y,2> x <X,2>]_C00.ome.npy'
expression_arteries = '211108_vasc_20-00-34/20-00-34_vasc_UltraII[<Y,2> x <X,2>]_C00.ome.npy'
expression_auto = '230323_auto_17-00-36/17-00-36_auto_UltraII_C00_xyz-Table Z<Z,4>.ome.tif'

resources_directory = settings.resources_path

ws = wsp.Workspace('TubeMap', directory=directory)#, prefix='14a');
ws.update(raw=expression_raw, arteries=expression_arteries, autofluorescence=expression_auto)
ws.info()
# Resampling
resample_parameter_auto = {
    "source_resolution": (5,5, 6),#(5.9,5.9, 6),
    "sink_resolution": ( 25, 25, 25),
    "processes": None,
    "verbose": True,
};

res.resample(ws.filename('autofluorescence'), sink=ws.filename('resampled', postfix='autofluorescence'),
             **resample_parameter_auto)



import vtk
from vtk.util.numpy_support import vtk_to_numpy
import os
import tifffile
# alignment parameter files
align_channels_affine_file = io.join(resources_directory, 'Alignment/align_affine.txt')
align_reference_affine_file = io.join(resources_directory, 'Alignment/align_affine.txt')
align_reference_bspline_file = io.join(resources_directory, 'Alignment/align_bspline.txt')

directory='/data_SSD_2to/ATLASES/autofluo25/P21'
# reference='/data_SSD_2to/ATLASES/autofluo25/P7/reference_new.tif'
directory = '/media/sophie.skriabine/sophie/auto-p14-556'
reference='/media/sophie.skriabine/sophie/auto-p14-556/reference_p14.tif'

for rootdir, dirs, files in os.walk(directory):
    dirs=['1', '2', '3', '4', '5']
    for subdir in dirs:
        try:
            print(io.join(directory,subdir))
            fluoR=io.read(io.join(directory,subdir+'/resampled_autofluorescence_R.tif'))
            fluoR=np.flip(np.swapaxes(fluoR, 0,2), 2)
            io.write(io.join(directory,subdir+'/resampled_autofluorescence_R_sagital.tif'), fluoR)
            fluoR=io.join(directory,subdir+'/resampled_autofluorescence_R_sagital.tif')
        except:
            print('no right hemisphere')

        try:
            fluoL=io.read(io.join(directory,subdir+'/resampled_autofluorescence_L.tif'))
            fluoL=np.swapaxes(fluoL, 0,2)
            io.write(io.join(directory,subdir+'/resampled_autofluorescence_L_sagital.tif'), fluoL)
            fluoL=io.join(directory,subdir+'/resampled_autofluorescence_L_sagital.tif')
        except:
            print('no left hemisphere')
        fluoL=io.join(directory,subdir+'/resampled_autofluorescence_L_sagital.tif')
        fluoR=io.join(directory,subdir+'/resampled_autofluorescence_R_sagital.tif')
        try:
            print('aligning right hemisphere...')

            align_channels_parameter = {
                # moving and reference images
                "moving_image": fluoR,
                "fixed_image": reference,

                # elastix parameter files for alignment
                "affine_parameter_file": align_reference_affine_file,
                "bspline_parameter_file": align_reference_bspline_file,

                # directory of the alig'/home/nicolas.renier/Documents/ClearMap_Ressources/Par0000affine.txt',nment result
                "result_directory": directory + '/'+subdir +'/'+ 'elastix_auto_to_chosen_auto_R'
            }
            elx.align(**align_channels_parameter);
            # elx.deformation_field(transform_directory = align_channels_parameter[ "result_directory"],
            #                       result_directory =directory + '/'+subdir +'/'+ 'elastix_def_filed_R')
            # f_name=directory + '/'+subdir +'/'+ 'elastix_def_filed_R'+'/deformationField.mhd'
            # imr = vtk.vtkMetaImageReader()
            # imr.SetFileName(f_name)
            # imr.Update()
            # im = imr.GetOutput()
            # rows, cols, z = im.GetDimensions()
            # sc = im.GetPointData().GetScalars()
            # arr = vtk_to_numpy(sc)
            # arr = arr.reshape(z, cols, rows, 3)
            # arr = arr.swapaxes(0, 2)
            # # tifffile.imsave(directory + '/'+subdir +'/'+ 'elastix_def_filed_R'+'/deformationField.tif', np.abs(np.swapaxes(arr, 0,2)).astype('uint8'),  photometric='rgb',imagej=True)
            # tifffile.imsave(directory + '/'+subdir +'/'+ 'elastix_def_filed_R'+'/deformationFieldnorm.tif', np.linalg.norm(np.swapaxes(arr, 0,2),axis=3).astype('uint8'))

        except:
            print('no right hemisphere')

        try:
            print('aligning left hemisphere...')

            align_channels_parameter = {
                # moving and reference images
                "moving_image": fluoL,
                "fixed_image": reference,

                # elastix parameter files for alignment
                "affine_parameter_file": align_reference_affine_file,
                "bspline_parameter_file": align_reference_bspline_file,

                # directory of the alig'/home/nicolas.renier/Documents/ClearMap_Ressources/Par0000affine.txt',nment result
                "result_directory": directory +'/'+  subdir +'/'+ 'elastix_auto_to_chosen_auto_L'
            }
            elx.align(**align_channels_parameter);
            # elx.deformation_field(transform_directory = align_channels_parameter[ "result_directory"],
            #                       result_directory =directory + '/'+subdir +'/'+ 'elastix_def_filed_L')
            # f_name=directory + '/'+subdir +'/'+ 'elastix_def_filed_L'+'/deformationField.mhd'
            # imr = vtk.vtkMetaImageReader()
            # imr.SetFileName(f_name)
            # imr.Update()
            # im = imr.GetOutput()
            # rows, cols, z = im.GetDimensions()
            # sc = im.GetPointData().GetScalars()
            # arr = vtk_to_numpy(sc)
            # arr = arr.reshape(z, cols, rows, 3)
            # arr = arr.swapaxes(0, 2)
            # # tifffile.imsave(directory + '/'+subdir +'/'+ 'elastix_def_filed_L'+'/deformationField.tif', np.abs(np.swapaxes(arr, 0,2)).astype('uint8'),  photometric='rgb',imagej=True)
            # tifffile.imsave(directory + '/'+subdir +'/'+ 'elastix_def_filed_L'+'/deformationFieldnorm.tif', np.linalg.norm(np.swapaxes(arr, 0,2),axis=3).astype('uint8'))
        except:
            print('no left hemisphere')

## averaging the field deformation files
k=0
directory_align='/media/sophie.skriabine/mercury/Elisa/autofluo25/P21/'
for rootdir, dirs, files in os.walk(directory_align):
    for subdir in dirs:
        print(subdir)
        if subdir!='aligned_auto':
            try:
                defL=io.read(directory_align + '/'+subdir +'/'+ 'elastix_def_filed_L'+'/deformationFieldnorm.tif')
                defR=io.read(directory_align + '/'+subdir +'/'+ 'elastix_def_filed_R'+'/deformationFieldnorm.tif')
                if k == 0:
                    condensed = defR[:, :, :, np.newaxis]
                    condensed = np.concatenate((condensed, defL[:, :, :, np.newaxis]), axis=3)
                    k=1
                else:
                    condensed = np.concatenate((condensed, defR[:, :, :, np.newaxis]), axis=3)
                    condensed = np.concatenate((condensed, defL[:, :, :, np.newaxis]), axis=3)
            except:
                print('no deformation file found')

avg_file= np.median(condensed, axis=3)
io.write(directory_align + '/' + 'avg_deform_field.tif', avg_file.astype('float32'))



from skimage.exposure import match_histograms,rescale_intensity

list_file = []
ref = io.read(reference)
directory_align='/data_SSD_2to/ATLASES/autofluo25/P21/aligned_auto'
root_only=True
for rootdir, dirs, files in os.walk(directory_align):
    if root_only:
        root_only=False
        for k, f in enumerate(files):
            img = io.read(io.join(directory_align, f))
            img=rescale_intensity(img, out_range=(np.min(ref), np.max(ref)))
            file=img

            if k == 0:
                condensed = file[:, :, :, np.newaxis]
            else:
                condensed = np.concatenate((condensed, file[:, :, :, np.newaxis]), axis=3)

    avg_file= np.mean(condensed, axis=3)
    io.write(directory_align + '/' + 'normed_max_atlas_auto_new_new.tif', avg_file.astype('float32'))

    def reject_outliers(data, m=2.5):
        data[abs(data - np.mean(data)) > m * np.std(data)]=np.NaN
        return data

    avg_n_outliers_file= np.nanmean(reject_outliers(condensed.flatten()).reshape(condensed.shape), axis=3)
    io.write(directory_align + '/' + 'normed_max_atlas_auto_new_outliers.tif', avg_n_outliers_file.astype('float32'))

