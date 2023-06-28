from ClearMap.Environment import *  #analysis:ignore
import sys
import numpy as np
# from ClearMap.IO.MHD import mhd_read

def get_workspace(script_path, match_str):
     with open(script_path, 'r') as script:
         code_lines = script.readlines()

     start_idx = [i for i, ln in enumerate(code_lines) if match_str in ln][0]
     code_str = ''.join(code_lines[:start_idx+1])  # line included
     print(code_str)
     locs = {}
     exec(code_str, globals(), locs)
     return locs


# templateP4='/data/elisa.delaunoit/ADMBA/P4/template_halfbrain.tif'
# annotation_P4='/data/elisa.delaunoit/ADMBA/ADMBA_thresholded/annotationADMBA_thresholded.tif'


#take best P7 autofluo
#goodP4
# directory='/data/elisa.delaunoit/210811_dev_p4/goodp4/'
# sub_dir='220111_P4t'

#P4
# directory='/data/elisa.delaunoit/210811_dev_p4/220105-test-ib4-p4/'
# sub_dir='220105-4t'

#P5
directory='/data/elisa.delaunoit/210811_dev_p5/'
sub_dir='210811-5b'
#P3
# directory='/data/elisa.delaunoit/210811_dev_p3/'
# sub_dir='210817-3d'
#P1
# directory='/data/elisa.delaunoit/210811_dev_p1/'
# sub_dir='210811-1b'
#P7
# directory='/data/elisa.delaunoit/210811_dev_p7/'
# sub_dir='210817-7a'

#P21
directory='/data/elisa.delaunoit/auto-p21/'
sub_dir='220511-3'


script_path = directory + sub_dir +'/'+sub_dir[-1:] + '-TubeMap.py'
vars_dict = get_workspace(script_path, 'align_reference_bspline_file = ')
ws = vars_dict['ws']
ws.info()

refP7=ws.filename('resampled', postfix='autofluorescence')

# templateP4=vars_dict['reference_file']
# annotation_P4=vars_dict['annotation_file']
annotation_P4='/data/elisa.delaunoit/anoP3_v23456789abcde.tif'
templateP4='/home/elisa.delaunoit/Documents/sophie/atlases/P3/normed_max_atlas_auto.tif'

#
# # affine alignement of refP7 on ADMBA
# align_channels_parameter = {
#     # moving and reference images
#     "moving_image": refP7,
#     "fixed_image": templateP4,
#
#     # elastix parameter files for alignment
#     "affine_parameter_file": vars_dict['align_channels_affine_file'],
#     "bspline_parameter_file": None,
#
#     # directory of the alig'/home/nicolas.renier/Documents/ClearMap_Ressources/Par0000affine.txt',nment result
#     "result_directory": directory + sub_dir +'/'+ 'elastix_P7_toP4'
#     };
# align_channels_parameter
# elx.align(**align_channels_parameter);
#
# expression_raw = directory + sub_dir+'/' + 'elastix_P7_toP4/result.0.mhd'
# # create raw data npy files
# mhd_file = mhd_read(expression_raw)
# io.write(expression_raw[:-3] + 'tif', mhd_file);
# alignedP7='/data/elisa.delaunoit/210811_dev_p1/210811-1b/elastix_P7_toP4/result.0.tif'





# alignedP7='/data/elisa.delaunoit/210811_dev_p1/210811-1a/1a_resampled_autofluorescence_corrected_sagital.tif'
# alignedP7='/data/elisa.delaunoit/210811_dev_p7/210817-7a/7a_resampled_autofluorescence_corrected_sagital.tif'
# alignedP7='/data/elisa.delaunoit/210811_dev_p3/210817-3d/3d_resampled_autofluorescence_corrected_sagital.tif'
alignedP7='/data/elisa.delaunoit/210811_dev_p5/210811-5b/5b_resampled_autofluorescence_corrected_sagital.tif'
# alignedP7='/data/elisa.delaunoit/210811_dev_p4/220105-test-ib4-p4/220105-4t/4t_resampled_autofluorescence_corrected_sagital.tif'
# alignedP7='/data/elisa.delaunoit/210811_dev_p4/goodp4/220111_P4t/4_resampled_autofluorescence_corrected_sagital.tif'


# bspline alignement of template ADMBA on autofluo aligned good P7
align_channels_parameter = {
    # moving and reference images
    "moving_image": templateP4,
    "fixed_image": alignedP7,

    # elastix parameter files for alignment
    "affine_parameter_file": '/data/elisa.delaunoit/210811_dev_p1/210811-1b/align_affine.txt',
    "bspline_parameter_file": '/data/elisa.delaunoit/210811_dev_p1/210811-1b/align_bspline.txt',

    # directory of the alig'/home/nicolas.renier/Documents/ClearMap_Ressources/Par0000affine.txt',nment result
    "result_directory": directory + sub_dir +'/'+ 'elastix_P4_to_P7aligned'
    };
align_channels_parameter
elx.align(**align_channels_parameter);


#use this transformation to adapt the deform the annotation to match the P7
# align_channels_parameter = {
#     # moving and reference images
#     "fixed_image": templateP4,
#
#     # elastix parameter files for alignment
#     "affine_parameter_file": vars_dict['align_channels_affine_file'],
#     "bspline_parameter_file": vars_dict['align_reference_bspline_file'],
#     "transform_directory" : directory + sub_dir +'/'+ 'elastix_P4_to_P7aligned',
#     # directory of the alig'/home/nicolas.renier/Documents/ClearMap_Ressources/Par0000affine.txt',nment result
#     "result_directory": directory + sub_dir +'/'+ 'anoP4_to_anoP7',Z
#     "processes":9
#     };
# align_channels_parameter
# elx.inverse_transform(**align_channels_parameter);

P7_annotation = elx.transform(
        annotation_P4, sink=[],
        transform_directory=directory + sub_dir +'/'+ 'elastix_P4_to_P7aligned',
        result_directory= directory + sub_dir +'/'+ 'anoP4_to_anoP7'
        )



# expression_raw = directory + sub_dir+'/' + 'anoP4_to_anoP7/result.mhd'
# P7_annotation = mhd_read(expression_raw)
# io.write(expression_raw[:-3] + 'tif', P7_annotation);





###aligne other fluorescence on the good one

directory='/data/elisa.delaunoit/auto-p21/'
sub_dir='p21-4'
script_path = directory + sub_dir +'/'+sub_dir[-4:] + '-TubeMap.py'
vars_dict = get_workspace(script_path, 'align_reference_bspline_file = ')
ws = vars_dict['ws']
ws.info()

refP7='/data/elisa.delaunoit/auto-p21/p21-4/214_resampled_autofluorescence.tif'
#ws.filename('resampled', postfix='autofluorescence')#'/data/elisa.delaunoit/p10/11/11_resampled_autofluorescence.tif'#

templateP4=vars_dict['reference_file']
annotation_P4=vars_dict['annotation_file']


autofluo=refP7
# reference='/data/elisa.delaunoit/210811_dev_p1/210811-1b/1b_resampled_autofluorescence_corrected_sagital.tif'
# reference='/data/elisa.delaunoit/210811_dev_p7/210817-7a/7a_resampled_autofluorescence_corrected_sagital.tif'
# reference='/data/elisa.delaunoit/210811_dev_p3/210817-3d/3d_resampled_autofluorescence_corrected_sagital.tif'
# reference='/data/elisa.delaunoit/210811_dev_p5/220131-mbp/mbp_resampled_autofluorescence_corrected_sagital.tif'
# reference='/data/elisa.delaunoit/210811_dev_p14/210811-14c/14c_resampled_autofluorescence_corrected_sagital.tif'
# reference='/data/elisa.delaunoit/p10/13/13_resampled_autofluorescence_corrected_sagital.tif'
reference='/data/elisa.delaunoit/auto-p21/p21-mbp/21mbp_resampled_autofluorescence.tif'

# bspline alignement of template ADMBA on autofluo aligned good P7
# directory='/data/elisa.delaunoit/p5-auto/'
# sub_dir='ICI'
# autofluo='/data/elisa.delaunoit/p5-auto/ICI/autofluo_sagital_downsampled.tif'
align_channels_parameter = {
    # moving and reference images
    "moving_image": autofluo,
    "fixed_image": reference,

    # elastix parameter files for alignment
    "affine_parameter_file": '/home/elisa.delaunoit/programs/ClearMap2/ClearMap/Resources/Alignment/align_affine.txt',
    "bspline_parameter_file": '/home/elisa.delaunoit/programs/ClearMap2/ClearMap/Resources/Alignment/align_bspline.txt',

    # directory of the alig'/home/nicolas.renier/Documents/ClearMap_Ressources/Par0000affine.txt',nment result
    "result_directory": directory + sub_dir +'/'+ 'elastix_auto_to_chosen_auto'
    };
align_channels_parameter
elx.align(workspace=ws, **align_channels_parameter);




#####atlas region transfer
old_aud_mot=io.read('/home/elisa.delaunoit/Documents/sophie/P4ADMA/3Drecon-ADMBA-P4_annotation-1_audbarmot-1.tif')
clean_ano=io.read('/home/elisa.delaunoit/Documents/sophie/P4ADMA/3Drecon-ADMBA-P4_annotation.tif')
clean_ano[np.where(old_aud_mot==17738)]=17738
clean_ano[np.where(old_aud_mot==17740)]=17740
clean_ano[np.where(old_aud_mot==1)]=1
p3d.plot(clean_ano)
io.write('/home/elisa.delaunoit/Documents/sophie/P4ADMA/3Drecon-ADMBA-P4_annotation-1_audbarmot.tif',clean_ano)


### fusion autofluorescences

import numpy as np
from scipy import interpolate


import matplotlib.pyplot as plt

from skimage import data
from skimage import exposure
from skimage.exposure import match_histograms,rescale_intensity



# dir_list=['/data/elisa.delaunoit/210811_dev_p1',
#           '/data/elisa.delaunoit/210811_dev_p5',
#           '/data/elisa.delaunoit/210811_dev_p7',
#           '/data/elisa.delaunoit/210811_dev_p3',
#           '/data/elisa.delaunoit/210811_dev_p14',
#           '/data/elisa.delaunoit/p10']
#
#
# sub_dir=[['210811-1a', '210811-1b', '210811-1d'],
#          ['210811-5a','210811-5b','220131-mbp'],
#          ['210817-7a', '210811-7b'],
#          ['210817-3a', '210817-3b', '210817-3c', '210817-3d'],
#          ['210811-14c', '210811-14b'],
#          ['11', '12', '13']]

# ref_list=['210811-1b','220131-mbp','210817-7a','210817-3d','210811-14c', '13']

dir_list=['/data/elisa.delaunoit/auto-p21']
sub_dir=[['p21-mbp', 'p21-1', 'p21-2', 'p21-4']]
ref_list=['p21-mbp']
# z_lim=354

for i, dl in enumerate(dir_list):
    print(dl)
    list_file = []
    ref = io.read(dl + '/' + ref_list[i] + '/' + '21mbp' + '_resampled_autofluorescence.tif')
    print(np.min(ref), np.max(ref))
    for j, sd in enumerate(sub_dir[i]):
        if sd==ref_list[i]:
            list_file.append(dl+'/'+sd+'/'+'21mbp'+'_resampled_autofluorescence.tif')
        else:
            list_file.append(dl + '/' + sd + '/' +'elastix_auto_to_chosen_auto/result.1.tif')

    for k, file_name in enumerate(list_file):
        img = io.read(file_name)
        sd=sub_dir[i][k]
        #normalization + hitogram matching
        # img = (img - np.min(img)) / (np.max(img) - np.min(img))
        if sd !=ref_list[i]:
            print(sd)
            # io.write(dl + '/' + sd[-2:] + '_intensity.tif', img)
            print(np.min(img), np.max(img))
            img=rescale_intensity(img, out_range=(np.min(ref), np.max(ref)))
            print(np.min(img), np.max(img))
            # file = match_histograms(img, ref)
            # io.write(dl + '/' + sd[-2:] + 'rescaled_intensity.tif', img)
            # io.write(dl + '/' + sd[-2:]+'matched_hist.tif', file)
            # unique, counts = np.unique(img, return_counts=True)
            # matched_values=np.zeros(unique.shape)
            # for i, u in enumerate(unique):
            #     print(i)
            #     pos=np.asarray(img==u).nonzero()
            #     x=pos[0]
            #     y=pos[1]
            #     z = pos[2]
            #     matched_values[i]=res[x, y, z].mean()
            #
            # f = interpolate.interp1d(unique, matched_values, kind='slinear', bounds_error=False, fill_value='extrapolate')
            # file = f(img)
        # else:
        #     img=img+1000
        file=img
        print(np.sum(file))
        if k == 0:
            condensed = file[:, :, :, np.newaxis]
        else:
            condensed = np.concatenate((condensed, file[:, :, :, np.newaxis]), axis=3)

    # avg_file = np.mean(condensed, axis=3)
    # avg_file=np.zeros(condensed.shape[:3])
    avg_file= np.max(condensed, axis=3)
    # print(np.min(avg_file), np.max(avg_file))
    cerebellum=rescale_intensity(np.max(condensed[:, z_lim:, :, 1:], axis=3), out_range=(np.min(avg_file[:, :z_lim, :]), np.max(avg_file[:, :z_lim, :])))
    # print(np.min(cerebellum), np.max(cerebellum))
    # avg_file[:, z_lim:, :] = cerebellum
    # print(np.min(avg_file), np.max(avg_file))
    # print(np.sum(avg_file))
    io.write(dl + '/' + 'normed_max_atlas_auto_new.tif', avg_file)


## chimera template file
from skimage.exposure import match_histograms,rescale_intensity

workdirs=['/home/elisa.delaunoit/Documents/sophie/atlases/P5']
ref=io.read('/home/elisa.delaunoit/Documents/sophie/atlases/P5/autofluo_sagital_downsampled_aligned.tif')

good_P5=io.read('/data/elisa.delaunoit/220427_test_p5auto/220427_auto_19-20-49_resampled_sagital_aligned.tif')
good_P5_intensity=match_histograms(good_P5, ref)#out_range=(np.min(ref), np.max(ref)))
good_P5_intensity[169:, :100, :]=np.min(good_P5_intensity)
test=np.concatenate((good_P5_intensity[:, :, :, np.newaxis], ref[:, :, :, np.newaxis]), axis=3)
tets_max=np.max(test, axis=3)
io.write('/home/elisa.delaunoit/Documents/sophie/atlases/P5/max_ref_goodp5.tif', tets_max)

res=np.zeros(ref.shape)
res[:, :302, :]=ref[:, :302, :]
res[:, 356:, :]=ref[:, 356:, :]

fillup=io.read('/home/elisa.delaunoit/Documents/sophie/atlases/P5/result.1.tif')
band=fillup[:,302:356, :]
band_intensity=rescale_intensity(band, out_range=(np.min(ref[:, :302, :]), np.max(ref[:, :302, :])))
band_intensity=match_histograms(band, ref[:, 302:356, :])

nocc=io.read('/home/elisa.delaunoit/Documents/sophie/atlases/P5/normed_max_atlas_auto_new.tif')
nocc_intensity=rescale_intensity(nocc[:, 302:356, :], out_range=(np.min(ref[:, 302:356, :]), np.max(ref[:, 302:356, :])))
nocc_intensity_matched=match_histograms(nocc_intensity,ref)

for i in range(nocc_intensity_matched.shape[2]):
    nocc_intensity_matched[:,:,i]=match_histograms(nocc[:, 302:356, i],ref[:, 302:356, :][:, :, i])
io.write('/home/elisa.delaunoit/Documents/sophie/atlases/P5/nocc_intensity_matched.tif', nocc_intensity_matched)


res[:,302:356,:]=nocc_intensity_matched
res[:,302:356,:]=band_intensity
io.write('/home/elisa.delaunoit/Documents/sophie/atlases/P5/chimere302_356.tif', res)


