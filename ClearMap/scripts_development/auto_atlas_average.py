from ClearMap.Environment import *  #analysis:ignore
import sys
import numpy as np

from ClearMap.IO.MHD import mhd_read

ano.initialize(label_file = '/home/elisa.delaunoit/programs/ClearMap2/ClearMap/Resources/Atlas/region_ids_test_ADMBA.json',
               extra_label = [], annotation_file = '/data/elisa.delaunoit/ADMBA/ADMBA_thresholded/annotationADMBA_thresholded.tif')

#init atlas and reference files
annotation_file, reference_file, distance_file = ano.prepare_annotation_files(  
    slicing=(slice(None),slice(None),slice(0,246)), orientation=(-1,2,3),
    verbose=True, annotation_file='/data/elisa.delaunoit/ADMBA/ADMBA_thresholded/annotationADMBA_thresholded.tif',
    reference_file='/data/elisa.delaunoit/ADMBA/P4/template_halfbrain.tif');


def get_workspace(script_path, match_str):
     with open(script_path, 'r') as script:
         code_lines = script.readlines()

     start_idx = [i for i, ln in enumerate(code_lines) if match_str in ln][0]
     code_str = ''.join(code_lines[:start_idx+1])  # line included
     print(code_str)
     locs = {}
     exec(code_str, globals(), locs)
     return locs

dir_list=['/data/elisa.delaunoit/210811_dev_p1',
          '/data/elisa.delaunoit/210811_dev_p3',
          '/data/elisa.delaunoit/210811_dev_p5',
          '/data/elisa.delaunoit/210811_dev_p7']


sub_dir=[['210811-1a', '210811-1b', '210811-1d'],
         ['210817-3a','210817-3b', '210817-3c','210817-3d'],
         ['210811-5a','210811-5b'],
         ['210817-7a', '210811-7b']]




## elastic alignment

for i, main_directory in enumerate(dir_list):
    list_file = []
    for j, sub_directory in enumerate(sub_dir[i]):
        directory = main_directory + '/' + sub_dir[i][j] + '/'
        print(directory)
        script_path = directory + sub_dir[i][j][-2:] + '-TubeMap.py'
        vars_dict = get_workspace(script_path, 'align_reference_bspline_file = ')
        ws = vars_dict['ws']
        ws.info()
        # % Aignment
        # align the two channels
        align_channels_parameter = {
            # moving and reference images
            "moving_image": ws.filename('resampled', postfix='autofluorescence'),
            "fixed_image": reference_file,

            # elastix parameter files for alignment
            "affine_parameter_file": vars_dict['align_channels_affine_file'],
            "bspline_parameter_file": vars_dict['align_reference_bspline_file'],

            # directory of the alig'/home/nicolas.renier/Documents/ClearMap_Ressources/Par0000affine.txt',nment result
            "result_directory": directory + 'elastix_resampled_to_auto_precise'
        };
        align_channels_parameter
        elx.align(**align_channels_parameter);

        expression_raw = directory + 'elastix_resampled_to_auto_precise/result.1.mhd'
        # create raw data npy files
        # mhd_file = mhd_read(expression_raw)
        # io.write(expression_raw[:-3] + 'tif', mhd_file);
        # list_file.append(expression_raw[:-3] + 'tif')


#%##  rigid alignment to template
for i, main_directory in enumerate(dir_list):
    list_file=[]
    for j , sub_directory in enumerate(sub_dir[i]):
        directory=main_directory+'/'+sub_dir[i][j]+'/'
        print(directory)
        script_path = directory + sub_dir[i][j][-2:] + '-TubeMap.py'
        vars_dict = get_workspace(script_path, 'align_reference_bspline_file = ')
        ws = vars_dict['ws']
        ws.info()
        #% Aignment
        # align the two channels
        align_channels_parameter = {            
            #moving and reference images
            "moving_image" : ws.filename('resampled', postfix='autofluorescence'),
            "fixed_image"  : reference_file,
            
            #elastix parameter files for alignment
            "affine_parameter_file"  : vars_dict['align_channels_affine_file'],
            "bspline_parameter_file" : None,
            
            #directory of the alig'/home/nicolas.renier/Documents/ClearMap_Ressources/Par0000affine.txt',nment result
            "result_directory" :  directory+'elastix_resampled_to_auto_inv' 
            }; 
        align_channels_parameter
        elx.align(**align_channels_parameter);
        
        expression_raw      = directory+'elastix_resampled_to_auto_inv/result.0.mhd'      
        #create raw data npy files                  
        mhd_file=mhd_read(expression_raw)
        io.write(expression_raw[:-3]+'tif', mhd_file);
        list_file.append(expression_raw[:-3]+'tif')
    
    
    
    print('compute the avg autofluorescence file')
    print(list_file)
    #directories and files
    avg_directory = directory+'/average_auto_atlas'

    for k, file_name in enumerate(list_file):
        file=io.read(file_name)
        if k ==0:
            condensed=file[:, :, :, np.newaxis]
        else:
            condensed=np.concatenate((condensed, file[:, :, :, np.newaxis]), axis=3)
    
    avg_file=np.mean(condensed, axis=3)
    io.write(main_directory+'/'+'avg0_atlas_auto.tif', avg_file)




