#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
CellMap
=======

This script is the main processing script to analyze immediate early gene
expression data from iDISCO cleared tissue.

Reference
---------
[1] Renier*, Adams*, Kirst*, Wu* et al. Cell 2016
"""
__author__    = 'Christoph Kirst <christoph.kirst@ucsf.edu>'
__license__   = 'MIT License <http://www.opensource.org/licenses/mit-license.php>'
__copyright__ = 'Copyright (c) 2020 by Christoph Kirst'

#%%############################################################################
### Initialization 
###############################################################################

#%% Initialize workspace

from ClearMap.Environment import *  #analysis:ignore

#directories and files
directory = '/home/ckirst/Science/Projects/WholeBrainClearing/Vasculature/Experiment/CFos_Example'    

expression_raw      = 'Raw/Fos/Z<Z,4>.tif'           
expression_auto     = 'Autofluorescence/Auto/Z<Z,4>.tif'  

ws = wsp.Workspace('CellMap', directory=directory);
ws.update(raw=expression_raw, autofluorescence=expression_auto)
ws.info()

ws.debug = False

resources_directory = settings.resources_path

#%% Initialize alignment 

#init atals and reference files
annotation_file, reference_file, distance_file=ano.prepare_annotation_files(
    slicing=(slice(None),slice(None),slice(0,256)), orientation=(1,-2,3),
    overwrite=False, verbose=True);

#alignment parameter files    
align_channels_affine_file   = io.join(resources_directory, 'Alignment/align_affine.txt')
align_reference_affine_file  = io.join(resources_directory, 'Alignment/align_affine.txt')
align_reference_bspline_file = io.join(resources_directory, 'Alignment/align_bspline.txt')


#%%############################################################################
### Resampling and atlas alignment 
###############################################################################
      
#%% Resample 
           
resample_parameter = {
    "source_resolution" : (4.0625, 4.0625, 3),
    "sink_resolution"   : (25,25,25),
    "processes" : None,
    "verbose" : True,             
    };

res.resample(ws.filename('raw'), sink=ws.filename('resampled'), **resample_parameter)

#%% Resample autofluorescence
    
resample_parameter_auto = {
    "source_resolution" : (5,5,6),
    "sink_resolution"   : (25,25,25),
    "processes" : None,
    "verbose" : True,                
    };    

res.resample(ws.filename('autofluorescence'), sink=ws.filename('resampled', postfix='autofluorescence'), **resample_parameter_auto)

#p3d.plot([ws.filename('resampled'), ws.filename('resampled', postfix='autofluorescence')])

#%% Aignment - resampled to autofluorescence

# align the two channels
align_channels_parameter = {            
    #moving and reference images
    "moving_image" : ws.filename('resampled', postfix='autofluorescence'),
    "fixed_image"  : ws.filename('resampled'),
    
    #elastix parameter files for alignment
    "affine_parameter_file"  : align_channels_affine_file,
    "bspline_parameter_file" : None,
    
    #directory of the alig'/home/nicolas.renier/Documents/ClearMap_Ressources/Par0000affine.txt',nment result
    "result_directory" :  ws.filename('elastix_resampled_to_auto')
    }; 

elx.align(**align_channels_parameter);

#%% Alignment - autoflourescence to reference

# align autofluorescence to reference
align_reference_parameter = {            
    #moving and reference images
    "moving_image" : reference_file,
    "fixed_image"  : ws.filename('resampled', postfix='autofluorescence'),
    
    #elastix parameter files for alignment
    "affine_parameter_file"  :  align_reference_affine_file,
    "bspline_parameter_file" :  align_reference_bspline_file,
    #directory of the alignment result
    "result_directory" :  ws.filename('elastix_auto_to_reference')
    };

elx.align(**align_reference_parameter);


#%%############################################################################
### Create test data
###############################################################################

#%% Crop test data 

#select sublice for testing the pipeline
slicing = (slice(2000,2200),slice(2000,2200),slice(50,80));
ws.create_debug('stitched', slicing=slicing);
ws.create_debug('stitched', postfix='arteries', slicing=slicing);
ws.debug = True; 

#p3d.plot(ws.filename('stitched'))
  

#%%############################################################################
### Cell detection
###############################################################################

#%% Cell detection:

cell_detection_parameter = cells.default_cell_detection_parameter.copy();
cell_detection_parameter['illumination'] = None;
cell_detection_parameter['background'] = None;
cell_detection_parameter['intensity_detection']['measure'] = ['source'];

cell_detection_parameter['maxima_detection']['save'] = ws.filename('cells', postfix='maxima')

processing_parameter = cells.default_cell_detection_processing_parameter.copy();
processing_parameter.update(
    processes = 'serial',
    size_max = 100,
    size_min = 50,
    overlap  = 32,
    verbose = True
    )

cells.detect_cells('test.npy', ws.filename('cells'),
                   cell_detection_parameter=cell_detection_parameter, 
                   processing_parameter=processing_parameter)

#%% Cell data structure

header = ['x','y','z'];
dtypes = [int, int, int]
if cell_detection_parameter['shape_detection'] is not None:
  header += ['size'];
  dtypes += [int];
header += cell_detection_parameter['intensity_detection']['measure'] 
dtypes += [float] * len(cell_detection_parameter['intensity_detection']['measure'])

structure = [(h,t) for h,t in zip(header, dtypes)];

cells_raw = ws.source('cells');
cells_raw = np.array([tuple(t) for t in cells_raw[:]], dtype=structure)

#%% visualization

p3d.plot([['test.npy', ws.filename('cells', postfix='maxima')]])

p = p3d.list_plot_3d(ws.source('cells')[:,:3])
p3d.plot_3d(io.as_source('test.npy'), view=p)


#%% Filter cells

thresholds = {
    'source' : None,
    'size'   : (20,900)
    }

ids = np.ones(cells_raw.shape[0], dtype=bool);
for k,t in thresholds.items():
  if t:
    if not isinstance(t, (tuple, list)):
      t = (t, None);
    if t[0] is not None:
      ids = np.logical_and(ids, t[0] <= cells_raw[k])
    if t[1] is not None:
      ids = np.logical_and(ids, t[1] > cells_raw[k]);
cells_filtered = cells_raw[ids];

io.write(ws.filename('cells', postfix='filtered'), cells_filtered)


#%%############################################################################
### Cell atlas alignment
###############################################################################

def transformation(coordinates):
  coordinates = res.resample_points(
                  coordinates, sink=None, orientation=None, 
                  source_shape=io.shape(ws.filename('raw')), 
                  sink_shape=io.shape(ws.filename('resampled')));
  
  coordinates = elx.transform_points(
                  coordinates, sink=None, 
                  transform_directory=ws.filename('resampled_to_auto'), 
                  binary=True, indices=False);
  
  coordinates = elx.transform_points(
                  coordinates, sink=None, 
                  transform_directory=ws.filename('auto_to_reference'),
                  binary=True, indices=False);
      
  return coordinates;
  

coordinates = np.array([cells_filtered[c] for c in ('x','y','z')]).T;

coordinates_transformed = transformation(coordinates);

#%% Cell annotation

label = ano.label_points(coordinates_transformed, key='order');
names = ano.convert_label(label, key='order', value='name');


#%%############################################################################
### Cell data generation
###############################################################################

#%% Data generation

coordinates_transformed = np.array([tuple(t) for t in coordinates_transformed], dtype=[(t,float) for t in ('xt', 'yt', 'zt')])
label = np.array(label, dtype=[('order', int)]);
names = np.array(names, dtype=[('name', 'a256')])

import numpy.lib.recfunctions as rfn
cells_data = rfn.merge_arrays([cells_filtered, coordinates_transformed, label, names], flatten=True, usemask=False)

io.write(ws.filename('cells', postfix='data'), cells_data)

#%% CSV export

structure = cells_data.dtype.descr
header = ', '.join([h[0] for h in structure]);

np.savetxt(ws.filename('cells', extension='csv'), cells_data, header=header, delimiter=',')


#%%############################################################################
### Voxelization - cell density
###############################################################################

cells_data = np.load(ws.filename('cells', postfix='data'))

coordinates = cells_data[['xt','yt','zt']];
intensities = cellsd_data[['source']];

#%% Unweighted 

voxelization_parameter = dict(
      shape = io.shape(annotation_file), 
      dtype = None, 
      weights = None,
      method = 'sphere', 
      radius = (7,7,7), 
      kernel = None, 
      processes = None, 
      verbose = True
    )

vox.voxelize(points, sink=ws.filename('density', postfix='counts'), **voxelization_parameter);


#%% Weighted 

voxelization_parameter = dict(
      shape = None, 
      dtype = None, 
      weights = intensities,
      method = 'sphere', 
      radius = (7,7,7), 
      kernel = None, 
      processes = None, 
      verbose = True
    )

vox.voxelize(points, sink=ws.filename('density', postfix='intensities'), **voxelization_parameter);


