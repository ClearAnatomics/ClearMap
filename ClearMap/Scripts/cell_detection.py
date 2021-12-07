#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CellMap
=======

This script is the main pipeline to analyze immediate early gene expression 
data from iDISCO+ cleared tissue [Renier2016]_.

See the :ref:`CellMap tutorial </CellMap.ipynb>` for a tutorial and usage.


.. image:: ../Static/cell_abstract_2016.jpg
   :target: https://doi.org/10.1016/j.cell.2020.01.028
   :width: 300

.. figure:: ../Static/CellMap_pipeline.png

  iDISCO+ and ClearMap: A Pipeline for Cell Detection, Registration, and 
  Mapping in Intact Samples Using Light Sheet Microscopy.


References
----------
.. [Renier2016] `Mapping of brain activity by automated volume analysis of immediate early genes. Renier* N, Adams* EL, Kirst* C, Wu* Z, et al. Cell. 2016 165(7):1789-802 <https://doi.org/10.1016/j.cell.2016.05.007>`_
"""
__author__    = 'Christoph Kirst <christoph.kirst.ck@gmail.com>'
__license__   = 'GPLv3 - GNU General Pulic License v3 (see LICENSE)'
__copyright__ = 'Copyright © 2020 by Christoph Kirst'
__webpage__   = 'http://idisco.info'
__download__  = 'http://www.github.com/ChristophKirst/ClearMap2'

if __name__ == "__main__":
     
  #%%############################################################################
  ### Initialization 
  ###############################################################################
  
  #%% Initialize workspace

  from ClearMap.Environment import *  #analysis:ignore

  #directories and files
  directory = ''    

  expression_raw      = ''           
  expression_auto     = ''  

  ws = wsp.Workspace('CellMap', directory=directory);
  ws.update(raw=expression_raw, autofluorescence=expression_auto)
  ws.info()
  
  ws.debug = False
  
  resources_directory = settings.resources_path
  
  
  #%% Initialize alignment -- 1,2,3 corresponds to x,y,z in a brain positioned sagitally, right hemisphere on top, so cortex on the left and OB in the upper part of the FOV. All changes correspond to 
  # twisting this initial model. Negative means mirror flip. Ex: (-3,2,1) Horizontal positioned; (1,2,3) Right hemisphere up, cortex left; (1,-2,3) Left hemisphere up, cortex left. Based on this indications
  # ClearMap will create automatically the correspondent template files in the ClearMap ressources folder.
  
  #init atals and reference files
  orientation = (1,-2,3)
  annotation_file, reference_file, distance_file=ano.prepare_annotation_files(
      slicing=(slice(None),slice(None),slice(0,230)), orientation=orientation,
      overwrite=False, verbose=True);
  
  #alignment parameter files    
  align_channels_affine_file   = io.join(resources_directory, 'Alignment/align_affine.txt')
  align_reference_affine_file  = io.join(resources_directory, 'Alignment/align_affine.txt')
  align_reference_bspline_file = io.join(resources_directory, 'Alignment/align_bspline.txt')
  
  

  #%%############################################################################
  ### Tile conversion
  ############################################################################# 

  #%% Convet raw data to npy files     
             
  io.convert_files(ws.file_list('raw', extension='tif'), extension='npy', 
                  processes=6, verbose=True);


  #%%############################################################################
  ### Stitching
  ###############################################################################

  #%% Rigid z-alignment    
          
  layout = stw.WobblyLayout(expression=ws.filename('raw'), tile_axes=['X','Y'], overlaps=(55, 55));  

  st.align_layout_rigid_mip(layout, depth=[60, 60, None], max_shifts=[(-15,15),(-15,15),(-10,10)],
                          ranges = [None,None,None], background=(000, 1000), clip=25000, 
                            processes=None, verbose=True)

  st.place_layout(layout, method='optimization', min_quality=-np.inf, lower_to_origin=True, verbose=True)

  st.save_layout(ws.filename('layout', postfix='aligned_axis'), layout)

  #%% Wobly alignment

  #layout = st.load_layout(ws.filename('layout', postfix='aligned_axis'))

  stw.align_layout(layout, axis_range=(None, None, 3), max_shifts=[(-10,10),(-6,6),(0,0)], axis_mip=None,
                  validate=dict(method='foreground', valid_range=(000, None), size=None),
                  prepare =dict(method='normalization', clip=None, normalize=True),
                  validate_slice=dict(method='foreground', valid_range=(000,20000), size= 0),
                  prepare_slice =None,
                  find_shifts=dict(method='tracing', cutoff=3*np.sqrt(2)),
                  processes=None, verbose=True)

  st.save_layout(ws.filename('layout', postfix='aligned'), layout)

  #%% Wobbly placement

  #layout = st.load_layout(ws.filename('layout', postfix='aligned'));
 
  stw.place_layout(layout, min_quality = -np.inf, 
                  method = 'optimization', 
                  smooth = dict(method = 'window', window = 'bartlett', window_length = 100, binary = None), 
                  smooth_optimized = dict(method = 'window', window = 'bartlett', window_length = 20, binary = 10),                             
                  fix_isolated = False, lower_to_origin = True,
                  processes = None, verbose = True)

  st.save_layout(ws.filename('layout', postfix='placed'), layout)

  #%% Wobbly stitching
  
  layout = st.load_layout(ws.filename('layout', postfix='placed'));

  stw.stitch_layout(layout, sink = ws.filename('stitched'), method = 'interpolation', processes='!serial', verbose=True)

  p3d.plot(ws.filename('stitched')) 
  
  
  #%% Conversion into TIF
  
  # io.convert_files(ws.file_list('stitched', extension='npy'), extension='tif', 
  #                 processes=12, verbose=True);
  
  
  #%%############################################################################
  ### Resampling and atlas alignment 
  ###############################################################################
        
  #%% Resample 
             
  resample_parameter = {
      "source_resolution" : (3.25, 3.25, 6),
      "sink_resolution"   : (25,25,25),
      "processes" : 'serial',
      "verbose" : True,             
      };
  
  io.delete_file(ws.filename('resampled'))
  
  res.resample(ws.filename('stitched'), sink=ws.filename('resampled'), **resample_parameter)

  # p3d.plot(ws.filename('resampled'))
  
  #%% Resample autofluorescence
      
  resample_parameter_auto = {
      "source_resolution" : (5,5,6),
      "sink_resolution"   : (25,25,25),
      "processes" : 'serial',
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
      "result_directory" :  ws.filename('resampled_to_auto')
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
      "result_directory" :  ws.filename('auto_to_reference')
      };
  
  elx.align(**align_reference_parameter);
  
  
  #%%############################################################################
  ### Create test data to determinate shape treshold
  ###############################################################################
  # running the background substraction in a subregion allows to check on one side that the disk size for this step has the right size, and also to set the right threshold for the cell detection. 
  # The threshold applied will be  based in the signal remaining after the background substraction, not based in the raw signal!!!
  #%% Crop test data 
  
  # select sublice for testing the pipeline
  slicing = (slice(200,2000),slice(2000,3000),slice(200,600));
  ws.create_debug('stitched', slicing=slicing);
  ws.debug = True; 
  
  # p3d.plot(ws.filename('stitched'))
    
  #run cell background substraction to set treshold
  cell_detection_parameter = cells.default_cell_detection_parameter.copy();
  cell_detection_parameter['illumination'] = None;
  cell_detection_parameter['background_correction']['shape'] = (10,10);
  
  cell_detection_parameter['intensity_detection']['measure'] = ['source'];
  cell_detection_parameter['shape_detection']['threshold'] = 500;
  cell_detection_parameter['background_correction']['save'] = ws.filename('cells', postfix='bkg')
   
  # io.delete_file(ws.filename('cells', postfix='maxima'))
  # cell_detection_parameter['maxima_detection']['save'] = ws.filename('cells', postfix='maxima')
  
  processing_parameter = cells.default_cell_detection_processing_parameter.copy();
  processing_parameter.update(
       processes =  7, #'serial',
      size_max = 25, #100, #35, 50
      size_min = 20,# 30, #30, 20
      overlap  = 7, #32, #10, 8
      verbose = True
      )
  
  cells.detect_cells(ws.filename('stitched'), ws.filename('cells', postfix='raw'),
                     cell_detection_parameter=cell_detection_parameter, 
                     processing_parameter=processing_parameter)
  
  #plot background substracted image 
  p3d.plot([ws.filename('stitched'),ws.filename('cells', postfix='bkg')])
  
   #%% visualization of maxima, shape, etc
  
  # p3d.plot([[ws.filename('stitched'), ws.filename('cells', postfix='maxima'),ws.filename('cells', postfix='shape')]])
  # p3d.plot([ws.filename('stitched'), ws.filename('cells', postfix='maxima'),ws.filename('cells', postfix='bkg')])
 
  # #%%
  # coordinates = np.hstack([ws.source('cells', postfix='raw')[c][:,None] for c in 'xyz']);
  # p = p3d.list_plot_3d(coordinates)
  # p3d.plot_3d(ws.filename('stitched'), view=p, cmap=p3d.grays_alpha(alpha=1))
  
  #%%############################################################################
  ### Cell detection
  ###############################################################################
  
  #%% Cell detection:
  
  cell_detection_parameter = cells.default_cell_detection_parameter.copy();
  cell_detection_parameter['illumination'] = None;
  cell_detection_parameter['background_correction']['shape'] = (10,10);
  
  cell_detection_parameter['intensity_detection']['measure'] = ['source'];
  cell_detection_parameter['shape_detection']['threshold'] = 700;
  # cell_detection_parameter['background_correction']['save'] = ws.filename('cells', postfix='bkg')
   
  # io.delete_file(ws.filename('cells', postfix='maxima'))
  #cell_detection_parameter['maxima_detection']['save'] = ws.filename('cells', postfix='maxima')
  
  processing_parameter = cells.default_cell_detection_processing_parameter.copy();
  processing_parameter.update(
       processes =  6, #'serial',
      size_max = 25, #100, #35, 50
      size_min = 20,# 30, #30, 20
      overlap  = 7, #32, #10, 8
      verbose = True
      )
  
  cells.detect_cells(ws.filename('stitched'), ws.filename('cells', postfix='raw'),
                     cell_detection_parameter=cell_detection_parameter, 
                     processing_parameter=processing_parameter)
  

  
  #%% Number of cells detected
    # cells_data = np.load(ws.filename('cells', postfix='raw'))
    # cells_data.shape

  #%% Cell statistics
  
  # source = ws.source('cells')
  
  # plt.figure(1); plt.clf();
  # names = source.dtype.names;
  # nx,ny = p3d.subplot_tiling(len(names));
  # for i, name in enumerate(names):
  #   plt.subplot(nx, ny, i+1)
  #   plt.hist(source[name]);
  #   plt.title(name)
  # plt.tight_layout();
  
  #%% Filter cells
  
  thresholds = {
      'source' : None,
      'size'   : (50,900)
      }
  
  cells.filter_cells(source = ws.filename('cells', postfix='raw'), 
                     sink = ws.filename('cells', postfix='filtered'), 
                     thresholds=thresholds);
  
  
  #%% Visualize
  
  # coordinates = np.array([ws.source('cells', postfix='filtered')[c] for c in 'xyz']).T;
  # p = p3d.list_plot_3d(coordinates, color=(1,0,0,0.5), size=10)
  # p3d.plot_3d(ws.filename('stitched'), view=p, cmap=p3d.grays_alpha(alpha=1))
  
  
  #%%############################################################################
  ### Cell atlas alignment and annotation
  ###############################################################################
  
  #%% Cell alignment
  
  source = ws.source('cells', postfix='filtered')
  
  def transformation(coordinates):
    coordinates = res.resample_points(
                    coordinates, sink=None, orientation=None, 
                    source_shape=io.shape(ws.filename('stitched')), 
                    sink_shape=io.shape(ws.filename('resampled')));
    
    coordinates = elx.transform_points(
                    coordinates, sink=None, 
                    transform_directory=ws.filename('resampled_to_auto'), 
                    binary=False, indices=False);
    
    coordinates = elx.transform_points(
                    coordinates, sink=None, 
                    transform_directory=ws.filename('auto_to_reference'),
                    binary=False, indices=False);
        
    return coordinates;
    
  
  coordinates = np.array([source[c] for c in 'xyz']).T;
  
  coordinates_transformed = transformation(coordinates);
  
  #%% Cell annotation
  
  label = ano.label_points(coordinates_transformed, key='order');
  names = ano.convert_label(label, key='order', value='name');
  
  #%% Save results
  
  coordinates_transformed.dtype=[(t,float) for t in ('xt','yt','zt')]
  label = np.array(label, dtype=[('order', int)]);
  names = np.array(names, dtype=[('name', 'a256')])
  
  import numpy.lib.recfunctions as rfn
  cells_data = rfn.merge_arrays([source[:], coordinates_transformed, label, names], flatten=True, usemask=False)
  
  io.write(ws.filename('cells'), cells_data)
  
  
  
  #%%############################################################################
  ### Cell csv generation for external analysis
  ###############################################################################
  
  #%% CSV export
  
   # source = ws.source('cells');
   # header = ', '.join([h[0] for h in source.dtype.names]);
   # np.savetxt(ws.filename('cells', extension='csv'), source[:], header=header, delimiter=',')
  
  #%% ClearMap 1.0 export
  
  source = ws.source('cells');
  
  clearmap1_format = {'points' : ['x', 'y', 'z'], 
                      'points_transformed' : ['xt', 'yt', 'zt'],
                      'intensities' : ['source', 'dog', 'background', 'size']}
  
  for filename, names in clearmap1_format.items():
    sink = ws.filename('cells', postfix=['ClearMap1', filename]);
    data = np.array([source[name] if name in source.dtype.names else np.full(source.shape[0], np.nan) for name in names]);
    data = data.T
    if orientation ==(1,-2,3):
      if filename == 'points_transformed':
        print(filename)
        data[:, 1]=528-data[:, 1]
        print(sink)
    io.write(sink, data);
  
  
  #%%############################################################################
  ### Voxelization - cell density
  ###############################################################################
  io.delete_file(ws.filename('density', postfix='counts'))
  source = ws.source('cells')
  
  coordinates = np.array([source[n] for n in ['xt','yt','zt']]).T;
  intensities = source['source'];
  size = source['size'];
  
  #%% Unweighted 
  
  voxelization_parameter = dict(
        shape = io.shape(annotation_file), 
        dtype = None, 
        weights = None,
        method = 'sphere', 
        radius = (5,5,5), 
        kernel = None, 
        processes = None, 
        verbose = True
      )
  
  vox.voxelize(coordinates, sink=ws.filename('density', postfix='counts'), **voxelization_parameter);

  #%% Remove crust
  
  dist2surf=io.read(distance_file)
  threshold=3
  shape=dist2surf.shape
  
  good_coordinates=np.logical_and(np.logical_and(coordinates[:,0]<shape[0],coordinates[:,1]<shape[1]),coordinates[:,2]<shape[2]).nonzero()[0]
  coordinates=coordinates[good_coordinates]
  coordinates_wcrust=coordinates[np.asarray([dist2surf[tuple(np.floor(coordinates[i]).astype(int))]>threshold for i in range(coordinates.shape[0])]).nonzero()[0]]
  
  
  vox.voxelize(coordinates_wcrust, sink=ws.filename('density', postfix='counts_wcrust'), **voxelization_parameter);  
  
  #%% 
  
  p3d.plot(ws.filename('density', postfix='counts'))
  
  
  #%% Weighted 
  
  voxelization_parameter = dict(
        shape = io.shape(annotation_file),
        dtype = None, 
        weights = intensities,
        method = 'sphere', 
        radius = (5,5,5), 
        kernel = None, 
        processes = None, 
        verbose = True
      )
  
  vox.voxelize(points, sink=ws.filename('density', postfix='intensities'), **voxelization_parameter);
  
  #%%
  
  p3d.plot(ws.filename('density', postfix='intensities'))
