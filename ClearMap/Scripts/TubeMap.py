#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
TubeMap
=======

This script is the main pipeline to generate annotated graphs from vascualture 
lightsheet data [Kirst2020]_.

See the :ref:`TubeMap tutorial </TubeMap.ipynb>` for a tutorial and usage.

.. image:: ../Static/cell_abstract_2020.jpg
   :target: https://doi.org/10.1016/j.cell.2016.05.007 
   :width: 300  

References
----------
.. [Kirst2020] `Mapping the Fine-Scale Organization and Plasticity of the Brain Vasculature. Kirst, C., Skriabine, S., Vieites-Prado, A., Topilko, T., Bertin, P., Gerschenfeld, G., Verny, F., Topilko, P., Michalski, N., Tessier-Lavigne, M. and Renier, N., Cell, 180(4):780-795 <https://doi.org/10.1016/j.cell.2020.01.028>`_
"""
__author__    = 'Christoph Kirst <christoph.kirst.ck@gmail.com>'
__license__   = 'GPLv3 - GNU General Pulic License v3 (see LICENSE.txt)'
__copyright__ = 'Copyright © 2020 by Christoph Kirst'
__webpage__   = 'http://idisco.info'
__download__  = 'http://www.github.com/ChristophKirst/ClearMap2'

if __name__ == '__main__':

  #%%############################################################################
  ### Initialization 
  ###############################################################################
  
  #%% Initialize workspace
  
  from ClearMap.Environment import *  #analysis:ignore
  
  #directories and files
  directory = '/home/ckirst/Programs/ClearMap2/ClearMap/Tests/Data/TubeMap_Example'    
  
  expression_raw      = 'Raw/20-54-41_acta2_555-podo_cd31l_647_UltraII[<Y,2> x <X,2>]_C00_UltraII Filter0001.ome.npy'          
  expression_arteries = 'Raw/20-54-41_acta2_555-podo_cd31l_647_UltraII[<Y,2> x <X,2>]_C00_UltraII Filter0000.ome.npy'       
  expression_auto     = 'Autofluorescence/19-44-05_auto_UltraII_C00_xyz-Table Z<Z,4>.ome.tif'  
  
  resources_directory = settings.resources_path
  
  ws = wsp.Workspace('TubeMap', directory=directory);
  ws.update(raw=expression_raw, arteries=expression_arteries, autofluorescence=expression_auto)
  ws.info()
  
  
  #%% Initialize alignment 
  
  #init atals and reference files
  annotation_file, reference_file, distance_file=ano.prepare_annotation_files(
      slicing=(slice(None),slice(None),slice(0,246)), orientation=(1,-2,3),
      overwrite=False, verbose=True);
  
  #alignment parameter files    
  align_channels_affine_file   = io.join(resources_directory, 'Alignment/align_affine.txt')
  align_reference_affine_file  = io.join(resources_directory, 'Alignment/align_affine.txt')
  align_reference_bspline_file = io.join(resources_directory, 'Alignment/align_bspline.txt')
  
  
  #%%############################################################################
  ### Tile conversion
  ############################################################################### 
  
  #%% Convet raw data to npy files     
               
  io.convert_files(ws.file_list('raw', extension='tif'), extension='npy', 
                   processes=12, verbose=True);
  
  #%% Convert artery data to npy files      
             
  io.convert_files(ws.file_list('arteries', extension='tif'), extension='npy', 
                   processes=12, verbose=True);                 
                   
  
  #%%############################################################################
  ### Stitching
  ###############################################################################
  
  #%% Rigid z-alignment    
            
  layout = stw.WobblyLayout(expression=ws.filename('raw'), tile_axes=['X','Y'], overlaps=(45, 155));  
  
  st.align_layout_rigid_mip(layout, depth=[55, 155, None], max_shifts=[(-30,30),(-30,30),(-20,20)],
                            ranges = [None,None,None], background=(400, 100), clip=25000, 
                            processes=None, verbose=True)
  
  st.place_layout(layout, method='optimization', min_quality=-np.inf, lower_to_origin=True, verbose=True)
  
  st.save_layout(ws.filename('layout', postfix='aligned_axis'), layout)
  
  #%% Wobly alignment
  
  #layout = st.load_layout(ws.filename('layout', postfix='aligned_axis'))
  
  stw.align_layout(layout, axis_range=(None, None, 3), max_shifts=[(-30,30),(-15,15),(0,0)], axis_mip=None,
                   validate=dict(method='foreground', valid_range=(200, None), size=None),
                   prepare =dict(method='normalization', clip=None, normalize=True),
                   validate_slice=dict(method='foreground', valid_range=(200,20000), size=1500),
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
  
  #p3d.plot(ws.filename('stitched')) 
  
  #%% Wobbly stitching - arteries
                                                                                                                                                                  
  layout.replace_source_location(expression_raw, expression_arteries, method='expression')
  
  stw.stitch_layout(layout, sink = ws.filename('stitched', postfix='arteries'), method = 'interpolation', processes='serial', verbose=True)
  
  #p3d.plot(ws.filename('stitched', postfix='arteries')) 
  #p3d.plot([ws.filename('stitched'), ws.filename('stitched', postfix='arteries')])
  
  
  #%%############################################################################
  ### Resampling and atlas alignment 
  ###############################################################################
        
  #%% Resample 
             
  resample_parameter = {
      "source_resolution" : (1.625,1.625,1.6),
      "sink_resolution"   : (25,25,25),
      "processes" : None,
      "verbose" : True,             
      };
  
  io.delete_file(ws.filename('resampled'));
  
  res.resample(ws.filename('stitched'), sink=ws.filename('resampled'), **resample_parameter)
  
  #%% Resample autofluorescence
      
  resample_parameter_auto = {
      "source_resolution" : (5,5,6),
      "sink_resolution"   : (25,25,25),
      "processes" : None,
      "verbose" : True,                
      };    
  
  io.delete_file(ws.filename('resampled', postfix='autofluorescence'));
  
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
  ### Create test data
  ###############################################################################
  
  #%% Crop test data 
  
  #select sublice for testing the pipeline
  slicing = (slice(None),slice(0,1000),slice(900,1500));
  ws.create_debug('stitched', slicing=slicing);
  ws.create_debug('stitched', postfix='arteries', slicing=slicing);
  ws.debug = True; 
  
  #p3d.plot(ws.filename('stitched'))
    
  
  #%%############################################################################
  ### Binarization
  ###############################################################################
  
  #%% Binarization
  
  source = ws.filename('stitched');
  sink   = ws.filename('binary');
  
  binarization_parameter = vasc.default_binarization_parameter.copy();
  binarization_parameter['clip']['clip_range'] = (200, 7000)
  
  processing_parameter = vasc.default_binarization_processing_parameter.copy();
  processing_parameter.update(processes = None,
                              as_memory = True,
                              verbose = True);
     
  vasc.binarize(source, sink, binarization_parameter=binarization_parameter, processing_parameter=processing_parameter);
  
  #p3d.plot([[source, sink]])
  
  #%% Smoothing and filling
  
  source = ws.filename('binary');
  sink   = ws.filename('binary', postfix='postprocessed');
  
  postprocessing_parameter = vasc.default_postprocessing_parameter.copy();
  #postprocessing_parameter['fill'] = None;
  
  postprocessing_processing_parameter = vasc.default_postprocessing_processing_parameter.copy();
  postprocessing_processing_parameter.update(size_max=100);
  
  vasc.postprocess(source, sink, 
                   postprocessing_parameter=postprocessing_parameter, 
                   processing_parameter=postprocessing_processing_parameter, 
                   processes=None, verbose=True)
  
  #p3d.plot([[source, sink]])
  
  #%% Binarization - arteries
  
  source = ws.filename('stitched', postfix='arteries');
  sink   = ws.filename('binary', postfix='arteries');
  
  binarization_parameter = vasc.default_binarization_parameter.copy();
  binarization_parameter['clip']['clip_range'] = (1000, 8000)
  binarization_parameter['deconvolve']['threshold'] = 450  
  binarization_parameter['equalize'] = None;
  binarization_parameter['vesselize'] = None;
  
  processing_parameter = vasc.default_binarization_processing_parameter.copy();
  processing_parameter.update(processes = 20,
                              as_memory = True);
  
  vasc.binarize(source, sink, binarization_parameter=binarization_parameter, processing_parameter=processing_parameter);
  
  #p3d.plot([source, sink])
  
  #%% Smoothing and filling - arteries
  
  source = ws.filename('binary', postfix='arteries');
  sink   = ws.filename('binary', postfix='arteries_postprocessed');
  sink_smooth = ws.filename('binary', postfix='arteries_smoothed');
  
  postprocessing_parameter = vasc.default_postprocessing_parameter.copy();
  
  postprocessing_processing_parameter = vasc.default_postprocessing_processing_parameter.copy();
  postprocessing_processing_parameter.update(size_max = 50);
  
  vasc.postprocess(source, sink, postprocessing_parameter=postprocessing_parameter, 
                   processing_parameter=postprocessing_processing_parameter, 
                   processes=None, verbose=True)
  
  #p3d.plot([source, sink])
  
  
  #%%############################################################################
  ### Vessel filling 
  ###############################################################################
       
  #%% Vessel filling
              
  source = ws.filename('binary', postfix='postprocessed');
  sink   = ws.filename('binary', postfix='filled');
  
  processing_parameter = vf.default_fill_vessels_processing_parameter.copy();
  processing_parameter.update(size_max = 500, 
                              size_min = 'fixed',
                              axes = all,
                              overlap = 50);                 
                              
  vf.fill_vessels(source, sink, resample=1, threshold=0.5, cuda=None, processing_parameter=processing_parameter, verbose=True)
  
  #p3d.plot([source, sink]);
  
  #%% Vessel filling - arteries
                   
  source = ws.filename('binary', postfix='arteries_postprocessed');
  sink   = ws.filename('binary', postfix='arteries_filled');
  io.delete_file(sink);
  
  processing_parameter = vf.default_fill_vessels_processing_parameter.copy();
  processing_parameter.update(size_max = 1000, 
                              size_min = 'fixed',
                              axes = all,
                              overlap = 100);                 
                              
  vf.fill_vessels(source, sink, resample=2, threshold=0.5, cuda=None, processing_parameter=processing_parameter, verbose=True)
  
  #p2d.plot([source, sink]);
  
  #%% Combine binaries
  
  source          = ws.filename('binary', postfix='filled');
  source_arteries = ws.filename('binary', postfix='arteries_filled');
  sink            = ws.filename('binary', postfix='final');
  
  bp.process(np.logical_or, [source, source_arteries], sink, size_max=500, overlap=0, processes=None, verbose=True)
  
  #p2d.plot([source, source_arteries, sink]);
  
  
  #%%############################################################################
  ### Graph construction and measurements
  ###############################################################################
  
  #%% Skeletonize
  
  binary   = ws.filename('binary', postfix='filled');
  skeleton = ws.filename('skeleton')                   
  
  skl.skeletonize(binary, sink=skeleton, delete_border=True, verbose=True);
  
  #%% Graph from skeleton
  
  graph_raw = gp.graph_from_skeleton(ws.filename('skeleton'), verbose=True)
  #graph_raw.save(ws.filename('graph', postfix='raw'))
  
  #p3d.plot_graph_line(graph_raw)
  
  #%% Measure radii
  
  coordinates = graph_raw.vertex_coordinates();   
  radii, indices = mr.measure_radius(ws.filename('binary', postfix='filled'), coordinates, 
                                     value=0, fraction=None, max_radius=150, 
  #                                   value=None, fraction=0.8, max_radius=150,
                                     return_indices=True, default=-1, verbose=True);  
  graph_raw.set_vertex_radii(radii)
  
  
  #%% Artery binary measure
  
  binary_arteries = ws.filename('binary', postfix='filled');
  
  coordinates = graph_raw.vertex_coordinates();
  radii = graph_raw.vertex_radii();
  radii_measure = radii + 10;
  
  expression = me.measure_expression(binary_arteries, coordinates, radii, 
                                     method='max', verbose=True);
  
  graph_raw.define_vertex_property('artery_binary', expression);
  
  
  #%% Artery raw measure
  
  artery_raw = ws.filename('stitched', postfix='arteries');
  
  coordinates = graph_raw.vertex_coordinates();
  radii = graph_raw.vertex_radii();
  radii_measure = radii + 10;
  
  expression = me.measure_expression(artery_raw, coordinates, radii_measure, 
                                     method='max', verbose=True);
  
  graph_raw.define_vertex_property('artery_raw', np.asarray(expression.array, dtype=float));
  
  
  #%% Save raw graph
  
  graph_raw.save(ws.filename('graph', postfix='raw'))
  #graph_raw = grp.load(ws.filename('graph', postfix='raw'))
  
  
  #%%############################################################################
  ### Graph cleaning and reduction
  ###############################################################################
  
  #%% Graph cleaning 
  graph_cleaned = gp.clean_graph(graph_raw, 
                                 vertex_mappings = {'coordinates'   : gp.mean_vertex_coordinates, 
                                                    'radii'         : np.max,
                                                    'artery_binary' : np.max,
                                                    'artery_raw'    : np.max},                    
                                 verbose=True)  
  
  #%% Save cleaned graph
  
  graph_cleaned.save(ws.filename('graph', postfix='cleaned'))
  #graph_cleaned = grp.load(ws.filename('graph', postfix='cleaned'));
  
  
  #%% Graph reduction
  
  def vote(expression):
    return np.sum(expression) >= len(expression) / 1.5;
  
  graph_reduced = gp.reduce_graph(graph_cleaned, edge_length=True,
                            edge_to_edge_mappings = {'length' : np.sum},
                            vertex_to_edge_mappings={'artery_binary' : vote,
                                                     'artery_raw'    : np.max,
                                                     'radii'         : np.max},  
                            edge_geometry_vertex_properties=['coordinates', 'radii', 'artery_binary', 'artery_raw'],
                            edge_geometry_edge_properties=None,                        
                            return_maps=False, verbose=True)
  
  #%% Save reduced graph
  
  graph_reduced.save(ws.filename('graph', postfix='reduced'))
  #graph_reduced = grp.load(ws.filename('graph', postfix='reduced'));
  
  
  #%% Visualize graph annotations
  
  #artery_label = graph_reduced.edge_property('artery_binary');
  #artery_color = np.array([[1,0,0,1],[0,0,1,1]])[artery_label]
  #p3d.plot_graph_line(graph_reduced, edge_color = artery_color)
  #p3d.plot_graph_mesh(graph_reduced, edge_colors=artery_color)
  
  #p3d.plot_graph_edge_property(graph_reduced, edge_property='artery_raw', 
  #                             percentiles=[2,98], normalize=True, mesh=True)
  
  
  #%%############################################################################
  ### Atlas registration and annotation
  ###############################################################################
  
  #%% Graph atlas registration
  
  def transformation(coordinates):
    coordinates = res.resample_points(
                    coordinates, sink=None, orientation=None, 
                    source_shape=io.shape(ws.filename('binary', postfix='filled')), 
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
  
  graph_reduced.transform_properties(transformation=transformation, 
                             vertex_properties = {'coordinates' : 'coordinates_atlas'},
                             edge_geometry_properties = {'coordinates' : 'coordinates_atlas'},
                             verbose=True);
  
  
  def scaling(radii):
    resample_factor = res.resample_factor(
                        source_shape=io.shape(ws.filename('binary', postfix='filled')), 
                        sink_shape=io.shape(ws.filename('resampled')))
    return radii * np.mean(resample_factor);
  
  
  graph_reduced.transform_properties(transformation=scaling,
                             vertex_properties = {'radii' : 'radii_atlas'},
                             edge_properties   = {'radii' : 'radii_atlas'},
                             edge_geometry_properties = {'radii' : 'radii_atlas'})
  
  
  #%% Graph atlas annotation
  
  ano.set_annotation_file('/home/nicolas.renier/Documents/ClearMap_Ressources/annotation_25_1-246Ld.nrrd')
  def annotation(coordinates):
    label = ano.label_points(coordinates, key='order');
    return label;
  
  graph_reduced.annotate_properties(annotation, 
                            vertex_properties = {'coordinates_atlas' : 'annotation'},
                            edge_geometry_properties = {'coordinates_atlas' : 'annotation'});
  
  
  #%% Distance to surface
  
  distance_atlas = io.as_source(distance_file)
  distance_atlas_shape = distance_atlas.shape
  
  def distance(coordinates):
    c = np.asarray(np.round(coordinates), dtype=int);
    c[c<0] = 0;
    x = c[:,0]; y = c[:,1]; z = c[:,2];
    x[x>=distance_atlas_shape[0]] = distance_atlas_shape[0]-1;
    y[y>=distance_atlas_shape[1]] = distance_atlas_shape[1]-1;
    z[z>=distance_atlas_shape[2]] = distance_atlas_shape[2]-1;
    d = distance_atlas[x,y,z];
    return d;
  
  graph_reduced.transform_properties(distance, 
                             vertex_properties = {'coordinates_atlas' : 'distance_to_surface'},
                             edge_geometry_properties = {'coordinates_atlas' : 'distance_to_surface'});
  
  distance_to_surface = graph_reduced.edge_geometry('distance_to_surface', as_list=True);   
  distance_to_surface__edge = np.array([np.min(d) for d in distance_to_surface])                 
   
  graph_reduced.define_edge_property('distance_to_surface', distance_to_surface__edge)
  
  
  #%% Graph largests component
  
  graph = graph_reduced.largest_component()
  
  #%% Save annotated graph
  
  graph.save(ws.filename('graph', postfix='annotated'))
  #graph = grp.load(ws.filename('graph', postfix='annotated'));
  
  #%%############################################################################
  ### Artery & Vein processing
  ###############################################################################
  
  #%% Veins - large
  
  #veins: large radii and low acta2 expression
  vein_large_radius = 8
  vein_artery_expression_min = 0;
  vein_artery_expression_max = 2500;
  
  radii  = graph.edge_property('radii');
  artery_expression = graph.edge_property('artery_raw');
  
  vessel_large  = radii >=  vein_large_radius;
  
  vein_expression = np.logical_and(artery_expression >= vein_artery_expression_min, 
                                   artery_expression <= vein_artery_expression_max);
  
  vein_large = np.logical_and(vessel_large, vein_expression)
  
  #%% Arteries
  
  min_artery_size = 3;
  
  artery = graph.edge_property('artery_binary');
  graph_artery = graph.sub_graph(edge_filter=artery, view=True);
  graph_artery_edge, edge_map = graph_artery.edge_graph(return_edge_map=True)
  
  artery_components, artery_size = graph_artery_edge.label_components(return_vertex_counts=True);
  remove = edge_map[np.in1d(artery_components, np.where(artery_size < min_artery_size)[0])];
  artery[remove] = False;
  
  artery = np.logical_and(artery, np.logical_not(vein_large))
  
  graph.define_edge_property('artery', artery)
  
  
  #%%  Visualize - arteries and veins
  
  #edge_id = graph.edge_property('artery_binary');
  #edge_id[graph.edge_property('artery_cleaned')>0] += 2;
  #edge_id[graph.edge_property('artery')>0] += 4;
  
  #p3d.plot_graph_edge_property(graph, edge_property=edge_id, normalize=True, mesh=True)
  
  
  #%% Artery tracing
  
  #stop at surface, vein or low artery expression
  artery_trace_radius = 5;
  artery_expression_min = 500;
  distance_threshold = 15;
  max_artery_tracing = 5;
  
  radii = graph.edge_property('radii');
  artery = graph.edge_property('artery');
  artery_expression = graph.edge_property('artery_raw');
  distance_to_surface = graph.edge_property('distance_to_surface');
  
  def continue_edge(graph, edge):
    if distance_to_surface[edge] < distance_threshold or vein[edge]:
      return False;
    else:
      return radii[edge] >= artery_trace_radius and artery_expression[edge] >= artery_expression_min;
  
  artery_traced = gp.trace_edge_label(graph, artery, condition=continue_edge, max_iterations=max_artery_tracing);
  
  #artery_traced = graph.edge_close_binary(artery_traced, steps=1);
  #artery_traced = graph.edge_open_binary(artery_traced, steps=1);
  
  graph.define_edge_property('artery', artery_traced);
  
  
  #%% Veins - big
  
  vein_big_radius = 6
  
  radii  = graph.edge_property('radii');
  artery = graph.edge_property('artery');
  big_vessel  = radii >=  vein_big_radius;
  
  vein = np.logical_and(np.logical_or(vein_large,big_vessel), np.logical_not(artery))
  
  graph.define_edge_property('vein_big', vein); 
  
  #%% Veins - tracing 
  
  #trace veins by hysteresis thresholding - stop before arteries
  vein_trace_radius = 5;
  max_vein_tracing = 5;
  min_distance_to_artery = 1;
  
  radii = graph.edge_property('radii');
  artery = graph.edge_property('artery');
  vein_big  = graph.edge_property('vein_big');
  
  artery_expanded = graph.edge_dilate_binary(artery, steps=min_distance_to_artery);
  
  def continue_edge(graph, edge):
    if artery_expanded[edge]:
      return False;
    else:
      return radii[edge] >= vein_trace_radius;
  
  vein = gp.trace_edge_label(graph, vein_big, condition=continue_edge, max_iterations=max_vein_tracing);
  
  #vein = graph.edge_close_binary(vein, steps=1);
  #vein = graph.edge_open_binary(vein, steps=1);
  
  graph.define_edge_property('vein', vein);
  
  #%% Arteries - remove small components 
  
  min_artery_size = 30;
  
  artery = graph.edge_property('artery');
  graph_artery = graph.sub_graph(edge_filter=artery, view=True);
  graph_artery_edge, edge_map = graph_artery.edge_graph(return_edge_map=True)
  
  artery_components, artery_size = graph_artery_edge.label_components(return_vertex_counts=True);
  remove = edge_map[np.in1d(artery_components, np.where(artery_size < min_artery_size)[0])];
  artery[remove] = False;
  
  graph.define_edge_property('artery', artery)
  
  #%% Veins - remove small vein components 
  
  min_vein_size = 30;
  
  vein = graph.edge_property('vein');
  graph_vein = graph.sub_graph(edge_filter=vein, view=True);
  graph_vein_edge, edge_map = graph_vein.edge_graph(return_edge_map=True)
  
  vein_components, vein_size = graph_vein_edge.label_components(return_vertex_counts=True);
  remove = edge_map[np.in1d(vein_components, np.where(vein_size < min_vein_size)[0])];
  vein[remove] = False;
  
  graph.define_edge_property('vein', vein)
  
  #%% Save graph
  
  graph.save(ws.filename('graph'))
  
  
  #%%############################################################################
  ### Analysis
  ###############################################################################
  
  #%% Graph - loading
  graph = grp.load(ws.filename('graph'))
  
  #%% Graph - sub-region
  
  label = graph.vertex_annotation();
  label_leveled = ano.convert_label(label, key='order', value='order', level=6)
  vertex_filter = label_leveled == 6;
  #vertex_filter = graph.expand_vertex_filter(vertex_filter, steps=2)
  
  gs = graph.sub_graph(vertex_filter=vertex_filter);
  
  #%% Visualization - line graph - ABA colors
  
  vertex_colors = ano.convert_label(gs.vertex_annotation(), key='order', value='rgba');
  p = p3d.plot_graph_line(gs, color=vertex_colors)
  
  #%% Visualization - ABA colors
  
  vertex_colors = ano.convert_label(gs.vertex_annotation(), key='order', value='rgba');
  p = p3d.plot_graph_mesh(gs, default_radius=0.15, vertex_colors=vertex_colors, n_tube_points=5)
  
  #%% Visualization -  add artery label
  
  artery_label = gs.edge_property('artery');
  colormap = np.array([[0.8,0.0,0.0,1.0], [0.0,0.0,0.8,1.0]]);
  edge_colors = colormap[np.asarray(artery_label, dtype=int)];
  
  p = p3d.plot_graph_mesh(gs, edge_colors=edge_colors, n_tube_points=3);
  
  #%% Graph - sub-slice brain regions
  
  #Cerebellum
  #gs = graph.sub_slice((slice(1,270), slice(1,220), slice(210,220)));
  
  #Hippocampus sagittal
  #gs = graph.sub_slice((slice(1,300), slice(50,480), slice(165,185)));
  #gs = graph.sub_slice((slice(18,180), slice(150,280), slice(153,180)));
                    
  # Striatum coronal                 
  #gs = graph.sub_slice((slice(1,270), slice(100,108), slice(1,240)));
  
  # Auditory coronal
  #gs = graph.sub_slice((slice(1,270), slice(200,210), slice(1,240)));
                    
  #Cortex saggittal hippocampus                  
  #gs = graph.sub_slice((slice(1,300), slice(270,280), slice(1,240)));
                    
  #Midline
  #gs = graph.sub_slice((slice(500,1500), slice(3000,4000), slice(2910,2960)));                  
                    
  #%% Visualization - sub graphs with veins and arteries
  
  #color edges
  edge_vein_label = gs.edge_property('vein');
  edge_artery_label = gs.edge_property('artery');
  
  vertex_colors = ano.convert_label(gs.vertex_annotation(), key='order', value='rgba');
  
  connectivity = gs.edge_connectivity();
  edge_colors = (vertex_colors[connectivity[:,0]] + vertex_colors[connectivity[:,1]])/2.0;
  edge_colors[edge_artery_label>0] = [0.8,0.0,0.0,1.0]
  edge_colors[edge_vein_label  >0] = [0.0,0.0,0.8,1.0]
  
  p = p3d.plot_graph_mesh(gs, edge_colors=edge_colors, n_tube_points=3);
  
  
  #%% Visualization - vessel orientation 
  
  vetex_coordinates = gs.vertex_coordinates()
  connectivity = gs.edge_connectivity();
  
  orientations = vetex_coordinates[connectivity[:,0]] - vetex_coordinates[connectivity[:,1]];
  orientations = (orientations.T / np.linalg.norm(orientations, axis=1)).T
  
  #edge_colors = col.orientation_to_rgb(orientations, alpha=1.0);
  edge_colors = col.orientation_to_boys(orientations, alpha=1.0);
  edge_artery_label = gs.edge_property('artery');
  edge_colors[edge_artery_label>0] = [0.8,0.0,0.0,1.0]
  p = p3d.plot_graph_mesh(gs, edge_colors=edge_colors, n_tube_points=3);
  
  
  #%% Visualization - veins and arteries only
  
  edge_vein_label = gs.edge_property('vein');
  edge_artery_label = gs.edge_property('artery')
  
  edge_filter=np.logical_or(edge_vein_label,edge_artery_label)
  gsrt = gs.sub_graph(edge_filter=edge_filter)
  
  edge_vein_label = gsrt.edge_property('vein');
  edge_artery_label = gsrt.edge_property('artery')
  
  vertex_colors = ano.convert_label(gsrt.vertex_annotation(), key='order', value='rgba');
  
  connectivity = gsrt.edge_connectivity();
  edge_colors = (vertex_colors[connectivity[:,0]] + vertex_colors[connectivity[:,1]])/2.0;
  edge_colors[edge_artery_label>0] = [0.8,0.0,0.0,1.0]
  #edge_colors[edge_vein_label>0] = [0.0,0.0,0.8,1.0]
  
  p = p3d.plot_graph_mesh(gsrt, edge_colors=edge_colors, n_tube_points=5);
  
                         
  #%%############################################################################                  
  ### Voxelize Branch density
  ############################################################################### 
  
  voxelize_branch_parameter = {
      "method"  : 'sphere',      
      "radius"  : (15,15,15),
      "weights" : None,
      "shape"   : io.shape(reference_file),
      "verbose" : True                  
  };
  
  vertices = graph.vertex_coordinates();    
      
  branch_density = vox.voxelize(vertices, sink=ws.filename('density', postfix='branches'), dtype='float32', **voxelize_branch_parameter);
      




