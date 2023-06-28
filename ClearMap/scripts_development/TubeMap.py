#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
TubeMap
=======

This script is the main processing script to generate annotated
graphs from vascualture lightsheet data.
"""
__author__ = 'Christoph Kirst <ckirst@rockefeller.edu>'
__license__ = 'MIT License <http://www.opensource.org/licenses/mit-license.php>'
__copyright__ = 'Copyright (c) 2019 by Christoph Kirst'

# %%############################################################################
### Initialization
###############################################################################

from ClearMap.Environment import *  # analysis:ignore
import pyqtgraph as pg


# def runs_on_spyder():
#     return any('SPYDER' in name for name in os.environ)
#
#
# if not runs_on_spyder():
#     pg.mkQApp()

# directories and files
directory = '/media/sophie.skriabine/mercury/Elisa/dev-auto-ours/p7/1'

expression_raw = '211108_vasc_20-00-34/20-00-34_vasc_UltraII[<Y,2> x <X,2>]_C01.ome.npy'
expression_arteries = '211108_vasc_20-00-34/20-00-34_vasc_UltraII[<Y,2> x <X,2>]_C00.ome.npy'
expression_auto = '221109_auto_18-04-04/18-04-04_auto_Blaze_C00_xyz-Table Z<Z,4>.ome.tif'

resources_directory = settings.resources_path

ws = wsp.Workspace('TubeMap', directory=directory)#, prefix='14a');
ws.update(raw=expression_raw, arteries=expression_arteries, autofluorescence=expression_auto)
ws.info()

ano.initialize(
    label_file='/home/elisa.delaunoit/programs/ClearMap2/ClearMap/Resources/Atlas/region_ids_test_ADMBA.json',
    extra_label=[],
    annotation_file='/home/elisa.delaunoit/programs/ClearMap2/ClearMap/Resources/Atlas/P14_annotation_halfbrain-REAL.tif')

# init atlas and reference files
annotation_file, reference_file, distance_file = ano.prepare_annotation_files(
    slicing=(slice(None), slice(None), slice(0, 246)), orientation=(-1, 2, 3),
    verbose=True,
    annotation_file='/home/elisa.delaunoit/programs/ClearMap2/ClearMap/Resources/Atlas/P14_annotation_halfbrain-REAL.tif',
    reference_file='/home/elisa.delaunoit/programs/ClearMap2/ClearMap/Resources/Atlas/P14_template_halfbrain_rescaled_oriented.tif');

# alignment parameter files
align_channels_affine_file = io.join(resources_directory, 'Alignment/align_affine.txt')
align_reference_affine_file = io.join(resources_directory, 'Alignment/align_affine.txt')
align_reference_bspline_file = io.join(resources_directory, 'Alignment/align_bspline.txt')

# %%############################################################################
### Tile conversion
###############################################################################

# create raw data npy files
io.convert_files(ws.file_list('raw', extension='tif'), extension='npy',
                 processes=24, verbose=True);

# %%

# create artery data npy files
io.convert_files(ws.file_list('arteries', extension='tif'), extension='npy',
                 processes=24, verbose=True);

# %%############################################################################
### Histogram correction
############################################################################### #%%
# from multiprocessing import Pool
# import time


# import numpy as np
# from scipy import interpolate


# import matplotlib.pyplot as plt

# from skimage import data
# from skimage import exposure
# from skimage.exposure import match_histograms
# #%% vessels equalization

# ref_path=ws.file_list('raw', extension='npy')[0]
# reference=io.read(ref_path)

# for i, image_path in enumerate(ws.file_list('raw', extension='npy')):
#     image=io.read(image_path)
#     # image_path_arteries=ws.file_list('arteries', extension='npy')[i]
#     # image_arteries=io.read(image_path_arteries)

#     matched=io.mmp.create(image_path[:-8]+'_equalized.npy',dtype='uint16', mode='w+',shape=image.shape)
#     # matched_arteries=io.mmp.create(image_path_arteries[:-8]+'_equalized.npy',dtype='uint16', mode='w+',shape=image.shape)

#     im=image[:,:,2273]
#     ref=reference[:,:,2273]

#     res=match_histograms(im, ref).astype('uint16')
#     unique, counts = np.unique(im, return_counts=True)

#     matched_values=np.zeros(unique.shape)
#     for i, u in enumerate(unique):
#         print(i)
#         pos=np.asarray(im==u).nonzero()
#         x=pos[0]
#         y=pos[1]
#         matched_values[i]=res[x, y].mean()

#     f = interpolate.interp1d(unique, matched_values, kind='slinear', bounds_error=False, fill_value='extrapolate')
#     # plt.figure()
#     # xnew = np.arange(0, 15000, 100)
#     # ynew = f(xnew)   # use interpolation function returned by `interp1d`
#     # plt.plot(unique, matched_values, 'o', xnew, ynew, '-')
#     # plt.show()

#     def equalize_hist_custom(args):
#         i=args
#         im=image[:,:,i]
#         print(i)

#         res=f(im).astype('uint16')
#         matched[:,:,i]=res

#     # def equalize_hist_custom_arteries(args):
#     #     i=args
#     #     im=image_arteries[:,:,i]
#     #     print(i)
#     #     res=f(im).astype('uint16')
#     #     matched_arteries[:,:,i]=res


#     with Pool(processes=20) as pool:
#         pool.map(equalize_hist_custom, [i for i in range(image.shape[2])])
#         # pool.map(equalize_hist_custom_arteries, [i for i in range(image_arteries.shape[2])])
#     pool.close()

# #%% arteries equalization

# # ref_path=ws.file_list('arteries', extension='npy')[0]
# # reference=io.read(ref_path)
# # reference=np.clip(reference, 760, 20000)
# # reference=reference-760#+400

# # for i, image_path in enumerate(ws.file_list('arteries', extension='npy')):
# #     image=io.read(image_path)
# #     matched=io.mmp.create(image_path[:-8]+'_equalized.npy',dtype='uint16', mode='w+',shape=image.shape)

# #     im=image[:,:,2273]
# #     ref=reference[:,:,2273]

# #     res=match_histograms(im, ref).astype('uint16')
# #     unique, counts = np.unique(im, return_counts=True)

# #     matched_values=np.zeros(unique.shape)
# #     for i, u in enumerate(unique):
# #         print(i)
# #         pos=np.asarray(im==u).nonzero()
# #         x=pos[0]
# #         y=pos[1]
# #         matched_values[i]=res[x, y].mean()

# #     f = interpolate.interp1d(unique, matched_values, kind='slinear', bounds_error=False, fill_value='extrapolate')
# #     # plt.figure()
# #     # xnew = np.arange(0, 15000, 100)
# #     # ynew = f(xnew)   # use interpolation function returned by `interp1d`
# #     # plt.plot(unique, matched_values, 'o', xnew, ynew, '-')
# #     # plt.show()

# #     def equalize_hist_custom(args):
# #         i=args
# #         im=image[:,:,i]
# #         print(i)
# #         res=f(im).astype('uint16')
# #         matched[:,:,i]=res


# #     with Pool(processes=20) as pool:
# #         pool.map(equalize_hist_custom, [i for i in range(image.shape[2])])
# #     pool.close()

# #%%
# expression_raw      = expression_raw[:-8]+'_equalized.npy'
# # expression_arteries = expression_arteries[:-8]+'_equalized.npy'
# ws.update(raw=expression_raw, autofluorescence=expression_auto)
# ws.info()

# %%############################################################################
### Stitching
###############################################################################

layout = stw.WobblyLayout(expression=ws.filename('raw'), tile_axes=['X', 'Y'], overlaps=(50, 60));

st.align_layout_rigid_mip(layout, depth=[110, 110, None], max_shifts=[(-100, 100), (-105, 105), (-50, 50)],
                          ranges=[None, None, None], background=(400, 100), clip=65000,
                          verbose=True, processes='!serial')

st.place_layout(layout, method='optimization', min_quality=-np.inf, lower_to_origin=True, verbose=True)

st.save_layout(ws.filename('layout', postfix='aligned_axis'), layout)

# %%

layout = st.load_layout(ws.filename('layout', postfix='aligned_axis'))

# %%
stw.align_layout(layout, axis_range=(None, None, 3), max_shifts=[(-45, 45), (-45, 45), (0, 0)], axis_mip=None,
                 validate=dict(method='foreground', valid_range=(200, None), size=None),
                 prepare=dict(method='normalization', clip=None, normalize=True),
                 validate_slice=dict(method='foreground', valid_range=(200, 20000), size=1500),
                 prepare_slice=None,
                 find_shifts=dict(method='tracing', cutoff=3 * np.sqrt(2)),
                 processes='!serial', verbose=True)

st.save_layout(ws.filename('layout', postfix='aligned'), layout)

# %%
layout = st.load_layout(ws.filename('layout', postfix='aligned'));

stw.place_layout(layout, min_quality=-np.inf,
                 method='optimization',
                 smooth=dict(method='window', window='bartlett', window_length=100, binary=None),
                 fix_isolated=False, lower_to_origin=True,
                 smooth_optimized=dict(method='window', window='bartlett', window_length=20, binary=10),

                 processes=None, verbose=True)

st.save_layout(ws.filename('layout', postfix='placed'), layout)

# %%
layout = st.load_layout(ws.filename('layout', postfix='placed'));

stw.stitch_layout(layout, sink=ws.filename('stitched'), method='interpolation', processes='!serial', verbose=True)

# p3d.plot(ws.filename('stitched'))

# %% Stitching - arteries
layout = st.load_layout(ws.filename('layout', postfix='placed'));
layout.replace_source_location(ws.filename('raw'), ws.filename('arteries'))  # [:-14]+'.ome.npy'

stw.stitch_layout(layout, sink=ws.filename('stitched', postfix='arteries'), method='interpolation', processes='!serial',
                  verbose=True)
# %%

# p3d.plot([ws.filename('stitched'), ws.filename('stitched', postfix='arteries')])

# p3d.plot([ws.filename('binary', postfix='arteries'), ws.filename('stitched', postfix='arteries')])


# p3d.plot([ws.filename('stitched', postfix='eq'), ws.filename('stitched')])


# p3d.plot([ws.filename('stitched', postfix='arteries_eq2'), ws.filename('stitched', postfix='arteries')])


# %%############################################################################
### Resampling and alignment
###############################################################################

resample_parameter = {
    "source_resolution": (1.625, 1.625, 2),
    "sink_resolution": (16.752, 16.752, 25),
    "processes": 8,
    "verbose": True,
};

res.resample(ws.filename('stitched'), sink=ws.filename('resampled'), **resample_parameter)

# Resampling
resample_parameter_auto = {
    "source_resolution": (5.9,5.9, 6),
    "sink_resolution": ( 25, 25, 25),
    "processes": None,
    "verbose": True,
};

res.resample(ws.filename('autofluorescence'), sink=ws.filename('resampled', postfix='autofluorescence'),
             **resample_parameter_auto)

# % Aignment
# align the two channels
align_channels_parameter = {
    # moving and reference images
    "moving_image": ws.filename('resampled', postfix='autofluorescence'),
    "fixed_image": ws.filename('resampled'),

    # elastix parameter files for alignment
    "affine_parameter_file": align_channels_affine_file,
    "bspline_parameter_file": None,

    # directory of the alig'/home/nicolas.renier/Documents/ClearMap_Ressources/Par0000affine.txt',nment result
    "result_directory": ws.filename('resampled_to_auto')
};
align_channels_parameter
elx.align(**align_channels_parameter);

# %%
# Distance
# autofluo=io.read(ws.filename('resampled', postfix='autofluorescence'))
ref = io.read(annotation_file)
import scipy.ndimage as ndi

distance = ndi.distance_transform_edt(ref > 0);

io.write(ws.filename('resampled', postfix='autofluorescence_distance'), distance)

distance_file = ws.filename('resampled', postfix='autofluorescence_distance')

# %%############################################################################
### run  to align to a reference autofluo file for the same devlpment timepoint
###############################################################################
# reference_file=ws.filename('resampled', postfix='autofluorescence')
distance_file = ws.filename('resampled', postfix='autofluorescence_distance')

# align autoflourescence to reference
align_reference_parameter = {
    # moving and reference images
    "moving_image": reference_file,
    "fixed_image": ws.filename('resampled', postfix='autofluorescence'),

    # elastix parameter files for alignment
    "affine_parameter_file": align_reference_affine_file,
    "bspline_parameter_file": align_reference_bspline_file,
    # directory of the alignment result
    "result_directory": ws.filename('auto_to_reference')
};

elx.align(**align_reference_parameter);

# %%############################################################################
### Binarization
###############################################################################


source = ws.filename('stitched');
sink = ws.filename('binary');

binarization_parameter = vasc.default_binarization_parameter.copy();
binarization_parameter['clip']['clip_range'] = (140, 1600)

processing_parameter = vasc.default_binarization_processing_parameter.copy();
processing_parameter['processes'] = 11;
processing_parameter['as_memory'] = True;

vasc.binarize(source, sink, binarization_parameter=binarization_parameter, processing_parameter=processing_parameter);

# p3d.plot([source, sink])

# %% Smoothing and filling

source = ws.filename('binary');
sink = ws.filename('binary', postfix='postprocessed');

postprocessing_parameter = vasc.default_postprocessing_parameter.copy();
postprocessing_parameter['fill'] = None
postprocessing_processing_parameter = vasc.default_postprocessing_processing_parameter.copy();
postprocessing_processing_parameter['size_max'] = 50;

vasc.postprocess(source, sink, postprocessing_parameter=postprocessing_parameter,
                 processing_parameter=postprocessing_processing_parameter,
                 processes=None, verbose=True)

p3d.plot([source, sink])

# %% Binarization - arteries

# source = ws.filename('stitched', postfix='arteries');
# sink   = ws.filename('binary', postfix='arteries');

# processing_parameter = vasc.default_binarization_processing_parameter.copy();
# processing_parameter['processes'] = 11;
# processing_parameter['as_memory'] = True;

# binarization_parameter = vasc.default_binarization_parameter.copy();
# binarization_parameter['clip']['clip_range'] = (600, 1400)
# binarization_parameter['deconvolve']['threshold'] = 950
# binarization_parameter['equalize'] = None;
# binarization_parameter['vesselize'] = None;

# vasc.binarize(source, sink, binarization_parameter=binarization_parameter, processing_parameter=processing_parameter);

# p3d.plot([source, sink])

# %% Smoothing and filling - arteries

source = ws.filename('binary', postfix='arteries');
sink = ws.filename('binary', postfix='arteries_postprocessed');
sink_smooth = ws.filename('binary', postfix='arteries_smoothed');

postprocessing_parameter = vasc.default_postprocessing_parameter.copy();
postprocessing_parameter['fill'] = None
postprocessing_processing_parameter = vasc.default_postprocessing_processing_parameter.copy();
postprocessing_processing_parameter['size_max'] = 50;

vasc.postprocess(source, sink, postprocessing_parameter=postprocessing_parameter,
                 processing_parameter=postprocessing_processing_parameter,
                 processes=None, verbose=True)

p3d.plot([source, sink])

# %%############################################################################
# ### Vessel filling
# ###############################################################################
# import ClearMap.ImageProcessing.MachineLearning.VesselFilling.VesselFilling as vf
# source = ws.filename('binary', postfix='postprocessed');
# sink   = ws.filename('binary', postfix='filled');

# processing_parameter = vf.default_fill_vessels_processing_parameter.copy();
# processing_parameter.update(size_max = 500,
#                             size_min = 'fixed',
#                             axes = all,
#                             overlap = 50);

# vf.fill_vessels(source, sink, resample=1, threshold=0.5, cuda=True, processing_parameter=processing_parameter, verbose=True)

# %%############################################################################
### Vessel filling
###############################################################################
import ClearMap.ImageProcessing.MachineLearning.VesselFilling.VesselFilling as vf

source = ws.filename('binary', postfix='arteries_postprocessed');
sink = ws.filename('binary', postfix='arteries_filled');

processing_parameter = vf.default_fill_vessels_processing_parameter.copy();
processing_parameter.update(size_max=1000,
                            size_min='fixed',
                            axes=all,
                            overlap=100);

vf.fill_vessels(source, sink, resample=2, threshold=0.5, cuda=True, processing_parameter=processing_parameter,
                verbose=True)

# %%############################################################################
### Binary combination
###############################################################################


source = ws.filename('binary', postfix='filled');
source_arteries = ws.filename('binary', postfix='arteries_filled');
sink = ws.filename('binary', postfix='combined');

bp.process(np.logical_or, [source, source_arteries], sink, size_max=500, overlap=0, processes=None, verbose=True)

# pd.plot([source, source_arteries, sink]);
# %% filling - combined

source = ws.filename('binary', postfix='combined');
sink = ws.filename('binary', postfix='final');

postprocessing_parameter = vasc.default_postprocessing_parameter.copy();
postprocessing_processing_parameter = vasc.default_postprocessing_processing_parameter.copy();
postprocessing_processing_parameter['size_max'] = 50;

vasc.postprocess(source, sink, postprocessing_parameter=postprocessing_parameter,
                 processing_parameter=postprocessing_processing_parameter,
                 processes=None, verbose=True)

# p3d.plot([source, sink])

# %%############################################################################
### Graph construction and measurements
###############################################################################

binary = ws.filename('binary', postfix='final');
skeleton = ws.filename('skeleton')

skl.skeletonize(binary, sink=skeleton, delete_border=True, verbose=True);
### Skeleton to Graph

graph_raw = gp.graph_from_skeleton(ws.filename('skeleton'), verbose=True)
# graph_raw.save(ws.filename('graph', postfix='raw'))

# p3d.plot_graph_line(graph_raw)

# %% Measure radii

coordinates = graph_raw.vertex_coordinates();
# radii, indices = mr.measure_radius(ws.filename('stitched'), coordinates, value = None, fraction = 0.8, max_radius = 100, return_indices=True, default = 1);
radii, indices = mr.measure_radius(ws.filename('binary', postfix='final'), coordinates, value=0, fraction=None,
                                   max_radius=150, return_indices=True, default=-1);

graph_raw.set_vertex_radii(radii)

# %% Artery binary measure

binary_arteries = ws.filename('binary', postfix='arteries_filled');

coordinates = graph_raw.vertex_coordinates();
radii = graph_raw.vertex_radii();
radii_measure = radii + 10;

expression = me.measure_expression(binary_arteries, coordinates, radii, method='max');

graph_raw.define_vertex_property('artery_binary', expression);

# %% Artery raw measure

artery_raw = ws.filename('stitched', postfix='arteries');

coordinates = graph_raw.vertex_coordinates();
radii = graph_raw.vertex_radii();
radii_measure = radii + 10;

expression = me.measure_expression(artery_raw, coordinates, radii_measure, method='max');

graph_raw.define_vertex_property('artery_raw', np.asarray(expression.array, dtype=float));

# %% save
graph_raw.save(ws.filename('graph', postfix='raw'))
# graph_raw = grp.load(ws.filename('graph', postfix='raw'))


# %%############################################################################
### Graph cleaning and reduction
###############################################################################

# %% Graph cleaning
graph_cleaned = gp.clean_graph(graph_raw,
                               vertex_mappings={'coordinates': gp.mean_vertex_coordinates,
                                                'radii': np.max,
                                                'artery_binary': np.max,
                                                'artery_raw': np.max},
                               verbose=True)

# %% save
graph_cleaned.save(ws.filename('graph', postfix='cleaned'))


# graph_cleaned = grp.load(ws.filename('graph', postfix='cleaned'));


# %% Graph reduction

def vote(expression):
    return np.sum(expression) >= len(expression) / 1.5;


graph_reduced = gp.reduce_graph(graph_cleaned, edge_length=True,
                                edge_to_edge_mappings={'length': np.sum},
                                vertex_to_edge_mappings={'artery_binary': vote,
                                                         'artery_raw': np.max,
                                                         'radii': np.max},
                                edge_geometry_vertex_properties=['coordinates', 'radii', 'artery_binary', 'artery_raw'],
                                edge_geometry_edge_properties=None,
                                return_maps=False, verbose=True)

# %%
graph_reduced.save(ws.filename('graph', postfix='reduced'))
graph_reduced = grp.load(ws.filename('graph', postfix='reduced'));


# %%############################################################################
### Atlas registration and annotation

# %% graph atlas registration

def transformation(coordinates):
    coordinates = res.resample_points(
        coordinates, sink=None, orientation=None,
        source_shape=io.shape(ws.filename('binary', postfix='final')),
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
                                   vertex_properties={'coordinates': 'coordinates_atlas'},
                                   edge_geometry_properties={'coordinates': 'coordinates_atlas'},
                                   verbose=True);


def scaling(radii):
    resample_factor = res.resample_factor(
        source_shape=io.shape(ws.filename('binary', postfix='final')),
        sink_shape=io.shape(ws.filename('resampled')))
    return radii * np.mean(resample_factor);


graph_reduced.transform_properties(transformation=scaling,
                                   vertex_properties={'radii': 'radii_atlas'},
                                   edge_properties={'radii': 'radii_atlas'},
                                   edge_geometry_properties={'radii': 'radii_atlas'})

# %% annotation
ano.set_annotation_file(annotation_file)


def annotation(coordinates):
    label = ano.label_points(coordinates, key='order');
    return label;


graph_reduced.annotate_properties(annotation,
                                  vertex_properties={'coordinates_atlas': 'annotation'},
                                  edge_geometry_properties={'coordinates_atlas': 'annotation'});

# %% distance to surface
distance_file = io.read(distance_file)
distance_file_shape = distance_file.shape;


def distance(coordinates):
    c = np.asarray(np.round(coordinates), dtype=int);
    c[c < 0] = 0;
    x = c[:, 0];
    y = c[:, 1];
    z = c[:, 2];
    x[x >= distance_file_shape[0]] = distance_file_shape[0] - 1;
    y[y >= distance_file_shape[1]] = distance_file_shape[1] - 1;
    z[z >= distance_file_shape[2]] = distance_file_shape[2] - 1;
    d = distance_file[x, y, z];
    return d;


graph_reduced.transform_properties(distance,
                                   vertex_properties={'coordinates_atlas': 'distance_to_surface'},
                                   edge_geometry_properties={'coordinates_atlas': 'distance_to_surface'});

distance_to_surface = graph_reduced.edge_geometry('distance_to_surface', as_list=True);
distance_to_surface__edge = np.array([np.min(d) for d in distance_to_surface])

graph_reduced.define_edge_property('distance_to_surface', distance_to_surface__edge)

# %%
graph = graph_reduced.largest_component()

graph.save(ws.filename('graph', postfix='annotated'))

# %%############################################################################
### Artery & Vein processing

graph = grp.load(ws.filename('graph', postfix='annotated'));

# %% detect acta2+ veins via large radii and low acta2 expression

vein_large_radius = 6.5  # 8
vein_artery_expression_min = 0;
vein_artery_expression_max = 2500;

radii = graph.edge_property('radii');
artery_expression = graph.edge_property('artery_raw');

vessel_large = radii >= vein_large_radius;

vein_expression = np.logical_and(artery_expression >= vein_artery_expression_min,
                                 artery_expression <= vein_artery_expression_max);

vein_large = np.logical_and(vessel_large, vein_expression)

# %% Remove small artery components

min_artery_size = 3;

artery = graph.edge_property('artery_binary');
graph_artery = graph.sub_graph(edge_filter=artery, view=True);
graph_artery_edge, edge_map = graph_artery.edge_graph(return_edge_map=True)

artery_components, artery_size = graph_artery_edge.label_components(return_vertex_counts=True);
remove = edge_map[np.in1d(artery_components, np.where(artery_size < min_artery_size)[0])];
artery[remove] = False;

artery = np.logical_and(artery, np.logical_not(vein_large))

graph.define_edge_property('artery', artery)

# %%
# edge_id = graph.edge_property('artery_binary');
# edge_id[graph.edge_property('artery_cleaned')>0] += 2;
# edge_id[graph.edge_property('artery')>0] += 4;
#
# p3d.plot_graph_edge_property(graph, edge_property=edge_id,
#                             normalize=True, mesh=True)


# %% detect veins via large radii and low acta2 expression


vein_big_radius = 7.5  # 8.5

radii = graph.edge_property('radii');
artery = graph.edge_property('artery');
big_vessel = radii >= vein_big_radius;

vein = np.logical_and(np.logical_or(vein_large, big_vessel), np.logical_not(artery))

graph.define_edge_property('vein_big', vein);

# %% trace artery - stop at surface, vein or low artery expression

artery_trace_radius = 4  # 5;
artery_expression_min = 400  # 500;
distance_threshold = 15;
max_artery_tracing = 4  # 5;

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

# artery_traced = graph.edge_close_binary(artery_traced, steps=1);
# artery_traced = graph.edge_open_binary(artery_traced, steps=1);

graph.define_edge_property('artery', artery_traced);

# %% detect veins via large radii and low acta2 expression


vein_big_radius = 5  # 6

radii = graph.edge_property('radii');
artery = graph.edge_property('artery');
big_vessel = radii >= vein_big_radius;

vein = np.logical_and(np.logical_or(vein_large, big_vessel), np.logical_not(artery))

graph.define_edge_property('vein_big', vein);

# %% trace veins by hysteresis thresholding - stop before arteries

vein_trace_radius = 3.5  # 5;
max_vein_tracing = 3.5  # 5;
min_distance_to_artery = 1;

radii = graph.edge_property('radii');
artery = graph.edge_property('artery');
vein_big = graph.edge_property('vein_big');

artery_expanded = graph.edge_dilate_binary(artery, steps=min_distance_to_artery);


def continue_edge(graph, edge):
    if artery_expanded[edge]:
        return False;
    else:
        return radii[edge] >= vein_trace_radius;


vein = gp.trace_edge_label(graph, vein_big, condition=continue_edge, max_iterations=max_vein_tracing);

# vein = graph.edge_close_binary(vein, steps=1);
# vein = graph.edge_open_binary(vein, steps=1);


graph.define_edge_property('vein', vein);

# %% Remove small artery components

min_artery_size = 30;

artery = graph.edge_property('artery');
graph_artery = graph.sub_graph(edge_filter=artery, view=True);
graph_artery_edge, edge_map = graph_artery.edge_graph(return_edge_map=True)

artery_components, artery_size = graph_artery_edge.label_components(return_vertex_counts=True);
remove = edge_map[np.in1d(artery_components, np.where(artery_size < min_artery_size)[0])];
artery[remove] = False;

graph.define_edge_property('artery', artery)
# %% Remove small artery components

min_vein_size = 30;

vein = graph.edge_property('vein');
graph_vein = graph.sub_graph(edge_filter=vein, view=True);
graph_vein_edge, edge_map = graph_vein.edge_graph(return_edge_map=True)

vein_components, vein_size = graph_vein_edge.label_components(return_vertex_counts=True);
remove = edge_map[np.in1d(vein_components, np.where(vein_size < min_vein_size)[0])];
vein[remove] = False;

graph.define_edge_property('vein', vein)

# %% Done!


graph.save(ws.filename('graph'))

# %%############################################################################
### Analysis
###############################################################################
# %% Plot_cortex_color_dev
grt = grp.load(ws.filename('graph'))

ano.set_annotation_file(annotation_file)


def annotation(coordinates):
    label = ano.label_points(coordinates, key='id');
    return label;


grt.annotate_properties(annotation,
                        vertex_properties={'coordinates_atlas': 'annotation'},
                        edge_geometry_properties={'coordinates_atlas': 'annotation'});

# gs = grt.sub_slice((slice(1300,1400), slice(0,5000), slice(0,5000)));

# gs = grt.sub_slice((slice(1,270), slice(200,220), slice(1,240)));

gs = grt.sub_slice((slice(1900, 2100), slice(0, 5000), slice(0, 5000)));
edge_vein_label = gs.edge_property('vein');
edge_artery_label = gs.edge_property('artery');

vertex_colors = ano.convert_label(gs.vertex_annotation(), key='id', value='rgba');

connectivity = gs.edge_connectivity();
edge_colors = (vertex_colors[connectivity[:, 0]] + vertex_colors[connectivity[:, 1]]) / 2.0;
edge_colors[edge_vein_label > 0] = [0.0, 0.0, 0.8, 1.0]
edge_colors[edge_artery_label > 0] = [0.8, 0.0, 0.0, 1.0]

p = p3d.plot_graph_mesh(gs, edge_colors=edge_colors, n_tube_points=5);

# %%
ano.set_annotation_file(annotation_file)


def annotation(coordinates):
    label = ano.label_points(coordinates, key='id');
    return label;


grt.annotate_properties(annotation,
                        vertex_properties={'coordinates_atlas': 'annotation'},
                        edge_geometry_properties={'coordinates_atlas': 'annotation'});

# gs = grt.sub_slice((slice(1800,1900), slice(0,5000), slice(0,5000)));
gs = grt.sub_slice((slice(1, 300), slice(50, 480), slice(80, 90)), coordinates='coordinates_atlas');

vertex_colors = ano.convert_label(gs.vertex_annotation(), key='id', value='rgb');

connectivity = gs.edge_connectivity();
edge_colors = (vertex_colors[connectivity[:, 0]] + vertex_colors[connectivity[:, 1]]) / 2.0;
# edge_colors[edge_vein_label>0] = [0.0,0.0,0.8,1.0]

p = p3d.plot_graph_mesh(gs, edge_colors=edge_colors, n_tube_points=5);

# %%

p3d.plot_graph_mesh(gs)
# grt = graph_reduced_transformed;

# print(os.path.exists(ws.filename('graph')))
# print(os.stat(ws.filename('graph')))
# graph = grp.load(ws.filename('graph'))
grt = grp.load(ws.filename('graph'))
# %% extract sub-region

label = grt.vertex_annotation();
label_leveled = ano.convert_label(label, key='order', value='order', level=1)
vertex_filter = label_leveled == 1;
# vertex_filter = grt.expand_vertex_filter(vertex_filter, steps=2)

gs = grt.sub_graph(vertex_filter=vertex_filter);

# %% extract sub-region

label = grt.vertex_annotation();
label_leveled = ano.convert_label(label, key='order', value='order', level=6)
vertex_filter = label_leveled == 6;

gs = grt.sub_graph(vertex_filter=vertex_filter);

# %% plot with ABA colors

vertex_colors = ano.convert_label(gs.vertex_annotation(), key='order', value='rgba');
p = p3d.plot_graph_mesh(gs, default_radius=0.15, vertex_colors=vertex_colors, n_tube_points=5)

# %% plot reduced line graph with ABA colors

vertex_colors = ano.convert_label(gs.vertex_annotation(), key='order', value='rgba');
p = p3d.plot_graph_line(gs, color=vertex_colors)

# %%  plot with artery label
artery_label = gs.edge_property('artery');

colormap = np.array([[0.8, 0.0, 0.0, 1.0], [0.0, 0.0, 0.8, 1.0]]);
edge_colors = colormap[np.asarray(artery_label, dtype=int)];

p = p3d.plot_graph_mesh(gs, edge_colors=edge_colors, n_tube_points=3);

# %% brain regions

# Cerebellum
gs = grt.sub_slice((slice(1, 270), slice(1, 220), slice(210, 220)));

# Hippocampus sagittalgs.vertex_annotation()
gs = grt.sub_slice((slice(1, 300), slice(50, 480), slice(105, 165)), coordinates='coordinates_atlas');

gs = grt.sub_slice((slice(18, 180), slice(150, 280), slice(153, 180)));

gs = grt.sub_slice((slice(1, 300), slice(10, 480), slice(95, 100)), coordinates='coordinates_atlas');

# Striatum coronal
gs = grt.sub_slice((slice(1, 310), slice(280, 288), slice(1, 240)), coordinates='coordinates_atlas');

# Auditory coronal
gs = grt.sub_slice((slice(1, 270), slice(200, 210), slice(1, 240)));

# Cortex saggittal hippocampus
gs = grt.sub_slice((slice(1, 300), slice(270, 280), slice(1, 240)));

# midline
gs = gr.sub_slice((slice(500, 1500), slice(3000, 4000), slice(2910, 2960)));
#                  KeyError: ('v', 'coordinates_atlas')

# %% Color orientations

vetex_coordinates = gs.vertex_coordinates()
connectivity = gs.edge_connectivity();

orientations = vetex_coordinates[connectivity[:, 0]] - vetex_coordinates[connectivity[:, 1]];
orientations = (orientations.T / np.linalg.norm(orientations, axis=1)).T

# edge_colors = col.orientation_to_rgb(orientations, alpha=1.0);
edge_colors = col.orientation_to_boys(orientations, alpha=1.0);
edge_artery_label = gs.edge_property('artery');
edge_colors[edge_artery_label > 0] = [0.8, 0.0, 0.0, 1.0]
p = p3d.plot_graph_mesh(gs, edge_colors=edge_colors, n_tube_points=3);

# %% plot sub graph wth veins and arteries

# color edges
edge_vein_label = gs.edge_property('vein');
edge_artery_label = gs.edge_property('artery');

vertex_colors = ano.convert_label(gs.vertex_annotation(), key='id', value='rgba');

connectivity = gs.edge_connectivity();
edge_colors = (vertex_colors[connectivity[:, 0]] + vertex_colors[connectivity[:, 1]]) / 2.0;
edge_colors[edge_artery_label > 0] = [0.8, 0.0, 0.0, 1.0]
edge_colors[edge_vein_label > 0] = [0.0, 0.0, 0.8, 1.0]

p = p3d.plot_graph_mesh(gs, edge_colors=edge_colors, n_tube_points=3);
# make over all label

# %% plot sub graph wth ONLY veins and arteries

edge_vein_label = gs.edge_property('vein');
edge_artery_label = gs.edge_property('artery')

edge_filter = np.logical_or(edge_vein_label, edge_artery_label)
gsrt = gs.sub_graph(edge_filter=edge_filter)

edge_vein_label = gsrt.edge_property('vein');
edge_artery_label = gsrt.edge_property('artery')

vertex_colors = ano.convert_label(gsrt.vertex_annotation(), key='id', value='rgba');
#
connectivity = gsrt.edge_connectivity();
edge_colors = (vertex_colors[connectivity[:, 0]] + vertex_colors[connectivity[:, 1]]) / 2.0;
edge_colors[edge_artery_label > 0] = [0.8, 0.0, 0.0, 1.0]
# edge_colors[edge_vein_label>0] = [0.0,0.0,0.8,1.0]
print('plotting...')
p = p3d.plot_graph_mesh(gsrt, edge_colors=edge_colors, n_tube_points=5);

# %%############################################################################
### Voxelize Branch density
###############################################################################

voxelize_branch_parameter = {
    "method": 'sphere',
    "radius": (15, 15, 15),
    "weights": None,
    "shape": io.shape(annotation_file),
    "verbose": True
};

vertices = grt.vertex_property('coordinates_atlas');

branch_density = vox.voxelize(vertices, sink=ws.filename('density', postfix='branches15'), dtype='float32',
                              **voxelize_branch_parameter);

# %%############################################################################
voxelize_branch_parameter = {
    "method": 'sphere',
    "radius": (10, 10, 10),
    "weights": None,
    "shape": io.shape(reference_file),
    "verbose": True
};

vertices = grt.vertex_property('coordinates_atlas');

branch_density = vox.voxelize(vertices, sink=ws.filename('density', postfix='branches'), dtype='float32',
                              **voxelize_branch_parameter);

# %%############################################################################
### Convert npy to tif
###############################################################################


# create raw data npy files
io.convert_files(ws.file_list('stitched', extension='npy'), extension='tif',
                 processes=24, verbose=True);

# %%

# create artery data npy files
io.convert_files(ws.file_list('stitched', postfix='arteries', extension='tif'), extension='npy',
                 processes=24, verbose=True);



