
import sys
import os
import glob

import numpy as np
import matplotlib.pyplot as plt

from importlib import reload

###############################################################################
### ClearMap
###############################################################################

#generic
import ClearMap.Settings as settings

import ClearMap.IO.IO as io
# import ClearMap.IO.Workspace as wsp
#
# import ClearMap.Tests.Files as tfs

import ClearMap.Visualization.Plot3d as p3d
# import ClearMap.Visualization.Color as col
#
# import ClearMap.Utils.TagExpression as te
# import ClearMap.Utils.Timer as tmr
#
# import ClearMap.ParallelProcessing.BlockProcessing as bp
# import ClearMap.ParallelProcessing.DataProcessing.ArrayProcessing as ap

#alignment
import ClearMap.Alignment.Annotation as ano
import ClearMap.Alignment.Resampling as res
import ClearMap.Alignment.Elastix as elx
# import ClearMap.Alignment.Stitching.StitchingRigid as st
# import ClearMap.Alignment.Stitching.StitchingWobbly as stw

#image processing
# import ClearMap.ImageProcessing.Clipping.Clipping as clp
# import ClearMap.ImageProcessing.Filter.Rank as rnk
# import ClearMap.ImageProcessing.Filter.StructureElement as se
# import ClearMap.ImageProcessing.Differentiation as dif
# import ClearMap.ImageProcessing.Skeletonization.Skeletonization as skl
# import ClearMap.ImageProcessing.Skeletonization.SkeletonProcessing as skp


#analysis
import ClearMap.Analysis.Graphs.GraphGt_old as grp
import ClearMap.Analysis.Graphs.GraphProcessing as gp

import ClearMap.Analysis.Measurements.MeasureExpression as me
import ClearMap.Analysis.Measurements.MeasureRadius as mr
import ClearMap.Analysis.Measurements.Voxelization as vox



#%%############################################################################
transfo_dir='/data_2to/alignement/atlases/new_region_atlases/P5/transfo/'
wor_dir='/data_2to/p5/'
brains=['5a', '5b']
# transfo_dirs=['/data_2to/alignement/atlases/new_region_atlases/P5/transfo/5a/elastix_auto_to_chosen_auto',
#       '/data_2to/alignement/atlases/new_region_atlases/P5/transfo/5b/elastix_auto_to_chosen_auto']
### Atlas registration and annotation

for j , b in enumerate(brains):
    print(wor_dir+b+'/'+b+'_graph_correcteduniverse.gt')
    graph=grp.load(wor_dir+b+'/'+b+'_graph_correcteduniverse.gt')
    annotation_file= '/data_2to/alignement/atlases/new_region_atlases/P7/smoothed_anot_P5.nrrd'
    transform_directory= '/data_2to/alignement/atlases/new_region_atlases/P5/transfo/5b/elastix_auto_to_chosen_auto'


    #%% graph atlas registration

    def transformation(coordinates_atlas):

        coordinates = elx.transform_points(
            coordinates_atlas, sink=None,
            transform_directory=transform_directory,
            binary=True, indices=False);

        return coordinates;

    graph.transform_properties(transformation=transformation,
                                       vertex_properties = {'coordinates_atlas' : 'coordinates_atlas'},
                                       edge_geometry_properties = {'coordinates_atlas' : 'coordinates_atlas'},
                                       verbose=True);

    def annotation(coordinates):
        label = ano.label_points(coordinates, key='order');
        return label;

    graph.annotate_properties(annotation,
                              vertex_properties = {'coordinates_atlas' : 'annotation'},
                              edge_geometry_properties = {'coordinates_atlas' : 'annotation'});

    graph.save(wor_dir+'/'+b+'/'+b+'_graph_correcteduniverse_smoothed.gt')




#%% annotation
import ClearMap.Settings as settings
atlas_path = os.path.join(settings.resources_path, 'Atlas');

wor_dir='/data_2to/'
# sub_dir=['p0/new','p6']
sub_dir=['p14']
# brains=[['0a', '0b', '0c', '0d'],
#         ['6a']]
brains=[[ '14c']]
# anot_f=['/data_2to/alignement/atlases/new_region_atlases/P1/smoothed_anot_P1.nrrd',
#         '/data_2to/alignement/atlases/new_region_atlases/P5/smoothed_anot_P5_half.nrrd']
anot_f=['/data_2to/alignement/atlases/new_region_atlases/P14/ano_P14_p10_corrected_smoothed_rescale.tif']
# reference_files=['/data_2to/alignement/atlases/new_region_atlases/P1/normed_max_atlas_auto_new.nrrd',
#                  '/data_2to/alignement/atlases/new_region_atlases/P5/normed_max_atlas_auto_new.nrrd']
reference_files=['/data_2to/alignement/atlases/new_region_atlases/P14/normed_max_atlas_auto_rescale.tif']
# sub_dir=['p1','p3','p5', 'p7']
#
# brains=[['1a', '1b', '1d'],
#         ['3a', '3b', '3c', '3d'],
#         ['5a', '5b'],
#         ['7a', '7b']]
# anot_f=['/data_2to/alignement/atlases/new_region_atlases/P1/smoothed_anot_P1.nrrd',
#         '/data_2to/alignement/atlases/new_region_atlases/P3/smoothed_anot_P3_V6.nrrd',
#         '/data_2to/alignement/atlases/new_region_atlases/P5/smoothed_anot_P5_half.nrrd',
#         '/data_2to/alignement/atlases/new_region_atlases/P7/smoothed_anot_P7.nrrd']
#
# reference_files=['/data_2to/alignement/atlases/new_region_atlases/P1/normed_max_atlas_auto_new.nrrd',
#                 '/data_2to/alignement/atlases/new_region_atlases/P3/normed_max_atlas_auto_new.nrrd',
#                 '/data_2to/alignement/atlases/new_region_atlases/P5/normed_max_atlas_auto_new.nrrd',
#                 '/data_2to/alignement/atlases/new_region_atlases/P7/normed_max_atlas_auto_new.nrrd']
# ref_brain=['1b', '3b', '7a']

resample_parameter = {
    "source_resolution" : (1.625,1.625,2),
    "sink_resolution"   : (16.752,16.752,20),
    "processes" : 8,
    "verbose" : True,
};
align_reference_affine_file='/home/sophie.skriabine/Projects/ClearMap3/ClearMap/ClearMap/Resources/Alignment/align_affine.txt'
align_reference_bspline_file='/home/sophie.skriabine/Projects/ClearMap3/ClearMap/ClearMap/Resources/Alignment/align_bspline.txt'
# res.resample(ws.filename('autofluorescence'), sink=ws.filename('resampled', postfix='autofluorescence'), **resample_parameter_auto)


for i, dir in enumerate(sub_dir):

    # ref_b=ref_brain[i]
    reference_file=reference_files[i]
    annotation_file=anot_f[i]
    ano.initialize(label_file = os.path.join(atlas_path, 'annotation.json'),
                   extra_label = None, annotation_file = annotation_file)
    transform_directory=wor_dir+dir+'/transfo'

    for j , b in enumerate(brains[i]):

        print(wor_dir+dir+'/'+b+'/'+b+'_graph_correcteduniverse_smoothed.gt')
        graph=grp.load(wor_dir+dir+'/'+b+'/'+b+'_graph.gt')
        autofluo=io.read(transform_directory+'/'+b+'_resampled_autofluorescence.tif')


        # if b==ref_b:
        #
        #     transfo1=transform_directory+'/'+b+'_elastix_resampled_to_auto'
        #
        #     def transformation(coordinates):
        #
        #         coordinates = res.resample_points(
        #             coordinates,sink_shape=autofluo.shape,**resample_parameter);
        #
        #         coordinates = elx.transform_points(
        #             coordinates, sink=None,
        #             transform_directory=transfo1,
        #             binary=True, indices=False);
        #
        #         return coordinates;
        #
        #     graph.transform_properties(transformation=transformation,
        #                                vertex_properties = {'coordinates' : 'coordinates_atlas'},
        #                                edge_geometry_properties = {'coordinates' : 'coordinates_atlas'},
        #                                verbose=True);
        #
        # else:


        align_reference_parameter = {
            #moving and reference images
            "moving_image" : reference_file,
            "fixed_image"  : transform_directory+'/'+b+'_resampled_autofluorescence.tif',

            #elastix parameter files for alignment
            "affine_parameter_file"  :  align_reference_affine_file,
            "bspline_parameter_file" :  align_reference_bspline_file,
            #directory of the alignment result
            "result_directory" :  transform_directory+'/'+b+'_elastix_chosen_auto_to_auto'
        };

        elx.align(**align_reference_parameter);

        transfo1=transform_directory+'/'+b+'_elastix_resampled_to_auto_new'
        transfo2=transform_directory+'/'+b+'_elastix_chosen_auto_to_auto'

        def transformation(coordinates):

            coordinates = res.resample_points(
                coordinates,sink_shape=autofluo.shape,**resample_parameter);

            coordinates1 = elx.transform_points(
                            coordinates, sink=None,
                            transform_directory=transfo1,
                            binary=True, indices=False);

            coordinates2 = elx.transform_points(
                            coordinates1, sink=None,
                            transform_directory=transfo2,
                            binary=True, indices=False);

            return coordinates2;

        graph.transform_properties(transformation=transformation,
                                   vertex_properties = {'coordinates' : 'coordinates_atlas'},
                                   edge_geometry_properties = {'coordinates' : 'coordinates_atlas'},
                                   verbose=True);






        ano.set_annotation_file(annotation_file)
        def annotation(coordinates_atlas):
            label = ano.label_points(coordinates_atlas, key='order');
            return label;

        graph.annotate_properties(annotation,
                                          vertex_properties = {'coordinates_atlas' : 'annotation'},
                                          edge_geometry_properties = {'coordinates_atlas' : 'annotation'});

        graph.save(wor_dir+dir+'/'+b+'/'+b+'_graph_correcteduniverse_smoothed.gt')



#%% Plot_cortex_color_dev
graph = grp.load('/data_2to/p14/14b/14b_graph_correcteduniverse_smoothed.gt')
#%%
ano.initialize(label_file = os.path.join(atlas_path, 'annotation.json'),
               extra_label = None, annotation_file = '/data_2to/alignement/atlases/new_region_atlases/P14/ano_P14_p10_corrected_smoothed.nrrd')
# gs = grt.sub_slice((slice(0,5000), slice(0,5000), slice(1300,1400)));
gs = graph.sub_slice((slice(0,700), slice(0,700), slice(150,160)),coordinates='coordinates_atlas');
vertex_colors = ano.convert_label(gs.vertex_annotation(), key='order', value='rgba');
connectivity = gs.edge_connectivity();
edge_colors = (vertex_colors[connectivity[:,0]] + vertex_colors[connectivity[:,1]])/2.0;
p = p3d.plot_graph_mesh(gs, edge_colors=edge_colors, n_tube_points=5);


coordinates=graph.vertex_property('coordinates')
print(np.min(coordinates[:, 0]), np.max(coordinates[:, 0]))
print(np.min(coordinates[:, 1]), np.max(coordinates[:, 1]))
print(np.min(coordinates[:, 2]), np.max(coordinates[:, 2]))