import time
from pathlib import Path

import cProfile as profile

import numpy as np
from matplotlib import pyplot as plt
from vispy import app

from ClearMap.IO import IO as cmp_io

from ClearMap.Analysis.graphs import graph_processing, graph_gt
from ClearMap.Analysis.vasculature import graph_corrections
from ClearMap.Visualization.Vispy.Plot3d import initialize_view
from ClearMap.Visualization.Vispy.plot_graph_3d import (plot_graph_nodes, plot_graph_mesh, plot_graph_line,
                                                        plot_graph_edge_property, plot_graph_edge_geometry_property)


def skeleton_2d_to_3d(skel_path, shift_range=False):
    skel_path = Path(skel_path)

    skel = cmp_io.read(skel_path).array
    if shift_range:
        skel = skel - 1  # If was drawn with 2 meaning on, 1 meaning off, 0 not existing
    skel = skel.astype(np.uint8)
    skel = np.expand_dims(skel, axis=2)  # Add a third dimension to make it 3D
    skel = np.pad(skel, 1, mode='constant', constant_values=0)
    cmp_io.write(skel_path.with_stem('skeleton_cmp'), skel)


def print_graph_lengths(graph):
    for edge in graph.edges:
        coordinates = graph.edge_geometry('coordinates', edge=edge)
        print(f'Edge {edge}, id={graph.edge_property("chain_id", edge=edge)} '
              f'with coordinates {coordinates}, '
              f'has length {graph.edge_property("length", edge=edge):.2f} um, ')
        
"""
# raw graph
branch_edges = [
    [(4, 0), (0, 3), (3, 4)],  # 0
    [(4, 7), (7, 8), (8, 16)],  # 1
    [(15, 6), (6, 2)],  # 2
    [(16, 9), (9, 10), (10, 11), (11, 12)],  # 3
    [(12, 13), (13, 14), (14, 15)],  # 4
    [(15, 17), (17, 19)],  # 5
    [(16, 18), (18, 20)],  # 6
    [(1, 5), (5, 12)]  # 7
]
coordinates = g.vertex_coordinates()
branch_v_coords = [
    [tuple([coordinates[v][:2] for v in edge])
     for edge in branch]
    for branch in branch_edges
]
branch_lengths_elements_raw = [[g.edge_property('length', edge=edge) for edge in edges] for edges in branch_edges]
branch_lengths_raw = [f'{np.sum(lengths):.2f}' for lengths in branch_lengths_elements_raw]
branch_lengths_elements_raw = [[f'{length:.2f}' for length in lengths] for lengths in branch_lengths_elements_raw]

# Graph cleaned
branch_edges = [
    [(4, 0), (0, 3), (3, 4)],       # 0
    [(4, 7), (7, 8), (8, 16)],      # 1
    [(15, 6), (6, 2)],              # 2
    [(16, 9), (9, 10), (10, 21)],   # 3
    [(21, 14), (14, 15)],           # 4
    [(15, 17), (17, 19)],           # 5
    [(16, 18), (18, 20)],           # 6
    [(21, 1)]                       # 7
]
branch_lengths_elements = [[g.edge_property('length', edge=edge) for edge in edges] for edges in branch_edges]
branch_lengths = [f'{np.sum(lengths):.2f}' for lengths in branch_lengths_elements]
branch_lengths_elements = [[f'{length:.2f}' for length in lengths] for lengths in branch_lengths_elements]
"""

data_dir = Path()  # Where you store large skeletons, graphs, etc.

def main(skel_path):
    base_dir = base_dir = Path(__file__).resolve().parent

    skel_path = Path(skel_path)
    if not skel_path.is_absolute():
        if str(data_dir):
            skel_path = data_dir / skel_path
        else:
            skel_path = base_dir / skel_path

    # graph_raw = graph_gt.load(base_dir / 'test_graph.gt')

    prof = profile.Profile()
    prof.enable()
    start_t = time.time()
    graph_raw = graph_processing.graph_from_skeleton(skel_path, spacing=(1.625, 1.625, 2.5), verbose=True)
    prof.disable()
    prof.dump_stats(base_dir / 'test_graph_from_skeleton.prof')


    title = 'test_graph_reduced'
    bg_color = 'grey'
    view = initialize_view(None, title=title, depth_value=100000000, fov=100, distance=5,
                           elevation=90, azimuth=90, show=True, bg_color=bg_color)

    # plot_graph_nodes(graph_raw, view=view, radii=0.25)

    # graph_raw.save(base_dir / 'test_graph.gt')
    # graph_raw = graph_gt.load(base_dir / 'test_graph.gt')

    prof = profile.Profile()
    prof.enable()
    cleaned_graph = graph_processing.clean_graph(graph_raw, verbose=True)
    prof.disable()
    prof.dump_stats(base_dir / 'test_graph_clean.prof')
    #
    # cleaned_graph.save(base_dir / 'test_graph_cleaned.gt')


    def vote(expression):  # FIXME: use
        return np.sum(expression) >= len(expression) / 1.5

    vertex_to_edge_mappings = {'radii': np.max}
    # edge_to_edge_mappings = {'length': np.sum}

    prof = profile.Profile()
    prof.enable()
    graph_reduced = graph_processing.reduce_graph(cleaned_graph, compute_edge_length=True,
                                                  edge_to_edge_mappings=None,
                                                  vertex_to_edge_mappings=vertex_to_edge_mappings,
                                                  # edge_geometry_vertex_properties=None,
                                                  # edge_geometry_edge_properties=None,
                                                  return_maps=False, verbose=True, label_branches=True,
                                                  save_modified_graph_path=base_dir / 'test_graph_cleaned.gt')
    print(f'Pipeline took {time.time() - start_t:.2f} seconds.')
    # print_graph_lengths(graph_reduced)
    prof.disable()
    prof.dump_stats(base_dir / 'test_graph_reduce.prof')

    graph_reduced.save(base_dir / 'test_graph_reduced.gt')

    # Reload with chain_id property
    cleaned_graph = graph_gt.load(base_dir / 'test_graph_cleaned.gt')

    edge_labels = cleaned_graph.edge_property('chain_id')
    colormap = plt.get_cmap('Set1')
    edge_colors = colormap(edge_labels % colormap.N)

    plot_graph_line(cleaned_graph, edge_colors=edge_colors, view=view, width=2, )
    # plot_graph_nodes(cleaned_graph, view=view, radii=0.25, vertex_colors=cleaned_graph.vertex_property('chain_id'))

    # graph_no_geom = graph_reduced.copy()  # To plot without edge geometry
    # plot_graph_3d.plot_graph_line(graph_no_geom, edge_colors=edge_colors, view=view)
    # plot_graph_3d.plot_graph_line(graph_no_geom, edge_colors=[[1, 0, 0, 1]], view=view, width=6,)
    print(f'Direct edges plotted for {cleaned_graph.n_edges} edges.')

    # plot_graph_3d.plot_graph_edge_property(cleaned_graph, edge_property='chain_id', colormap='Set1', view=view, width=7)
    # plot_graph_3d.plot_graph_mesh(graph_reduced, edge_colors=[[0, 1, 0, 0.75]], view=view, n_tube_points=5,
    #                               default_radius=0.5, radii=0.5)

    # plot_graph_edge_geometry_property(graph_reduced, 'chain_id', reduction_fn=np.min, mesh=True, alpha=0.3, colormap='Set1',
    #                                   n_tube_points=5, default_radius=0.2, radii=0.2, view=view)

    reduced_graph_no_edge_geom = graph_reduced.copy()  # To plot without edge geometry
    reduced_graph_no_edge_geom.remove_edge_geometry()  # remove edge geometry to plot without it
    # plot_graph_edge_property(reduced_graph_no_edge_geom, edge_property='chain_id', colormap='Set1', mesh=True,
    #                          view=view, n_tube_points=5, default_radius=0.2, radii=0.2)
    # FIXME: implement plot_edge_geometry_property to plot edge geometry properties

    # app.run()

    # #########################################   Prune and re-reduce   #########################################
    branch_length = graph_reduced.edge_property_map('length').a
    branch_length[0] = 0.2  # set a branch length to 0.2 to test the removal of spurious branches
    graph_reduced_pruned = graph_corrections.remove_spurious_branches(graph_reduced)
    graph_reduced_pruned = graph_reduced_pruned.sub_graph(vertex_filter=graph_reduced_pruned.vertex_degrees() > 0)

    # plot_graph_edge_geometry_property(graph_reduced_pruned, 'chain_id', reduction_fn=np.min, alpha=0.3, colormap='Set1',
    #                                   mesh=True, n_tube_points=5, default_radius=0.2, radii=0.2, view=view)

    # app.run()

    graph_reduced_pruned.save(base_dir / 'test_graph_reduced_corrected.gt')
    n_d_2s = len(graph_reduced_pruned.vertex_degrees() == 2)
    print(f'Reduced graph has {graph_reduced_pruned.n_vertices} vertices, {graph_reduced_pruned.n_edges} edges '
          f'and {n_d_2s} 2-degree vertices.')

    re_reduced = graph_processing.reduce_graph(graph_reduced_pruned, compute_edge_length=True,
                                               edge_to_edge_mappings=None,
                                               vertex_to_edge_mappings=vertex_to_edge_mappings,
                                               # edge_geometry_vertex_properties=None, edge_geometry_edge_properties=None,
                                               return_maps=False, verbose=True)


    # print_graph_lengths(re_reduced)

    # plot_graph_edge_geometry_property(re_reduced, 'chain_id', reduction_fn=np.min,  alpha=0.8, n_tube_points=5,
    #                                   colormap='Set1', default_radius=0.5, radii=0.5, view=view)


    plot_graph_edge_geometry_property(re_reduced, 'edge_chain_id', reduction_fn=np.min,
                                      mesh=True, alpha=0.8, n_tube_points=5, colormap='Set1',
                                      default_radius=0.25, radii=0.25, view=view)


    #  #######################################  Bridge between edge labeled 2 and 5 ###################################

    app.run()
    test_bridging = False
    if test_bridging:
        eprops = {name: re_reduced.edge_property(name) for name in ('length', 'chain_id', 'edge_geometry_indices')}
        re_reduced.add_edge((0, 4))
        egeom_size = re_reduced.edge_geometry_property('chain_id').size
        for eprop_name, new_val in zip(('length', 'chain_id', 'edge_geometry_indices'),
                                       (6, 8, np.array([[egeom_size, egeom_size + 5+2]]))):
            if not isinstance(new_val, np.ndarray):
                new_val = np.array([new_val])
            re_reduced.set_edge_property(eprop_name, np.concatenate([eprops[eprop_name], new_val]))

        for egeom_name in re_reduced.edge_geometry_properties:
            egeom = re_reduced.edge_geometry_property(egeom_name)
            if egeom_name == 'coordinates':  # WARNING: put first and last from other edges
                new_vals = np.array([[1, 13, 1], [1, 14, 1], [2, 15, 1], [3, 15, 1], [4, 15, 1], [5, 14, 1], [5, 13, 1]])
            elif egeom_name == 'chain_id':
                new_vals = np.array([2, 8, 8, 8, 8, 8, 5])
            elif egeom_name == 'edge_chain_id':
                new_vals = np.array([8, 8, 8, 8, 8, 8, 8])
            re_reduced.set_edge_geometry(egeom_name, np.concatenate([egeom, new_vals]))  # FIXME: could simplify by passing indices or edge

        re_reduced = graph_processing.reduce_graph(re_reduced, compute_edge_length=True,
                                                   edge_to_edge_mappings=None,
                                                   vertex_to_edge_mappings=vertex_to_edge_mappings,
                                                   return_maps=False, verbose=True)

    plot_graph_edge_geometry_property(re_reduced, 'edge_chain_id', reduction_fn=np.min,
                                      mesh=True, alpha=0.8, n_tube_points=5, colormap='Set1',
                                      default_radius=0.25, radii=0.25, view=view)

    app.run()
    print('done')


if __name__ == '__main__':
    skel_path='skeleton_cmp.npy'  # 20 voxels
    # skel_path='crop_1_vessels-arteries_skeleton.npy'  # 48M
    # skel_path='1_sub_skeleton.npy'   # 1.8G
    # skel_path='1_skeleton.npy'  # 131G
    main(skel_path)
