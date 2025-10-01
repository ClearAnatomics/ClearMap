# A simple demo script to test partial loading of graph properties from a .gt file.
# Usage: python partial_gt_load_demo.py /path/to/graph.gt
# Please modify the path to ClearMap below if necessary.



import sys
import os

USER = os.getenv("USER")
sys.path.insert(0, f"/home/{USER}/code/ClearAnatomics/ClearMap")

from ClearMap.Analysis.graphs.graph_gt import Graph
from loguru import logger


def stat_graph_properties(g):
    return len(list(g.graph_properties)), len(list(g.vertex_properties)), len(list(g.edge_properties))

if __name__ == "__main__":

    # path of the graph file
    path = sys.argv[1] 

    logger.info("Scanning graph properties...")
    props = Graph.scan_gt_properties(path)
    logger.success(f"Found {len(props)} properties")
    props = Graph.scan_gt_properties(path, as_dict=True)
    logger.info(props)

    logger.info("Testing exclude edge_geometry_properties...")
    g = Graph.partial_load(path, exclude_edge_geometry_properties=True)
    n_gp, n_vp, n_ep = stat_graph_properties(g)
    logger.success(f"Graph properties: {n_gp}, Vertex properties: {n_vp}, Edge properties: {n_ep}")

    logger.info("Testing include dict...")
    g = Graph.partial_load(path, include_dict={'vertex': ['coordinates'], 'edge': ['length']})
    n_gp, n_vp, n_ep = stat_graph_properties(g)
    logger.success(f"Graph properties: {n_gp}, Vertex properties: {n_vp}, Edge properties: {n_ep}")

    logger.info("Testing exclude dict...")
    g = Graph.partial_load(path, exclude_dict={'vertex': ['coordinates'], 'edge': ['radii'], "graph": ['edge_geometry_coordinates', 'edge_geometry_radii', 'edge_geometry_coordinates_atlas', 'edge_geometry_radii_atlas', 'edge_geometry_annotation', 'edge_geometry_distance_to_surface', 'edge_geometry_coordinates_mri', 'edge_geometry_length_coordinates', 'edge_geometry_length_coordinates_mri', 'edge_geometry_length_coordinates_atlas']})
    n_gp, n_vp, n_ep = stat_graph_properties(g)
    logger.success(f"Graph properties: {n_gp}, Vertex properties: {n_vp}, Edge properties: {n_ep}")

    logger.info("Testing include...")
    g = Graph.partial_load(path, include=['edge_geometry_coordinates', "shape", 'edge_geometry_type', "coordinates"] )
    n_gp, n_vp, n_ep = stat_graph_properties(g)
    logger.success(f"Graph properties: {n_gp}, Vertex properties: {n_vp}, Edge properties: {n_ep}")

    logger.info("Testing exclude...")
    g = Graph.partial_load(path, exclude=['edge_geometry_coordinates', "shape", 'edge_geometry_type', "coordinates"])
    n_gp, n_vp, n_ep = stat_graph_properties(g)
    logger.success(f"Graph properties: {n_gp}, Vertex properties: {n_vp}, Edge properties: {n_ep}")
