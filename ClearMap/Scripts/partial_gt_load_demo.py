"""
A simple demo script to test partial loading of graph properties from a .gt file.

Usage:
    python partial_gt_load_demo.py /path/to/graph.gt

Please uncomment and modify the path to ClearMap below if necessary.
"""

import sys

import logging

# from pathlib import Path
# sys.path.insert(0, str(Path('~/code/ClearAnatomics/ClearMap').expanduser()))
from ClearMap.Analysis.graphs.graph_gt import Graph

# ######## Just the boilerplate to get pretty logging without external dependencies  ########
SUCCESS_LEVEL_NUM = 25
logging.addLevelName(SUCCESS_LEVEL_NUM, 'SUCCESS')

class SuccessLogger(logging.Logger):
    def success(self, msg, *args, **kwargs):
        if self.isEnabledFor(SUCCESS_LEVEL_NUM):
            self._log(SUCCESS_LEVEL_NUM, msg, args, **kwargs)

logging.setLoggerClass(SuccessLogger)

class ColorFormatter(logging.Formatter):
    GREEN = "\033[32m" if sys.platform != "win32" else ""  # Not supported on Windows
    RESET = "\033[0m" if sys.platform != "win32" else ""
    def format(self, record: logging.LogRecord) -> str:
        s = super().format(record)
        return f"{self.GREEN}{s}{self.RESET}" if record.levelname == "SUCCESS" else s

logger: SuccessLogger = logging.getLogger('partial_gt_load_demo')
logger.setLevel(logging.INFO)

if not logger.handlers:
    _handler = logging.StreamHandler(sys.stdout)
    _handler.setFormatter(ColorFormatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
    logger.addHandler(_handler)

# ##########################################################################################


def stat_graph_properties(g):
    """
    Count the number of graph, vertex, and edge properties in the given graph.

    Parameters
    ----------
    g: Graph
        The graph object to analyze.

    Returns
    -------
    tuple: (n_graph_properties, n_vertex_properties, n_edge_properties)
        The counts of graph, vertex, and edge properties.
    """
    n_g_props = len(list(g.graph_properties))
    n_v_props = len(list(g.vertex_properties))
    n_e_props = len(list(g.edge_properties))
    return n_g_props, n_v_props, n_e_props


if __name__ == '__main__':
    if len(sys.argv) != 2:
        logger.error("Usage: python %s /path/to/graph.gt", sys.argv[0])
        sys.exit(1)

    # path of the graph file
    graph_path = sys.argv[1]

    logger.info('Scanning graph properties...')
    props = Graph.scan_gt_properties(graph_path)
    logger.success(f'Found {len(props)} properties')
    props = Graph.scan_gt_properties(graph_path, as_dict=True)
    logger.info(props)

    logger.info('Testing exclude edge_geometry_properties...')
    g = Graph.partial_load(graph_path, exclude_edge_geometry_properties=True)
    n_gp, n_vp, n_ep = stat_graph_properties(g)
    logger.success(f'Graph properties: {n_gp}, Vertex properties: {n_vp}, Edge properties: {n_ep}')

    logger.info('Testing include dict...')
    g = Graph.partial_load(graph_path, include_dict={'vertex': ['coordinates'], 'edge': ['length']})
    n_gp, n_vp, n_ep = stat_graph_properties(g)
    logger.success(f'Graph properties: {n_gp}, Vertex properties: {n_vp}, Edge properties: {n_ep}')

    logger.info('Testing exclude dict...')
    g = Graph.partial_load(graph_path, exclude_dict={
        'vertex': ['coordinates'],
        'edge': ['radii'],
        'graph': ['edge_geometry_coordinates', 'edge_geometry_radii', 'edge_geometry_coordinates_atlas',
                  'edge_geometry_radii_atlas', 'edge_geometry_annotation', 'edge_geometry_distance_to_surface',
                  'edge_geometry_coordinates_mri', 'edge_geometry_length_coordinates',
                  'edge_geometry_length_coordinates_mri', 'edge_geometry_length_coordinates_atlas']
    })
    n_gp, n_vp, n_ep = stat_graph_properties(g)
    logger.success(f'Graph properties: {n_gp}, Vertex properties: {n_vp}, Edge properties: {n_ep}')

    logger.info('Testing include...')
    g = Graph.partial_load(graph_path, include=['edge_geometry_coordinates', 'shape', 'edge_geometry_type', 'coordinates'])
    n_gp, n_vp, n_ep = stat_graph_properties(g)
    logger.success(f'Graph properties: {n_gp}, Vertex properties: {n_vp}, Edge properties: {n_ep}')

    logger.info("Testing exclude...")
    g = Graph.partial_load(graph_path, exclude=['edge_geometry_coordinates', 'shape', 'edge_geometry_type', 'coordinates'])
    n_gp, n_vp, n_ep = stat_graph_properties(g)
    logger.success(f'Graph properties: {n_gp}, Vertex properties: {n_vp}, Edge properties: {n_ep}')
