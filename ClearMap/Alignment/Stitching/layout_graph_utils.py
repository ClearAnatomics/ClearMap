import numpy as np

try:
    import graph_tool as gt
    import graph_tool.topology as gtt
    graph_lib = 'graph_tool'
except ImportError:
    import igraph as gt
    graph_lib = 'igraph'


def get_graph(n_vertices, directed=False):
    g = gt.Graph(directed=directed)
    g.add_vertex(n_vertices)
    return g


def graph_to_connected_components(g):
    if graph_lib == 'graph_tool':
        connected_components, hist = gtt.label_components(g)
        connected_components = np.array(connected_components.a)
        n_components = len(hist)
    else:
        connected_components = g.clusters()
        _, n_components = np.unique(connected_components, return_counts=True)
    return connected_components, n_components


def get_connected_components(alignments, n_sources, source_to_index=None):
    """determine connected components"""
    g = get_graph(n_sources)
    if source_to_index is not None:
        for a in alignments:
            g.add_edge(source_to_index[a.pre], source_to_index[a.post])
    else:
        for a in alignments:
            g.add_edge(a[0], a[1])
    return graph_to_connected_components(g)


def connect_sources(alignments, n_sources, source_to_index):
    """construct minimal tree to connect sources"""
    graph = get_graph(n_sources, directed=True)
    if graph_lib == '':   # FIXME: check name
        pos_tree = graph.new_edge_property('vector<int>')  # FIXME: missing
    else:
        pos_tree = {}  # An object indexable by edge should suffice for our purpose
    for a in alignments:
        i_pre = source_to_index[a.pre]
        i_post = source_to_index[a.post]

        e = graph.add_edge(i_pre, i_post)
        pos_tree[e] = a.displacement

        e = graph.add_edge(i_post, i_pre)
        pos_tree[e] = tuple(-s for s in a.displacement)  # invert
    return graph, pos_tree
    
 
def get_positions_from_tree(g, fixed_source, source_to_index, n_sources, ndim, pos_tree):
    """calculate positions from tree"""
    start_id = 0
    if fixed_source is not None:
        start_id = source_to_index[fixed_source]
    positions = np.zeros((n_sources, ndim), dtype=int)
    for i in range(n_sources):
        if graph_lib == 'graph_tool':
            _, edge_list = gtt.shortest_path(g, g.vertex(start_id), g.vertex(i))
        else:
            edge_indices = g.get_shortest_paths(g.vs[3], g.vs[8], output='epath')[0]
            edge_list = [g.es[i] for i in edge_indices]
        for edge in edge_list:
            positions[i] += pos_tree[edge]
    return positions

  
def get_color_ids(sources, n_sources, color_ids):
    if color_ids is None:
        # find sources that overlap
        edges = []
        for i, s in enumerate(sources):
            p1 = np.array(s.position, dtype=int)
            s1 = s.shape
            for j in range(i + 1, n_sources):
                p2 = np.array(sources[j].position, dtype=int)
                s2 = sources[j].shape
                if np.all(np.max([p1, p2], axis=0) < np.min([p1 + s1, p2 + s2], axis=0)):
                    edges.append((i, j))

        # find graph coloring of overlap structure
        g = get_graph(n_vertices=n_sources)
        if graph_lib == 'graph_tool':
            g.add_edge_list(edges)
            color_ids = np.array(gtt.sequential_vertex_coloring(g).a)  # FIXME: missing
        else:
            g.add_edges(edges)
            color_ids = np.array(g.vertex_coloring_greedy())
    return color_ids


def cluster_components(components):
    """Find the connected components of the cluster components"""
    c_lens = [len(c) for c in components]
    c_ids = np.cumsum(c_lens)
    c_ids = np.hstack([0, c_ids])
    n_components = np.sum(c_lens)

    def is_to_c(s, i):
       return c_ids[s] + i

    def c_to_si(c):
        s = np.searchsorted(c_ids, c, side='right') - 1
        i = c - c_ids[s]
        return s,i

    g = get_graph(n_components)
    for s in range(1, len(components)):
        for i, ci in enumerate(components[s - 1]):
            for j, cj in enumerate(components[s]):
                for c in ci:
                    if c in cj:
                        g.add_edge(is_to_c(s - 1, i), is_to_c(s, j))
                        break
    connected_components, n_components = graph_to_connected_components(g)

    components_full = [np.where(connected_components == i)[0] for i in range(n_components)]

    # remove isolated nodes
    components_full = [c for c in components_full if len(c) > 1]

    return components_full, is_to_c, c_to_si
