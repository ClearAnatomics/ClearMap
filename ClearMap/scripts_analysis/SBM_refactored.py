
import graph_tool.inference as gti
import ClearMap.Analysis.Graphs.GraphGt as ggt
import ClearMap.Alignment.Annotation as ano



def load_graph(work_dir, control):
    """
    Loads the graph from the given work directory and control.
    """
    try:
        graph = ggt.load(work_dir + '/' + control + '/' + 'data_graph_correcteduniverse.gt')
    except:
        graph = ggt.load(work_dir + '/' + control + '/' + str(control)+'_graph.gt')

    # Filter out vertices with degrees <= 1 or > 4
    degrees = graph.vertex_degrees()
    vf = np.logical_and(degrees > 1, degrees <= 4)
    graph = graph.sub_graph(vertex_filter=vf)

    return graph

def extract_AnnotatedRegion(graph, region_id):

    print(region_id, ano.find(region_id, key='id')['name'])
    level=ano.find(region_id, key='id')['level']
    label = graph.vertex_annotation();
    label_leveled = ano.convert_label(label, key='id', value='id', level=level)
    vertex_filter = label_leveled == region_id;

    gss4 = graph.sub_graph(vertex_filter=vertex_filter)
    return gss4

def modularity_measure(partition, graph, vertex_prop):
    u, c= np.unique(partition, return_counts=True)
    vp=graph.vertex_property(vertex_prop)
    K=graph.n_edges
    # trash_clusters=u[np.where(c<20)]
    Q=0
    Qs=[]
    for e in u:
        vf=np.zeros(graph.n_vertices)
        vf[np.where(vp==e)[0]]=1
        cluster= graph.sub_graph(vertex_filter=vf)
        ms=cluster.n_edges
        ks=np.sum(cluster.vertex_degrees())
        Q=Q+(ms/K)-((ks/(2*K))**2)
        Qs.append((ms/K)-((ks/(2*K))**2))
    print(Q)
    return Q, Qs



if __name__ == '__main__':
    work_dir=''
    control=''
    region_id=1006

    # Load the graph
    graph = load_graph(work_dir, control)

    # Extract specific region
    graph = extract_AnnotatedRegion(graph, region_id)

    # Compute SBM
    state = gti.minimize_blockmodel_dl(graph.base)
    blocks_leveled = state.get_blocks().a

    # Compute modularity
    graph.add_vertex_property('sbm', blocks_leveled)
    Q, Qs = modularity_measure(blocks_leveled, graph, 'sbm')
    print('modularity : ',Q)


