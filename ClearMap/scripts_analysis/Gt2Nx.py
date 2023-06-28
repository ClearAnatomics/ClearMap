import networkx as nx
import graph_tool as gt
import networkx.algorithms.similarity as nxas
import numpy as np
from networkx.algorithms.community.quality import modularity
from ClearMap.vi import *

def gt2nx(gt_g, weight=None):
    G = nx.Graph()
    print(gt_g)
    coordinates=gt_g.vertex_coordinates()
    artery=gt_g.vertex_property('artery')
    vein=gt_g.vertex_property('vein')
    av=np.logical_or(artery, vein).astype(int)
    radii=gt_g.edge_property('radii')
    rad=gt_g.vertex_property('radii')
    print(rad.shape)
    print(vein.shape)
    try:
        G.add_nodes_from(np.arange(gt_g.n_vertices))#gt_g.n_vertices-1
        connectivity = gt_g.edge_connectivity()
    except:
        G.add_nodes_from(np.arange(gt_g.num_vertices()))
        connectivity = [(e[0], e[1]) for e in gt_g.get_edges()]
    if weight != None:
        if weight.all()!=None:
            # connectivity=np.array(connectivity)
            print(weight.shape, len(connectivity))
            weighted_connectivity=np.concatenate((connectivity,np.expand_dims(weight, axis=1)), axis=1)
            G.add_weighted_edges_from(weighted_connectivity)
    else:
        G.add_edges_from(connectivity)
        # for i, k in enumerate(connectivity):
        #     G.add_edge(k[0],k[1],key=k, new_attr= radii[i])

    xx=dict(enumerate(coordinates[:, 0].tolist()))
    yy=dict(enumerate(coordinates[:, 1].tolist()))
    zz=dict(enumerate(coordinates[:, 2].tolist()))
    rad=dict(enumerate(rad.tolist()))
    AV=dict(enumerate(av.tolist()))

    for i, k in enumerate(connectivity):
        G[k[0]][k[1]]['radii']=radii[i]
        # G.edge[itr[0][itr[1]]['radii']= radii[i]
    # for i, k in enumerate(connectivity):
        #     G.add_edge(k[0],k[1],key=k, new_attr= radii[i])
    nx.set_node_attributes(G, xx, "coordX")
    nx.set_node_attributes(G,yy, "coordY")
    nx.set_node_attributes(G, zz, "coordZ")
    nx.set_node_attributes(G, AV, "class")
    nx.set_node_attributes(G, rad, "rad")






    return G


def GED(g1, g2):
    ged=nxas.graph_edit_distance(g1, g2)
    return ged


def get_modularity(g, modules):
    gnx=gt2nx(g)
    partition=getpartition(modules)
    res=modularity(gnx,partition)
    return(res)



def get_modularity_gt(g, modules):
    gnx=gt2nx(g)
    partition=getpartition(modules)
    res=modularity(gnx,partition)
    return(res)

# control='0b'
# nx.write_graphml_lxml(N0b, work_dir+ '/' + control + '/' + str(control)+'_graph.graphml')
# nx.write_gpickle(N0b, work_dir+ '/' + control + '/' + str(control)+'_graph.gpickle')
# N0b = nx.read_gpickle(work_dir+ '/' + control + '/' + str(control)+'_graph.gpickle')