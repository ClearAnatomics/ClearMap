import numpy as np

import ClearMap.Settings as settings

import ClearMap.Alignment.Annotation as ano


import ClearMap.Visualization.Plot3d as p3d
import graph_tool.inference as gti

print('TEST')
import math
import matplotlib.pyplot as plt
import ClearMap.Analysis.Graphs.GraphGt as ggt
import graph_tool.centrality as gtc
print('loading...')
import numpy as np
import numexpr as ne
import graph_tool.topology as gtt
from sklearn import preprocessing

import math
pi=math.pi



def extract_AnnotatedRegion(graph, region):
    order, level = region
    print(level, order, ano.find_name(order, key='order'))

    label = graph.vertex_annotation();
    label_leveled = ano.convert_label(label, key='order', value='order', level=level)
    vertex_filter = label_leveled == order;

    gss4 = graph.sub_graph(vertex_filter=vertex_filter)
    return gss4, vertex_filter


def removeAutoLoops(graph):
    connectivity=graph.edge_connectivity()
    autoloops=np.asarray((connectivity[:, 0]-connectivity[:,1])==0).nonzero()[0]
    print(autoloops)
    ef=np.ones(graph.n_edges)
    for edge in autoloops:
        ef[edge]=0
    g=graph.sub_graph(edge_filter=ef)
    return g

def f_min(X,p):
    plane_xyz = p[0:3]
    distance = (plane_xyz*X.T).sum(axis=1) + p[3]
    return distance / np.linalg.norm(plane_xyz)

def residuals(params, signal, X):
    return f_min(X, params)

from scipy.optimize import leastsq


def get_edges_from_vertex_filter(prev_graph,vertex_filter):
  connectivity=prev_graph.edge_connectivity()
  edges=np.logical_and(vertex_filter[connectivity[:,0]], vertex_filter[connectivity[:,1]])
  return(edges)

def cleandeg4nodes(graph):
    degrees_m=graph.vertex_degrees()
    deg4=np.asarray(degrees_m==4).nonzero()[0]
    graph_cleaned=graph
    conn = graph.edge_connectivity()
    e_g = np.array(graph.edge_geometry())
    for j, d in enumerate(deg4):
        ns=graph.vertex_neighbours(d)
        vf=np.zeros(graph.n_vertices)

        vf[ns]=1
        vf[d]=1
        ef = get_edges_from_vertex_filter(graph, vf)
        conn_f=conn[ef]

        gs=graph.sub_graph(vertex_filter=vf)

        es=np.array(gs.edge_geometry())
        cs=gs.edge_connectivity()
        ds = gs.vertex_degrees()

        d4 = np.asarray(ds == 4).nonzero()[0]
        if isinstance(d4, (list, tuple, np.ndarray)):
            d4=d4[0]
        pts=[]
        pts_index=[]
        for i,c in enumerate(cs):
            # id=np.asarray(c==d4).nonzero()[0][0]
            if c[0]==d4:
                pts.append(np.array(es[i])[1])
                pts_index.append(i)
                if i==0:
                    pts.append(np.array(es[i])[0])
                    pts_index.append(100)
            elif c[1]==d4:
                pts.append(np.array(es[i])[-2])
                pts_index.append(i)
                if i==0:
                    pts.append(np.array(es[i])[-1])
                    pts_index.append(100)
            else:
                print('...')

        XYZ=np.array(pts).transpose()
        p0=[0,0,1,np.mean(XYZ[2])]
        sol = leastsq(residuals, p0, args=(None, XYZ))[0]
        x=sol[0:3]
        new_error=(f_min(XYZ, sol) ** 2).sum()
        old_error=(f_min(XYZ, p0) ** 2).sum()

        if new_error<1:
            z=np.array([0, 0, 1])
            costheta=np.dot(x,z)/(np.linalg.norm(x)*np.linalg.norm(z))
            # print(np.dot(x,z), (np.linalg.norm(x)*np.linalg.norm(z)))
            if abs(np.arccos(costheta))<0.3:
                print(j)
                print("Solution: ", x / np.linalg.norm(x), sol[3])
                print("Old Error: ", old_error)
                print("New Error: ", new_error)

                dn4 = np.asarray(ds != 4).nonzero()[0]
                coord=gs.vertex_coordinates()

                pos=[]
                neg=[]
                pos_e=[]
                neg_e = []
                e_i = np.asarray(ef == 1).nonzero()[0]
                v_i=np.asarray(vf == 1).nonzero()[0]
                for i, co in enumerate(np.array(pts)):
                    if i != d4:
                        res=sol[0]*co[0]+sol[1]*co[1]+sol[2]*co[2]+sol[3]
                        if res<0:
                            # neg.append((v_i[i], d))
                            neg.append((i, d4))
                        if res>=0:
                            # pos.append((v_i[i], d))
                            pos.append((i, d4))


                for p in pos:
                    for i,cf in enumerate(conn_f):
                        if p[0] in cf:
                            print(i)
                            pos_e.append(e_i[i])
                            break

                for n in neg:
                    for i,cf in enumerate(conn_f):
                        if n[0] in cf:
                            print(i)
                            neg_e.append(e_i[i])
                            break

                graph_cleaned.remove_vertex(d)
                newpos_edge=[]
                new_conn=[]
                for i, p in enumerate(pos_e):
                    newpos_edge.append(e_g[p])
                    new_conn.append(pos[i][0])
                newpos_edge=np.array(newpos_edge).ravel()


                graph_cleaned.add_edge((new_conn[0],new_conn[1]))

                newpos_edge = []
                new_conn = []
                for i, p in enumerate(neg_e):
                    newpos_edge.append(e_g[p])
                    new_conn.append(neg[i][0])
                newpos_edge = np.array(newpos_edge).ravel()

                graph_cleaned.add_edge((new_conn[0], new_conn[1]))



def removeSpuriousBranches(graph ,rmin=1, length=5):
    radii = graph.vertex_radii()
    conn=graph.edge_connectivity()
    degrees_m = graph.vertex_degrees()
    deg1 = np.asarray(degrees_m <= 1).nonzero()[0]
    rad1 = np.asarray(radii <= rmin).nonzero()[0]
    lengths = graph.edge_geometry_lengths()

    vertex2rm=[]
    for i in rad1:
        if i in deg1:
            # if lengths[i]<=length:
            vertex2rm.append(i)


    vertex2rm=np.array(vertex2rm)
    print(vertex2rm.shape)
    ef=np.ones(graph.n_vertices)
    ef[vertex2rm]=0
    graph=graph.sub_graph(vertex_filter=ef)
    return graph




def mutualLoopDetection(args):
    res=0
    ind, i, rmin, length, conn, radii, lengths = args
    co = conn[i]
    # print(ind)
    similaredges = np.logical_or(np.logical_and(conn[:, 0] == co[0], conn[:, 1] == co[1]),
                                 np.logical_and(conn[:, 1] == co[0], conn[:, 0] == co[1]))
    # print(similaredges.shape)
    similaredges = np.asarray(similaredges == True).nonzero()[0]

    if similaredges.shape[0] >= 2:
        rs = radii[similaredges]
        # print(rs)
        imin = np.argmin(rs)
        if rs[imin] <= rmin:
            if lengths[imin] <= length:
                print('adding edge to remove ', similaredges[imin])
                # e2rm.append(similaredges[imin])
                res=similaredges[imin]
    return res


def removeMutualLoop(graph, rmin=3, length=5):
    radii = graph.edge_radii()
    conn = graph.edge_connectivity()
    rad1 = np.asarray(radii <= rmin).nonzero()[0]
    print(rad1.shape)
    edge2rm = []
    lengths=graph.edge_geometry_lengths()

    n=0
    for i in rad1:
       
        co=conn[i]
        

        similaredges=np.logical_or(np.logical_and(conn[:,0]==co[0], conn[:, 1]==co[1]), np.logical_and(conn[:,1]==co[0], conn[:, 0]==co[1]))
        # print(similaredges.shape)
        similaredges=np.asarray(similaredges ==True).nonzero()[0]


        if similaredges.shape[0]>=2:
            rs=radii[similaredges]
            # print(rs)
            imin=np.argmin(rs)
            if rs[imin]<=rmin:
                if lengths[imin]<=length:
                    # print('adding edge to remove ', imin)
                    edge2rm.append(similaredges[imin])
                    # n = True

       

    edge2rm=np.array(edge2rm)
    print(edge2rm.shape)
    ef = np.ones(graph.n_edges)
    if edge2rm.shape[0] !=0:
        ef[edge2rm] = 0
        graph = graph.sub_graph(edge_filter=ef)
    return graph


def createHighLevelGraph(gss4):
   
    diff=np.load('/data_SSD_2to/' + brainnb + '/sbm/diffusion_penetrating_vessel_overlap_end_point_cluster_per_region_iteration_' + ano.find_name(order, key='order') +'_graph_corrected'+ '.npy')
    # diff=np.load('/mnt/data_SSD_2to/' + brainnb + '/sbm/diffusion_penetrating_art_end_point_cluster_per_region_'+ ano.find_name(order,key='order') + '.npy')
    gss4.add_vertex_property('diff_val', diff)
    diff=gss4.vertex_property('diff_val')
    coordinates=gss4.vertex_coordinates()
    u, c = np.unique(diff, return_counts=True)#diffusion_through_penetrating_arteries_vector
    high_lev_coord=np.zeros((u.shape[0], 3))
    conn = gss4.edge_connectivity()
    edges_all=[[diff[conn[i, 0]], diff[conn[i,1]]] for i in range(conn.shape[0])]
    edges_all=np.array(edges_all)
    g = ggt.Graph(n_vertices=u.shape[0], directed=False)
    # radii = np.zeros((0, 1), dtype=int)
    cluster_size = np.zeros(g.n_vertices)
    print(g)
    for i, uc in enumerate(u):
        vf=diff==uc
        high_lev_coord[i]=np.sum(coordinates[vf], axis=0)/c[i]
        cluster_size[uc] = c[i]

    intra_edges=np.asarray([edges_all[i, 0]==edges_all[i, 1] for i in range(edges_all.shape[0])]).nonzero()
    intra_edges=edges_all[intra_edges]

    inter_edges = np.asarray([edges_all[i, 0]!=edges_all[i, 1] for i in range(edges_all.shape[0])]).nonzero()
    inter_edges = edges_all[inter_edges]

    eu, ec=np.unique(inter_edges, return_counts=True, axis=0)
    g.add_edge(eu)
    # radii=np.ones(edges_all.shape[0])
    print(g)
    g.set_edge_geometry(name='radii', values=ec)
    g.set_vertex_coordinates(high_lev_coord)


    eu, ec=np.unique(intra_edges, return_counts=True, axis=0)

    inter_connectivity=np.zeros(g.n_vertices)

    for i, u in enumerate(eu):
        inter_connectivity[u]=ec[i]


    g.add_vertex_property('inter_connectivity', inter_connectivity)
    g.add_vertex_property('cluster_size', cluster_size)
    return g



def graphCorrection(graph, graph_dir, region, save=True):


    gss = graph.largest_component()

    degrees_m = gss.vertex_degrees()
    deg4 = np.asarray(degrees_m >= 5).nonzero()[0]
    print('deg4 init ', deg4.shape[0] / gss.n_vertices)

    gss = removeSpuriousBranches(gss, rmin=2.9, length=7)
    gss = remove_surface(gss, 2)
    gss = gss.largest_component()

    gss = removeAutoLoops(gss)
    

    rmin = 2.9
    length = 13
    radii = gss.edge_radii()
    lengths = gss.edge_geometry_lengths()

    conn = gss.edge_connectivity()
    rad1 = np.asarray(radii <= rmin)  # .nonzero()[0]
    len1 = np.asarray(lengths <= length)  # .nonzero()[0]
    l = np.logical_and(len1, rad1).nonzero()[0]

    print(l.shape)
  
    from multiprocessing import Pool
    p = Pool(20)
    import time
    start = time.time()

    e2rm = np.array(
        [p.map(mutualLoopDetection, [(ind, i, rmin, length, conn, radii, lengths) for ind, i in enumerate(l)])])

    end = time.time()
    print(end - start)

    print(gss)
    degrees_m = gss.vertex_degrees()
    deg4 = np.asarray(degrees_m >= 4).nonzero()[0]
    print('deg4 exit ', deg4.shape)

    u = np.unique(e2rm[0].nonzero())
    ef = np.ones(gss.n_edges)
    ef[u] = 0
    g = gss.sub_graph(edge_filter=ef)
   

    degrees_m = g.vertex_degrees()
    deg4 = np.asarray(degrees_m >= 5).nonzero()[0]
    print('deg4 exit ', deg4.shape[0] / g.n_vertices)

   
    if save:
        g.save(graph_dir+'/data_graph_corrected'+ano.find_name(region[0], key='id')+'.gt')
    return g
    
    
    
if __name__ == "__main__":
	
	region_list = [(0, 0)]
	mutants=[ '7o', '8c']
	work_dir='/media/sophie.skriabine/sophie/HFD_VASC'
	for c in mutants:
	    graph = ggt.load(work_dir + '/' + str(c) + '/' + str(c)+'_graph.gt')
	    giso=graphCorrection(graph, work_dir+'/'+c, region_list[0])
    

