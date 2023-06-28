
controls=['2R','3R','5R', '8R']
mutants=['1R','7R', '6R', '4R']
work_dir='/data_SSD_2to/191122Otof'
states=[controls, mutants]

region_list = [(54, 9), (47, 9)] # , (75, 9)]  # barrels

control='2R'
print(control)
graph = ggt.load(work_dir + '/' + control + '/' + 'data_graph_correcteduniverse.gt')
vertex_filter = np.zeros(graph.n_vertices)
for i, rl in enumerate(region_list):
    order, level = region_list[i]
    print(level, order, ano.find(order, key='order')['name'])
    label = graph.vertex_annotation();
    label_leveled = ano.convert_label(label, key='order', value='order', level=level)
    vertex_filter[label_leveled == order] = 1;
graph = graph.sub_graph(vertex_filter=vertex_filter)


distance_file=io.read('/data_SSD_2to/cFosDeprived/3weeks-secondbatch/Annotation_180305_FOS_distance_atlas.tif')

coord=graph.edge_geometry_property('coordinates')

X=coord[:, 0]
Y=coord[:, 1]
Z=coord[:, 2]

minX=np.min(X)
minY=np.min(Y)
minZ=np.min(Z)

maxX=np.max(X)
maxY=np.max(Y)
maxZ=np.max(Z)

distance_file=distance_file[int(minX*1.62/25):,int(minY*1.62/25):,int(minZ*1.62/25):]


X=X-minX
maxX=np.max(X)

Y=Y-minY
maxY=np.max(Y)

Z=Z-minZ
maxZ=np.max(Z)

binary=np.ones((int(maxX)+1, int(maxY)+1, int(maxZ)+1))
global output
output=np.zeros((int(maxX)+1, int(maxY)+1, int(maxZ)+1))

binary[X.astype(int), Y.astype(int), Z.astype(int)]=0

patch_size=(500,500,500)

from scipy import ndimage
from skimage.morphology import skeletonize

def computeDistanceTransform(args):
    i, j, k=args
    print(i, j, k)
    dt=ndimage.distance_transform_edt(binary[i:i+patch_size[0],j:j+patch_size[1],k:k+patch_size[2]])
    dt = skeletonize(dt>10, method='lee')
    # print(np.unique(dt))
    # output[i:i+patch_size[0],j:j+patch_size[1],k:k+patch_size[2]]=dt
    return dt


def skeletonizaion(args):
    i, j, k=args
    print(i, j, k)
    dt=skeletonize(binary[i:i+patch_size[0],j:j+patch_size[1],k:k+patch_size[2]], method='lee')
    # print(np.unique(dt))
    # output[i:i+patch_size[0],j:j+patch_size[1],k:k+patch_size[2]]=dt
    return dt




import tqdm

import multiprocessing

args=[]
for i in range(0, binary.shape[0] - patch_size[0], patch_size[0]):
    for j in range(0, binary.shape[1] - patch_size[1], patch_size[1]):
        for k in range(0, binary.shape[2] - patch_size[2], patch_size[2]):
            args.append([i, j, k])

pool=multiprocessing.Pool(10)
with pool:
   res=[pool.map(computeDistanceTransform, args)]
   # res = [pool.map(skeletonizaion, args)]


for n, r in enumerate(res[0]):
    output[args[n][0]:args[n][0]+patch_size[0],args[n][1]:args[n][1]+patch_size[1],args[n][2]:args[n][2]+patch_size[2]]=r

p3d.plot(output)
skeleton=output.copy()

skeleton[0, :, :]=0
skeleton[:, 0, :]=0
skeleton[:, :, 0]=0
skeleton[:, :, int(maxZ)]=0
skeleton[:, int(maxY), :]=0
skeleton[:, int(maxY), :]=0
skeleton[int(maxX), :, :]=0

np.save('/data_SSD_2to/191122Otof/DT/skeletonSnout.npy', skeleton)

p3d.plot([skeleton])



import ClearMap.Analysis.Graphs.GraphProcessing as gp

graph_raw = gp.graph_from_skeleton(skeleton, verbose=True)
radii=np.ones(graph_raw.n_vertices)
graph_raw.set_vertex_radii(radii)

graph_cleaned = gp.clean_graph(graph_raw,
                               vertex_mappings={'coordinates': gp.mean_vertex_coordinates,'radii': np.max},
                               verbose=True)


graph_reduced = gp.reduce_graph(graph_cleaned, edge_length=True,
                                edge_to_edge_mappings={'length': np.sum},
                                vertex_to_edge_mappings={'radii': np.max},
                                edge_geometry_vertex_properties=['coordinates', 'radii'],
                                edge_geometry_edge_properties=None,
                                return_maps=False, verbose=True)


# %% save
# graph_cleaned.save(ws.filename('graph', postfix='cleaned'))


def transformation(coordinates):
    coordinates=coordinates*1.62/25
    return coordinates;
graph_reduced.transform_properties(transformation=transformation,
                                   vertex_properties={'coordinates': 'coordinates_atlas'},
                                   edge_geometry_properties={'coordinates': 'coordinates_atlas'},
                                   verbose=True);
graph_reduced.save('/data_SSD_2to/191122Otof/DT/graph_reduced_transformed.gt')
# graph_reduced.save('/data_SSD_2to/191122Otof/DT/graph_reduced.gt')

gs = graph_reduced.sub_slice((slice(1,320), slice(290,310), slice(1,228)))
p3d.plot_graph_mesh(gs)

graphDT=ggt.load('/data_SSD_2to/191122Otof/DT/graph_reduced_transformed.gt')
gs = graphDT.sub_slice((slice(1,8000), slice(2400,2800), slice(1,8000)))
p3d.plot_graph_mesh(gs)
# distance_file=np.swapaxes(distance_file, 0,2)
distance_file_flipped = np.flip(distance_file,axis=0)
distance2surface=[distance_file_flipped[((coord[0])*1.62/25).astype(int), ((coord[1])*1.62/25).astype(int), ((coord[2])*1.62/25).astype(int)] for coord in graphDT.vertex_property('coordinates')]

plt.figure()
plt.hist(distance2surface, bins=20)
plt.xticks(size='xx-large')
plt.yticks(size='xx-large')



