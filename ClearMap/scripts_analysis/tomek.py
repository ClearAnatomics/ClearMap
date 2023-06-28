import matplotlib.pyplot as plt
import numpy as np
import ClearMap.Alignment.Annotation as ano
import ClearMap.Analysis.Graphs.GraphGt as ggt
import seaborn as sns
import ClearMap.IO.IO as io
import json



def get_volume_region(region_leaves, atlas):
    val=0
    for l in region_leaves:
        id=ano.find(l[0], key='order')['id']
        print(ano.find(l[0], key='order')['name'], 'volume: ',np.sum(atlas==id))
        val=val+(np.sum(atlas==id))
    return val*1.5625e-5 #convrsion from atlas voxel to mm3




with open('/home/sophie.skriabine/Projects/clearVessel_New/ClearMap/ClearMap/Resources/Atlas/annotation.json') as json_data:
    data_dict = json.load(json_data)['msg']
    print(data_dict)

atlas=io.read(ano.default_annotation_file)
leafArray=[]
names=[]
annotated_region=np.unique(atlas)
for ar in annotated_region[2:]:
    leafArray.append(ar)
    names.append(ano.find(ar, key='id')['acronym'])


leafArray_new=[]

for ar in leafArray:
    leafArray_new.append([[ano.find(ar, key='id')['order'],ano.find(ar, key='id')['level']]])

leafArray=leafArray_new

work_dir='/data_2to/tomek'
controls=['3', '4', '5']


subregion=True
for control in controls:
    length=[]
    L_density=[]
    BP_density=[]
    volumes=[]
    Leafs=[]
    try:
        gts = ggt.load(work_dir + '/' + control+ '/' + str(control)+'_graph_annotated.gt')
    except:
        print('could npot find graph')
    Nb=0

    for leaf in leafArray:
        vertex_filter = np.zeros(gts.n_vertices)
        volume=0
        L=0
        for l in leaf:
            # try:
            new_vol=get_volume_region([l], atlas)
            volume=volume+new_vol#leafArray/2
            print(leaf, volume)
            if new_vol>0:

                label = gts.vertex_annotation();
                # for reg in [leaf]:
                orderl, levell = l
                idl=ano.find(orderl, key='order')['id']
                print(ano.find(orderl, key='order')['name'])
                label_leveled = ano.convert_label(label, key='order', value='order', level=levell)

                print(levell, orderl, ano.find(orderl, key='order')['name'])
                vertex_filter[label_leveled == orderl]=1;

        gss4 = gts.sub_graph(vertex_filter=vertex_filter)
        # coordinates=gss4.edge_geometry_property('coordinates_atlas')
        # indices=gss4.edge_property('edge_geometry_indices')
        coordinates = gss4.edge_geometry_property('coordinates')
        indices = gss4.edge_property('edge_geometry_indices')
        bp=gss4.n_vertices



        for i, ind in enumerate(indices):
            diff=np.diff(coordinates[ind[0]:ind[1]], axis=0)
            L=L+np.sum(np.linalg.norm(diff, axis=1))
        Lmm=L*0.000025
        print(Lmm)#m
        print(Lmm / volume)#m/mm3
        print(bp / volume)#bp/mm3
        volumes.append(volume)
        length.append(Lmm)
        L_density.append(Lmm/volume)
        BP_density.append(bp/volume)
        # if control=='2R':
        Leafs.append(orderl)
        Nb=Nb+1

    np.save(work_dir+'/'+control+'/controls_length.npy', length)
    np.save(work_dir+'/'+control+'/controls_L_density.npy', L_density)
    np.save(work_dir+'/'+control+'/controls_volumes.npy', volumes)
    np.save(work_dir+'/'+control+'/controls_BP_density.npy', BP_density)
    np.save(work_dir+'/'+control+'/controls_leafs.npy', Leafs)



        # except:
        #     print('error')
    # Ls.append(length)
# Ls=np.array(Ls)


# np.save('/data_SSD_1to/stata/length.npy', length)
# np.save('/data_SSD_1to/stata/L_density.npy', L_density)
# np.save('/data_SSD_1to/stata/volumes.npy', volumes)
# np.save('/data_SSD_1to/stata/BP_density.npy', BP_density)
# np.save('/data_SSD_1to/stata/leafs.npy', Leafs)
#
# length=np.array(length).reshape((len(controls),Nb))#len(region_list)
# L_density=np.array(L_density).reshape((len(controls),Nb))
# BP_density=np.array(BP_density).reshape((len(controls),Nb))
# volumes=np.array(volumes).reshape((len(controls),Nb))

# np.save(work_dir+'/old2_length.npy', length)
# np.save(work_dir+'/old2_L_density.npy', L_density)
# np.save(work_dir+'/old2_volumes.npy', volumes)
# np.save(work_dir+'/old2_BP_density.npy', BP_density)
# np.save(work_dir+'/old2_leafs.npy', Leafs)

# np.save(work_dir+'/controls_length.npy', length)
# np.save(work_dir+'/controls_L_density.npy', L_density)
# np.save(work_dir+'/controls_volumes.npy', volumes)
# np.save(work_dir+'/controls_BP_density.npy', BP_density)
# np.save(work_dir+'/controls_leafs.npy', Leafs)

Length_all=[]
for control in controls:
    length=np.load(work_dir+'/'+control+'/controls_L_density.npy')
    Length_all.append(length)

# Nb=int(len(length)/len(controls))
# Length_all=np.array(length).reshape((len(controls),Nb))

length_avg=np.mean(Length_all, axis=0)

correct_values1=np.logical_and(np.mean(length_avg, axis=0)>0 , np.mean(volumes, axis=0)>0.1)#np.where(bp_den<100 and tort<500)
correct_values=np.asarray(correct_values1).nonzero()[0]#np.where(correct_values1==1)
real_L_density=length_avg[:,correct_values]
names_c=names[correct_values]


orders=[]
for ar in annotated_region[2:]:
    orders.append(ano.find(ar, key='id')['order'])
orders=np.array(orders)


fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
sns.set_style('white')
sns.despine()
indices=np.argsort(orders)
names_cs=np.array(names_c)[indices]
ticks=[]
for i, ind in enumerate(indices):
    r=orders[ind]
    print(r,ano.find(r, key='order')['name'],names_c[ind],length_avg[np.asarray(orders==r).nonzero()[0]])
    plt.bar(i, length_avg[np.asarray(orders==r).nonzero()[0]], color=ano.find(r, key='order')['rgb'], alpha=1.0)
    ticks.append(ano.find(r, key='order')['name'])


plt.xticks(np.arange(orders.shape[0]), names_cs, rotation=90, fontsize=5)#ticks
plt.title('axon tomek')

plt.ylim(-2,50)

