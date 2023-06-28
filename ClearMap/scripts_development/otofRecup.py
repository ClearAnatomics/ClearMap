import numpy as np

from ClearMap.Environment import *  # analysis:ignore

from scipy import signal
import matplotlib.pyplot as plt
from sklearn import linear_model

template_shape=(320,528,228)



controls=['17R', '17L', '18R', '18L', '19R', '19L', '20R','20L', '21L', '21R', '22L', '22R']
work_dir='/data_SSD_2to/otofRECUP'
radius=10


#####analyze diff betweeen hemisphers

#vox if needed
control='18L'
graph = ggt.load(work_dir + '/' + control + '/' + 'data_graph_correcteduniverse.gt')
coordinates=graph.vertex_property('coordinates_atlas')
v = vox.voxelize(coordinates[:, :3], shape=template_shape, weights=None, radius=(radius, radius, radius),method='sphere');
io.write(work_dir + '/' + str(18)+'L'+ 'vox_bp_'+str(radius)+'.tif', v.astype('float32'))


####

controls=['17', '18', '19','20', '21', '22']
for i, b1 in enumerate(controls):
        print(b1)
        # try:
        #     x=io.read(work_dir+'/'+b1+'/'+'vox_mutant'+b1+'10.tif')
        # except:
        #     x=io.read(work_dir+'/'+b1+'/'+'vox_control'+b1+'10.tif')
        # try:
        #     y=io.read(work_dir+'/'+b2+'/'+'vox_mutant'+b2+'10.tif')
        # except:
        #     y=io.read(work_dir+'/'+b2+'/'+'vox_control'+b2+'10.tif')
        x=io.read(work_dir + '/' + str(b1)+'R'+ 'vox_bp_'+str(radius)+'.tif')
        y=io.read(work_dir + '/' + str(b1)+'L'+ 'vox_bp_'+str(radius)+'.tif')
        y=np.flip(y,0)
        x=x[:, :, :].flatten().reshape(-1, 1)
        y=y[:, :, :].flatten().reshape(-1, 1)

        clf = linear_model.LinearRegression()
        clf.fit(x, y)
        score=clf.score(x, y)
        print(score)
        y_=clf.predict(x)
        dif=y_-y
        dif=dif.reshape(template_shape)
        dif=np.abs(dif)
        io.write(work_dir+ '/' + str(b1)+ 'bp_linear_dif_'+str(radius)+'.tif',dif)













controls=['17', '18', '19','20', '21', '22']
otof=['17', '20','22']
recup=['18', '19','21']
#correlation



dif_otof=np.zeros((320,528,228,3))
dif_recup=np.zeros((320,528,228,3))

for i, b1 in enumerate(otof):
    x=io.read(work_dir+ '/' + str(b1)+ 'bp_linear_dif_'+str(radius)+'.tif')
    dif_otof[:, :, : ,i]=x

for i, b1 in enumerate(recup):
    x=io.read(work_dir+ '/' + str(b1)+ 'bp_linear_dif_'+str(radius)+'.tif')
    dif_recup[:, :, : ,i]=x

avg_dif_recup=np.mean(dif_recup, 3)
avg_dif_otof=np.mean(dif_otof, 3)


x=avg_dif_otof.flatten().reshape(-1, 1)
y=avg_dif_recup.flatten().reshape(-1, 1)

clf = linear_model.LinearRegression()
clf.fit(x, y)
score=clf.score(x, y)
print(score)
y_=clf.predict(x)
dif=y_-y
dif=dif.reshape(template_shape)

dif=np.abs(dif)
io.write(work_dir+ '/' + 'otofVSrecuplinear_dif_'+str(radius)+'.tif',dif)



import seaborn as sns
plt.figure()
sns.heatmap(corr_matrix, cmap="crest", xticklabels=controls, yticklabels=controls)

N = len(corr_matrix)

from scipy.cluster.hierarchy import dendrogram, linkage
Z = linkage(corr_matrix, 'ward')

res_order = seriation(Z, N, N + N - 2)
seriated_dist = np.zeros((N, N))
a, b = np.triu_indices(N, k=1)
seriated_dist[a, b] = corr_matrix[[res_order[i] for i in a], [res_order[j] for j in b]]
seriated_dist[b, a] = seriated_dist[a, b]


# fig = plt.figure()
plt.subplot(1, 2, 1)
sns.heatmap(seriated_dist, cmap="crest")
sns.despine()
# plt.pcolormesh(seriated_dist)
plt.xlim([0,N])
plt.ylim([0,N])


plt.subplot(1, 2, 2)
Z1 =dendrogram(Z, labels=controls, orientation='right')

sns.despine()


avg_diff=np.mean(diffs, axis=3)
avg_std=np.std(diffs, axis=3)
p3d.plot([-avg_diff, avg_std])

# work_dir='/media/sophie.skriabine/sophie/otof6M'
io.write(work_dir+'/avg_diff_otof_control_allbatch.tif',avg_diff)
io.write(work_dir+'/avg_std_otof_control_allbatch.tif',avg_std)




####### analyze R/L hemispheres

side='R'
controls=['17', '18', '19','20', '21', '22']
otof=['17', '20','22']
recup=['18', '19','21']
template_shape=(320,528,228)
vox_shape=(template_shape[0],template_shape[1],template_shape[2], len(otof))
vox_control_otof=np.zeros(vox_shape)
vox_control_recup=np.zeros(vox_shape)



for i, b1 in enumerate(otof):
    print(b1)
    x=io.read(work_dir + '/' + str(b1)+side+ 'vox_bp_'+str(radius)+'.tif')
    vox_control_otof[:,:,:,i]=x

    b1_=recup[i]
    y=io.read(work_dir + '/' + str(b1_)+side+ 'vox_bp_'+str(radius)+'.tif')
    vox_control_recup[:,:,:,i]=y


x=np.mean(vox_control_otof, axis=3)
y=np.mean(vox_control_recup, axis=3)

# y=np.flip(y,0)
x=x[:, :, :].flatten().reshape(-1, 1)
y=y[:, :, :].flatten().reshape(-1, 1)

clf = linear_model.LinearRegression()
clf.fit(x, y)
score=clf.score(x, y)
print(score)
y_=clf.predict(x)
dif=y_-y
dif=dif.reshape(template_shape)
dif=np.abs(dif)
io.write(work_dir+ '/' + str(b1)+ 'bp_linear_'+side+'_'+str(radius)+'.tif',dif)




