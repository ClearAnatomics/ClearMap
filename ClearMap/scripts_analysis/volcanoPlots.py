## import

import numpy as np
import ClearMap.IO.IO as io
#### test volcano plots data stat representation
from scipy import stats
import matplotlib.pyplot as plt
import ClearMap.Alignment.Annotation as ano
work_dir = '/data_SSD_2to/earlyDep'
# work_dir='/data_2to/earlyDep_ipsi'
# work_dir='/data_2to/201120'
condition='whiskers'#'whiskers'#otof
datatype='ori'##ori loops bp
# datatype='extractedFrac'
import tifffile

s=True

control = tifffile.imread(work_dir + '/' +'vox_ori_fi_bv_control_rad10'+str(s)+'.tif')
mutant = tifffile.imread(work_dir + '/' +'vox_ori_fi_bv_mutant_rad10'+str(s)+'.tif')
pvalfile=tifffile.imread(work_dir+'/pvalcolors_radORI_fi_bv_bicol10_0.01'+str(s)+'.tif')
atlas = tifffile.imread('/home/sophie.skriabine/Documents/ClearMap_Ressources/annotation_25_HeadLightOrientation.tif')[:230, :,:]
# atlas = np.flip(atlas, axis=1)
control=np.swapaxes(control, 1,3)
mutant=np.swapaxes(mutant, 1,3)
control=np.swapaxes(control, 0,2)
mutant=np.swapaxes(mutant, 0,2)
pcutoff = 0.05

#atlas region you are ineterested in and for which the name is going to be display
reg_l_int=['AUD', 'VIS', 'bfd', 'IC', 'll', 'CN', 'SOCl', 'SSp-n', 'SSp-m', 'MOs', 'MOp', 'SSp-ul', 'SSp-ll', 'SSs']





## function stats

import numpy


def colorPValues(pvals, psign, positive=[1, 0], negative=[0, 1], pcutoff=None, positivetrend=[0, 0, 1, 0],
                 negativetrend=[0, 0, 0, 1], pmax=None):
    pvalsinv = pvals.copy();
    if pmax is None:
        pmax = pvals.max();
    pvalsinv = pmax - pvalsinv;

    if pcutoff is None:  # color given p values

        d = len(positive);
        ds = pvals.shape + (d,);
        pvc = numpy.zeros(ds);

        # color
        ids = psign > 0;
        pvalsi = pvalsinv[ids];
        for i in range(d):
            pvc[ids, i] = pvalsi * positive[i];

        ids = psign < 0;
        pvalsi = pvalsinv[ids];
        for i in range(d):
            pvc[ids, i] = pvalsi * negative[i];

        return pvc;

    else:  # split pvalues according to cutoff

        d = len(positivetrend);

        if d != len(positive) or d != len(negative) or d != len(negativetrend):
            raise RuntimeError(
                'colorPValues: postive, negative, postivetrend and negativetrend option must be equal length!');

        ds = pvals.shape + (d,);
        pvc = numpy.zeros(ds);

        idc = pvals < pcutoff;
        ids = psign > 0;

        ##color
        # significant postive
        ii = numpy.logical_and(ids, idc);
        pvalsi = pvalsinv[ii];
        w = positive;
        for i in range(d):
            pvc[ii, i] = pvalsi * w[i];

        # non significant postive
        ii = numpy.logical_and(ids, numpy.negative(idc));
        pvalsi = pvalsinv[ii];
        w = positivetrend;
        for i in range(d):
            pvc[ii, i] = pvalsi * w[i];

        # significant negative
        ii = numpy.logical_and(numpy.negative(ids), idc);
        pvalsi = pvalsinv[ii];
        w = negative;
        for i in range(d):
            pvc[ii, i] = pvalsi * w[i];

        # non significant postive
        ii = numpy.logical_and(numpy.negative(ids), numpy.negative(idc))
        pvalsi = pvalsinv[ii];
        w = negativetrend;
        for i in range(d):
            pvc[ii, i] = pvalsi * w[i];

        return pvc;


## cluster computation

tvals, pvals = stats.ttest_ind(control[:,:,:,:], mutant, axis = 3, equal_var = False);


pi = np.isnan(pvals);
pvals[pi] = 1.0;
tvals[pi] = 0;

pvals2 = pvals.copy();
pvals2[pvals2 > pcutoff] = pcutoff;
psign=np.sign(tvals)

pvalfile = colorPValues(pvals2, psign, positive = [255,0,0], negative = [0,255,0])
rad=pvalfile
# Compute DBSCAN
# rad=np.swapaxes(pvalfile, 1, 3)
# rad=np.swapaxes(rad, 1, 0)
neg_rad=np.array(np.asarray(rad[:, :, :, 0]>0).nonzero()).T
pos_rad=np.array(np.array(rad[:, :, :, 1]>0).nonzero()).T

if atlas.shape[:-1]==rad.shape[:-2]:
    print('True')
print(atlas.shape, rad.shape, pvals.shape, tvals.shape)

from sklearn.cluster import DBSCAN
X_tofit=neg_rad
db = DBSCAN(eps=3, min_samples=10).fit(X_tofit)
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)

print('Estimated number of clusters: %d' % n_clusters_)
print('Estimated number of noise points: %d' % n_noise_)



#atlas region you are ineterested in and for which the name is going to be display
# reg_l_int=['AUD',  'IC',]
reg_l_int=['PIR', 'AUD', 'VIS', 'bfd', 'IC', 'll', 'CN', 'SOCl', 'SSp-n', 'SSp-m', 'MOs', 'MOp', 'SSp-ul', 'SSp-ll', 'SSs', 'AI', 'ENT', 'VPM']

fig=plt.figure()


### down regulated cluster computation
ax = fig.add_subplot(111)#, projection='3d')
u, c = np.unique(labels , return_counts=True)
m=np.mean(c)
for i, n in enumerate(u):
    # if c[i]>m:
        indtoplot=np.asarray(labels==n).nonzero()[0]
        pvals_avg=[]
        tvals_avg= []
        regions=[]
        for x_reg in X_tofit[indtoplot]:
            regions.append(atlas[x_reg[0], x_reg[1], x_reg[2]])
            pvals_avg.append(pvals[x_reg[0], x_reg[1], x_reg[2]])
            tvals_avg.append(tvals[x_reg[0], x_reg[1], x_reg[2]])
        regions=np.array(regions)
        u_regions, c_regions=np.unique(regions[regions>0], return_counts=True)
        x_m = np.log10(np.mean(tvals_avg))
        y_m = np.log10(c[i])#-np.log(np.mean(pvals_avg))

        try:
            color = 'grey'
            c_max=np.max(c_regions)
            cs=np.asarray(c_regions>=(c_max-(c_max/2))).nonzero()[0]
            u_regions=u_regions[cs]
            u_region=[ano.find(r, key='id')['acronym'] for r in u_regions]
            s='-'.join(u_region)
            b = False
            for r in reg_l_int:
                # plt.text(x_m, y_m, s, fontsize=12)
                if r in s:
                    # plt.text(x_m, y_m, s, fontsize=12)
                    if -np.log(np.mean(pvals_avg)) > 4:
                        b = True
                        plt.text(x_m, y_m, s, fontsize=12)
                        color = 'indianred'

            if s != 'NoL':#and '6' not in s:
                if c[i] > np.mean(c):
                    color = 'indianred'
                    plt.text(x_m, y_m, s, fontsize=12)
                    for r in reg_l_int:
                        if r in s:
                            plt.text(x_m, y_m, s, fontsize=12)
                if b:
                    plt.text(x_m, y_m, s, fontsize=12)
                    for r in reg_l_int:
                        if r in s:
                            plt.text(x_m, y_m, s, fontsize=12)
                    color = 'indianred'
                if c[i] > np.mean(c) + 1*np.std(c):
                    plt.text(x_m, y_m, s, fontsize=12)
                    for r in reg_l_int:
                        if r in s:
                            plt.text(x_m, y_m, s, fontsize=12)
                    # color = 'indianred'
                if -np.log(np.mean(pvals_avg)) > 4:
                    plt.text(x_m, y_m, s, fontsize=12)
                    for r in reg_l_int:
                        if r in s:
                            plt.text(x_m, y_m, s, fontsize=12)
                    color = 'indianred'
                if abs(x_m) > 1.5:
                    plt.text(x_m, y_m, s, fontsize=12)
                    for r in reg_l_int:
                        if r in s:
                            plt.text(x_m, y_m, s, fontsize=12)
                    color = 'indianred'
            print(u_region, c_regions, x_m, y_m, c[i])
            ax.scatter(x_m, y_m, alpha=0.3, color=color, s=100)  # X_proj_transformed[indtoplot,2] # s=c[i] / 10,
        except:
            print('non registered region')



# reg_l_int=['AUD', 'bfd', 'IC', 'MOs', 'MOp','SSp-n']
X_tofit=pos_rad
db = DBSCAN(eps=3, min_samples=10).fit(X_tofit)
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)

print('Estimated number of clusters: %d' % n_clusters_)
print('Estimated number of noise points: %d' % n_noise_)


### up regulated cluster computation
ax = fig.add_subplot(111)#, projection='3d')
u, c = np.unique(labels , return_counts=True)
m=np.mean(c)
for i, n in enumerate(u):
    # if c[i]>m:
        indtoplot=np.asarray(labels==n).nonzero()[0]
        pvals_avg=[]
        tvals_avg= []
        regions=[]
        for x_reg in X_tofit[indtoplot]:
            regions.append(atlas[x_reg[0], x_reg[1], x_reg[2]])
            pvals_avg.append(pvals[x_reg[0], x_reg[1], x_reg[2]])
            tvals_avg.append(tvals[x_reg[0], x_reg[1], x_reg[2]])
        regions=np.array(regions)
        u_regions, c_regions=np.unique(regions[regions>0], return_counts=True)
        x_m=-np.log10(-np.mean(tvals_avg))
        y_m=np.log10(c[i])#-np.log(np.mean(pvals_avg))

        try:
            color='grey'
            c_max=np.max(c_regions)
            cs=np.asarray(c_regions>=(c_max-(c_max/3))).nonzero()[0]
            u_regions=u_regions[cs]
            u_region=[ano.find(r, key='id')['acronym'] for r in u_regions]
            s='-'.join(u_region)
            b=False
            for r in reg_l_int:
                if r in s:
                    # plt.text(x_m, y_m, s, fontsize=12)
                    if -np.log(np.mean(pvals_avg)) > 4:
                        b=True
                        plt.text(x_m, y_m, s, fontsize=12)
                        color = 'cadetblue'

            if s != 'NoL':# and '6' not in s:
                if c[i] > np.mean(c):
                    plt.text(x_m, y_m, s, fontsize=12)
                    for r in reg_l_int:
                        if r in s:
                            plt.text(x_m, y_m, s, fontsize=12)
                    color = 'cadetblue'
                if b:
                    plt.text(x_m, y_m, s, fontsize=12)
                    color = 'cadetblue'
                if c[i] > np.mean(c)+np.std(c):
                    plt.text(x_m, y_m, s, fontsize=12)
                    for r in reg_l_int:
                        if r in s:
                            plt.text(x_m, y_m, s, fontsize=12)
                    color = 'cadetblue'
                if -np.log(np.mean(pvals_avg))>4:
                    plt.text(x_m, y_m, s, fontsize=12)
                    for r in reg_l_int:
                        if r in s:
                            plt.text(x_m, y_m, s, fontsize=12)
                    color = 'cadetblue'
                if abs(x_m)>1.5:
                    plt.text(x_m, y_m, s, fontsize=12)
                    for r in reg_l_int:
                        if r in s:
                            plt.text(x_m, y_m, s, fontsize=12)
                    color = 'cadetblue'
            print(u_region, c_regions, x_m, y_m, c[i])
            ax.scatter(x_m, y_m, alpha=0.3, color=color, s=100)  # X_proj_transformed[indtoplot,2] # s=c[i] / 10,
        except:
            print('non registered region')


s='orientation'
plt.title(' significative clusters '+s+ ' controls VS '+condition, size='x-large')
plt.tight_layout()
# plt.xscale('log')
# plt.yscale('log')
plt.xticks(size='x-large')
plt.yticks(size='x-large')
plt.ylabel('log cluster size in voxels', size='x-large')
plt.xlabel('-log tvals', size='x-large')


