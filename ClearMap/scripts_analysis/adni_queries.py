import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import ClearMap.Settings as settings
import os
import ClearMap.Alignment.Annotation as ano
import graph_tool.inference as gti
import pandas as pd
import graph_tool.topology as gtt
import ClearMap.Analysis.Graphs.GraphGt as ggt
import seaborn as sns





dataset=pd.read_csv('/data_SSD_2to/adni/hearing_loss_ADNI.csv')
dataset_filtered=dataset[['MMSE_TOT', 'AGE_vis', 'ADAS_TOT13', 'DIAG_VISIT', 'deafness']]
dataset_nonan=dataset_filtered.dropna()

deaf_cohort=dataset_nonan[dataset_nonan['deafness']=='deaf']
nondeaf_cohort=dataset_nonan[dataset_nonan['deafness']!='deaf']

deaf_cohort_AD=deaf_cohort[deaf_cohort['DIAG_VISIT'].isin(['AD'])]
deaf_cohort_CN=deaf_cohort[deaf_cohort['DIAG_VISIT']=='CN']

nondeaf_cohort_AD=nondeaf_cohort[nondeaf_cohort['DIAG_VISIT'].isin(['AD'])]
nondeaf_cohort_CN=nondeaf_cohort[nondeaf_cohort['DIAG_VISIT']=='CN']




plt.figure()
sns.scatterplot(x='AGE_vis', y='MMSE_TOT', data=nondeaf_cohort_AD, color='cadetblue')
sns.scatterplot(x='AGE_vis', y='MMSE_TOT', data=nondeaf_cohort_CN, color='forestgreen')

sns.scatterplot(x='AGE_vis', y='MMSE_TOT', data=deaf_cohort_AD, color='indianred')
sns.scatterplot(x='AGE_vis', y='MMSE_TOT', data=deaf_cohort_CN, color='gold')



plt.figure()
sns.scatterplot(x='AGE_vis', y='ADAS_TOT13', data=nondeaf_cohort_AD, color='cadetblue')
sns.scatterplot(x='AGE_vis', y='ADAS_TOT13', data=nondeaf_cohort_CN, color='forestgreen')

sns.scatterplot(x='AGE_vis', y='ADAS_TOT13', data=deaf_cohort_AD, color='indianred')
sns.scatterplot(x='AGE_vis', y='ADAS_TOT13', data=deaf_cohort_CN, color='gold')



dataset=pd.read_csv('/data_SSD_2to/adni/hearing_loss_ADNI.csv')
dataset_filtered=dataset.copy()#dataset[['MMSE_TOT', 'AGE_vis', 'ADAS_TOT13', 'DIAG_VISIT', 'deafness']]
dataset_nonan=dataset_filtered.dropna()


deaf_cohort=dataset_nonan[dataset_nonan['deafness']=='deaf']
nondeaf_cohort=dataset_nonan[dataset_nonan['deafness']!='deaf']

deaf_cohort_AD=deaf_cohort[deaf_cohort['DIAG_VISIT'].isin(['AD', 'MCI'])]
deaf_cohort_CN=deaf_cohort[deaf_cohort['DIAG_VISIT']=='CN']

nondeaf_cohort_AD=nondeaf_cohort[nondeaf_cohort['DIAG_VISIT'].isin(['AD', 'MCI'])]
nondeaf_cohort_CN=nondeaf_cohort[nondeaf_cohort['DIAG_VISIT']=='CN']



plt.figure()
sns.scatterplot(x='AGE_vis', y='MES_MRI_HIPPOCAMPUS_ICV', data=nondeaf_cohort_AD, color='cadetblue')
sns.scatterplot(x='AGE_vis', y='MES_MRI_HIPPOCAMPUS_ICV', data=nondeaf_cohort_CN, color='forestgreen')

sns.scatterplot(x='AGE_vis', y='MES_MRI_HIPPOCAMPUS_ICV', data=deaf_cohort_AD, color='indianred')
sns.scatterplot(x='AGE_vis', y='MES_MRI_HIPPOCAMPUS_ICV', data=deaf_cohort_CN, color='gold')


from sklearn.manifold import TSNE
import time


diag=dataset_nonan['DIAG_VISIT'].values
diag[diag=='AD']=2
diag[diag=='MCI']=1
diag[diag=='CN']=0

dataset_nonan['diag'] = diag

data=dataset_nonan[['AGE_vis', 'MMSE_TOT', 'ADAS_TOT13', 'MES_MRI_HIPPOCAMPUS_ICV',
                    'MES_MRI_VENTRICLES_ICV']]

time_start = time.time()
tsne = TSNE(n_components=2, verbose=1)
tsne_results = tsne.fit_transform(data)
# np.save('/data_2to/dev/TSNE_good_res_12_200_30.npy', tsne_results)
# np.save('/data_2to/dev/TSNE_good_res_12_1000_80.npy', tsne_results)
print('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))

data['tsne-2d-one'] = tsne_results[:,0]
data['tsne-2d-two'] = tsne_results[:,1]
data['deafness'] = dataset_nonan['deafness']
data['DIAG_VISIT'] = dataset_nonan['DIAG_VISIT']
data['DIAG_MAX'] = dataset_nonan['DIAG_MAX']
data['diag'] = dataset_nonan['diag']


diag=dataset_nonan['DIAG_VISIT'].values
deafness=dataset_nonan['deafness'].values

catgories=np.zeros(deafness.shape)
catgories[np.logical_and(diag==2 , deafness=='deaf')]=0
catgories[np.logical_and(diag==1 , deafness=='deaf')]=1
catgories[np.logical_and(diag==0 , deafness=='deaf')]=2

catgories[np.logical_and(diag==2 , deafness!='deaf')]=6
catgories[np.logical_and(diag==1 , deafness!='deaf')]=5
catgories[np.logical_and(diag==0 , deafness!='deaf')]=4

data['catgories']=catgories

fig=plt.figure()
palette=sns.color_palette("hls",2 )
g_sns=sns.scatterplot(
    x="tsne-2d-one", y="tsne-2d-two",
    hue="deafness",
    palette=palette,#sns.color_palette("hls", 7),
    data=data,
    legend="full",
    alpha=0.3,
    picker = True
)
















