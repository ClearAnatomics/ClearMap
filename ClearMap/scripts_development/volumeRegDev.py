import numpy as np

from ClearMap.Environment import *  # analysis:ignore



import pandas as pd
data=pd.DataFrame()
tps=[9, 12, 14, 21]
for tp in tps:
    atlas=io.read('/data_SSD_2to/ATLASES/FINAL/P'+str(tp)+'/p'+str(tp)+'_new annotation_masked_NEW.nrrd')

    aligned_nnotation = elx.transform(
        atlas,
        sink=[],
        transform_directory='/data_SSD_2to/ATLASES/NICOLAS/P5/'+ 'elastix_auto_03_to_05',
        result_directory= '/data_SSD_2to/ATLASES/NICOLAS/P5/'+ 'res_dir'
    )

    regions_ids=np.unique(atlas)
    for i, reg in enumerate(regions_ids):
        vol=np.sum(np.asarray(atlas==reg).nonzero()[0])
        vol=vol*25*25*25*1e-9 #(volume in mm^3)
        data=data.append({'region': reg, 'volume': vol},ignore_index=True)

    data.to_csv('/data_SSD_2to/ATLASES/FINAL/P'+str(tp)+'/p'+str(tp)+'_volume_region.csv', index=False)



data2plot=pd.DataFrame()
tps=[9, 12, 14, 21]
for i, reg in enumerate(regions_ids):
    vs=[]
    for tp in tps:
        dat=pd.read_csv('/data_SSD_2to/ATLASES/FINAL/P'+str(tp)+'/p'+str(tp)+'_volume_region.csv')
        # data2plot=data2plot.append({'region': reg,
        #                             'volume': dat[dat['region']==reg]['volume'].values[0],
        #                             'timepoint' : tp},ignore_index=True)

        vs.append(dat[dat['region']==reg]['volume'].values[0])
    print(vs)
    data2plot=data2plot.append({'region': reg,
                                'color': ano.find(reg)['rgb'],
                                'volume p9': vs[0],
                                'volume p12': vs[1],
                                'volume p14': vs[2],
                                'volume p21': vs[3]},ignore_index=True)

import seaborn as sns
import matplotlib.pyplot as plt
# sns.lineplot(x='timepoint', y='volume', hue='region', data=data2plot, palette=sns.color_palette("Spectral", as_cmap=True))


## filter by metaregions
metaregions=['Isocortex', 'HIP', 'HY', 'CB', 'HB', 'MB', 'TH']

level=ano.find('HB', key='acronym')['level']
id=ano.find('HB', key='acronym')['id']
regions_ids_leveled=ano.convert_label(regions_ids, key='id', value='id', level=level)
id_filtered=regions_ids[regions_ids_leveled == id]

plt.figure()
for i, reg in enumerate(id_filtered):
    plt.plot(data2plot[tps, data2plot['region']==reg][['volume p9', 'volume p12', 'volume p14', 'volume p21']].values[0], color=data2plot[data2plot['region']==reg]['color'].values[0])






























from ClearMap.Environment import *  # analysis:ignore
resources_directory = settings.resources_path
align_channels_affine_file = io.join(resources_directory, 'Alignment/align_affine.txt')
align_reference_affine_file = io.join(resources_directory, 'Alignment/align_affine.txt')
align_reference_bspline_file = io.join(resources_directory, 'Alignment/align_bspline.txt')





new_annotation=io.read('/data_SSD_2to/ATLASES/FINAL/P21/p21_new_annotation_masked_NEW.tif')

template='/data_SSD_2to/ATLASES/FINAL/P21/base/p21_new_template_coronal_symetric.tif'
target_template='/data_SSD_2to/ATLASES/FINAL/adult/LightSheetTemplate_full_coronal_25micron.tif'


align_channels_parameter = {
    # moving and reference images
    "moving_image": template,
    "fixed_image": target_template,

    # elastix parameter files for alignment
    "affine_parameter_file": align_reference_affine_file,
    "bspline_parameter_file": align_reference_bspline_file,

    # directory of the alig'/home/nicolas.renier/Documents/ClearMap_Ressources/Par0000affine.txt',nment result
    "result_directory": '/data_SSD_2to/ATLASES/FINAL/P21/21_to_adult'
}
elx.align(**align_channels_parameter);


# new_annotation='/data_SSD_2to/ATLASES/NICOLAS/P12/Combined_Stacks_annotation_toalignonp9.tif'
aligned_annotation = elx.transform(
    '/data_SSD_2to/ATLASES/FINAL/P21/p21_new_annotation_masked_NEW.tif',
    sink=[],
    transform_directory='/data_SSD_2to/ATLASES/FINAL/P21/21_to_adult/',
    result_directory= '/data_SSD_2to/ATLASES/FINAL/P21/'+ 'res_dir/'
)




## compute volume and overlap
p21=io.read('/data_SSD_2to/ATLASES/FINAL/P21/res_dir/result.tif')-32768
adult=io.read('/data_SSD_2to/ATLASES/FINAL/adult/Combined Stacks.tif')
regions_ids=np.unique(p21)

adult=np.flip(adult, 2)


plt.figure()
sns.despine()
metaregions=['Isocortex', 'HIP', 'HY', 'CB', 'HB', 'MB', 'TH']
# for mr in metaregions:
#     level=ano.find(mr, key='acronym')['level']
#     id=ano.find(mr, key='acronym')['id']
#     col=ano.find(mr, key='acronym')['rgb']
regions_ids=[329]
for mr in regions_ids:
    level=ano.find(mr, key='id')['level']
    id=ano.find(mr, key='id')['id']
    col=ano.find(mr, key='id')['rgb']
    regions_ids_leveled_21=ano.convert_label(p21, key='id', value='id', level=level)
    id_filtered_21=regions_ids_leveled_21[regions_ids_leveled_21 == id]
    vol_p21=np.sum(regions_ids_leveled_21 == id)

    regions_ids_leveled_adult=ano.convert_label(adult, key='id', value='id', level=level)
    regions_ids_leveled_adult[regions_ids_leveled_adult==353]=329
    id_filtered_adult=regions_ids_leveled_adult[regions_ids_leveled_adult == id]
    vol_adult=np.sum(regions_ids_leveled_adult == id)

    vol_dif=(vol_adult-vol_p21)/vol_adult
    overlap=np.sum(np.logical_and(regions_ids_leveled_adult == id, regions_ids_leveled_21 == id))/vol_adult


    plt.scatter(vol_dif, overlap, color=col)
    plt.text(vol_dif, overlap, ano.find(mr, key='id')['acronym'], color=col)



plt.ylim(0,1)
plt.xlim(-1.2,1.2)
plt.xlabel('volume difference')
plt.ylabel('overlap rate')
import seaborn as sns





plt.figure(10)
plt.figure(11)
sns.despine()
metaregions=['Isocortex', 'HIP', 'HY', 'CB', 'HB', 'MB', 'TH']


vol_brain_p21=np.sum(p21 != 0)
vol_brain_adult=np.sum(adult != 0)
for i, mr in enumerate(metaregions):
    level=ano.find(mr, key='acronym')['level']
    id=ano.find(mr, key='acronym')['id']
    col=ano.find(mr, key='acronym')['rgb']

    regions_ids_leveled_21=ano.convert_label(p21, key='id', value='id', level=level)
    id_filtered_21=regions_ids_leveled_21[regions_ids_leveled_21 == id]
    vol_p21=np.sum(regions_ids_leveled_21 == id)

    regions_ids_leveled_adult=ano.convert_label(adult, key='id', value='id', level=level)
    id_filtered_adult=regions_ids_leveled_adult[regions_ids_leveled_adult == id]
    vol_adult=np.sum(regions_ids_leveled_adult == id)

    plt.figure(10)
    plt.bar(i,vol_adult/vol_brain_adult, 0.5,  color=col)

    plt.figure(11)
    plt.bar(i,vol_p21/vol_brain_p21, 0.5,  color=col)

plt.ylim(0,0.3)
