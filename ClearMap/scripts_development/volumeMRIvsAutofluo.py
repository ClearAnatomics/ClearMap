import pandas as pd
import os

import numpy as np
import pyqtgraph as pg

from ClearMap.IO.elastix_config import ElastixParser
from ClearMap.gui.widgets import LandmarksSelectorDialog, Scatter3D
from ClearMap.Environment import *  # analysis:ignore

resources_directory = settings.resources_path

align_channels_affine_file = io.join(resources_directory, 'Alignment/align_affine.txt')
align_reference_affine_file = io.join(resources_directory, 'Alignment/align_affine.txt')
align_reference_bspline_file = io.join(resources_directory, 'Alignment/align_bspline.txt')

directory='/data_SSD_2to/ATLASES/autofluo25/P12'
reference='/data_SSD_2to/ATLASES/FINAL/P12/sym_ref_sagital.tif'
annotation_file='/data_SSD_2to/ATLASES/FINAL/P12/sym_ano_sagital.tif'

# directory='/data_SSD_2to/ATLASES/autofluo25/P3'
# subdirs=['1', '2', '3', '4', '5', '6', '7', '8', '9']
directory='/data_SSD_2to/ATLASES/autofluo25/P12'
subdirs=['093731','102359', '173158']
# subdirs=['1', '2', '3', '4','5', '6', '7', '8', '9', '10']
# subdirs=['6', '7', '8', '9', '14', '15']#['1', '2', '3', '4', '12', '13', '14', '15'
for subdir in subdirs:
    print(io.join(directory,subdir))
    print(io.join(directory,subdir))
    align_channels_parameter = {
        # moving and reference images
        "moving_image": reference,
        "fixed_image": io.join(directory,subdir)+'/'+'resampled_autofluorescence.tif',

        # elastix parameter files for alignment
        "affine_parameter_file": align_reference_affine_file,
        "bspline_parameter_file": align_reference_bspline_file,

        # directory of the alig'/home/nicolas.renier/Documents/ClearMap_Ressources/Par0000affine.txt',nment result
        "result_directory": directory + '/'+subdir +'/'+ 'elastix_auto_to_chosen_auto_full'
    }
    elx.align(**align_channels_parameter);

    aligned_nnotation = elx.transform(
        annotation_file,
        sink=[],
        transform_directory=directory + '/'+subdir   +'/'+ 'elastix_auto_to_chosen_auto_full',
        result_directory= directory + '/'+subdir  +'/'+ 'ano_to_auto'
    )


reg_ids=[315, 549, 1097, 512, 477,313, 1065]
reg_name=['Isocortex', 'Thalamus', 'Hypothalamus', 'Cerebellum', 'Striatum', 'Midbrain', 'Hindbrain']
regions=[]
for id in reg_ids:
    regions.append((ano.find(id)['id'],ano.find(id)['level']))



import vtk
from vtk.util.numpy_support import vtk_to_numpy
import os
import tifffile
D1=pd.DataFrame()

directory='/data_SSD_2to/ATLASES/autofluo25/P12'
# subdirs=['1', '2', '3', '4','5','6', '7', '8', '9', '10']
subdirs=['093731','102359', '173158']

status='idisco'
for subdir in subdirs:
    print(io.join(directory,subdir))
    try:
        f_name=directory + '/'+subdir  +'/'+ 'ano_to_auto'+'/result.tif'
        arr=io.read(f_name)
    except:
        print('no tif file, reading .mhd')
        f_name=directory + '/'+subdir  +'/'+ 'ano_to_auto'+'/result.mhd'
        imr = vtk.vtkMetaImageReader()
        imr.SetFileName(f_name)
        imr.Update()
        im = imr.GetOutput()
        rows, cols, z = im.GetDimensions()
        sc = im.GetPointData().GetScalars()
        arr = vtk_to_numpy(sc)
        arr = arr.reshape(z, cols, rows, 1)
    arr = arr.swapaxes(0, 2)
    for id, level in regions:
        # try:
        arr_leveled = ano.convert_label(arr, key='id', value='id', level=level)
        # except:
        #     print('nope')
        volume_pix=np.sum(arr_leveled==id)
        print(volume_pix, ano.find(id)['name'])
        D1=D1.append({'timepoint': subdir, 'status': status, 'volume_pix':volume_pix, 'region':ano.find(id)['name']},ignore_index=True)

status='mri'
subdirs=['1', '2', '3']
directory='/data_SSD_2to/IRM/P12_T2'
for subdir in subdirs:
    print(io.join(directory,subdir))
    f_name=directory + '/'+subdir  +'/'+ 'ano_to_MRI'+'/result.tif'
    arr=io.read(f_name)
    for id,level in regions:
        try:
            arr_leveled = ano.convert_label(arr, key='id', value='id', level=level)
        except:
            arr_leveled = ano.convert_label(arr-32768, key='id', value='id', level=level)
        volume_pix=np.sum(arr_leveled==id)
        print(volume_pix, ano.find(id)['name'])
        D1=D1.append({'timepoint': subdir, 'status': status, 'volume_pix':volume_pix, 'region':ano.find(id)['name']},ignore_index=True)


D1.to_csv(directory+'/P12_volumeRegionMRIvsDISCO.csv', index=False)



D1=pd.read_csv(directory+'/P12_volumeRegionMRIvsDISCO.csv')

D1=pd.read_csv('/data_SSD_2to/IRM/P3_volumeRegionMRIvsDISCO.csv')
D1=D1.replace('Striatum', 'Caudoputamen')
import matplotlib.pyplot as plt
from math import pi
import seaborn as sns
df=D1
status=['idisco', 'mri']
x='volume_pix'#'n_vertices/length'#
y='region'
val_min=0
val_max=500000
step=50000
pal = ['indianred', 'forestgreen']

categories=list(df)[1:]
N = len(categories)

categories=np.unique(list(df['region']))
idx = [0, 3, 1, 2, 4, 5, 6]
# idx=[0, 3, 2, 4, 1, 5, 6]
categories=categories[idx]#put isocortex and brainstem close together

N = len(categories)
# What will be the angle of each axis in the plot? (we divide the plot / number of variable)
angles = [n / float(N) * 2 * pi for n in range(N)]
angles += angles[:1]

plt.figure()

# Initialise the spider plot
ax = plt.subplot(111, polar=True)

# If you want the first axis to be on top:
ax.set_theta_offset(pi / 2)
ax.set_theta_direction(-1)

# Draw one axe per variable + add labels
plt.xticks(angles[:-1], categories)

# Draw ylabels
ax.set_rlabel_position(0)
# plt.yticks([10,20,30], ["10","20","30"], color="grey", size=7)
# plt.ylim(0,40)

# plt.yticks(list(np.arange(val_min, val_max+step, step)), np.arange(val_min, val_max+step, step).astype('str'), color="grey", size=7)
# plt.ylim(val_min,val_max)

df=df[df['region']!='Brain stem']
for i, st in enumerate(status):
    values=df[df['status']==st]
    # x = df.get(x, x)
    # y = df.get(y, y)
    for brain in values['timepoint'].unique():
        values=df[df['status']==st]
        values=values[values['timepoint']==brain]['volume_pix'].values.tolist()
        # idx_ = [3, 0, 6, 1, 5, 4, 2]#p12
        idx_ = [4, 2, 3, 6, 0, 5, 1]#px
        # idx_=idx_.tolist()
        values=np.array(values)[idx_].tolist()
        values.append(values[0])
        ax.plot(angles, values, linewidth=1, linestyle='--', label=str(st), color=pal[i], alpha=0.5)

    values=df[df['status']==st]
    values=values.groupby(y)[x].mean().values.tolist()
    values=np.array(values)[idx].tolist()
    values.append(values[0])
    ax.plot(angles, values, linewidth=1, linestyle='solid', label=str(st), color=pal[i])
    # ax.fill(angles, values, 'b', alpha=0.1)

# Add legend
plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))

# Show the graph
plt.show()


