list_brains=[['1', '12','13'], ['31','32','33','34'], ['51', '5b'], [['71', '72']]]
brains=[1, 3, 5, 7]
work_dir='/data_2to/alignement/'
for i in range(len(list_brains)):
    print(brains[i])
    for j in range(len(list_brains[i])):
        if j==0:
            img=io.read(work_dir+'result'+list_brains[i][j]+'.tif')[:, :, :, np.newaxis]
        else:
            img=np.concatenate((img, io.read(work_dir+'result'+list_brains[i][j]+'.tif')[:, :, :, np.newaxis]), axis=3)

    avg_img=np.mean(img, axis=3)
    io.write(work_dir+'avg'+str(brains[i])+'.tiff', avg_img.astype('float32'))

ali_1=io.read('/data_2to/alignement/resulte18.tif').swapaxes(0,2)
io.write('/data_2to/alignement/resulte18_swap.tif', ali_1.astype('float32'))
/data_2to/alignement/resulte18.tif




#create colors atlas map
import matplotlib.pyplot as plt
import numpy as np
import ClearMap.IO.IO as io
import ClearMap.Settings as settings
import os
import ClearMap.Alignment.Annotation as ano
import graph_tool.inference as gti
import pandas as pd
import graph_tool.topology as gtt
import ClearMap.Analysis.Graphs.GraphGt as ggt
import tifffile

atlas_path = os.path.join(settings.resources_path, 'Atlas');

atlasref=io.read('/data_2to/pix/annotation_halfbrain_with_audbarmot.tif')
atlas=io.read('/data_2to/alignement/atlases/atlasP1.tif')
# ano.initialize(label_file = '/data_2to/pix/region_ids_test_ADMBA_audbarmo.json',
#                extra_label = None, annotation_file = '/data_2to/alignement/atlases/atlasP1.tif')


ano.initialize(label_file = '/home/sophie.skriabine/Projects/ClearMap3/ClearMap/ClearMap/Resources/Atlas/region_ids_test_ADMBA.json',
               extra_label = None, annotation_file = '/home/sophie.skriabine/Documents/ano_full.tif')

atlas[np.where(atlas>51000)]=32768
atlas[np.where(atlas<48470)]=32768
atlas_colored=np.zeros((atlas.shape[0],atlas.shape[1],atlas.shape[2],3))

color_id_ref=np.unique(atlasref)[1:]
color_id=np.unique(atlas)[1:]-32768
for c in color_id:
    print(c)
    try:
        rgb=ano.find(c)['color_hex_triplet'].lstrip('#')
        rgb=tuple(int(rgb[k:k+2], 16) for k in (0, 2, 4))
        print(rgb)
        if rgb==(176, 255, 184):
            for i in range(5):
                if c+i in color_id_ref:
                    rgb=ano.find(c+i)['color_hex_triplet'].lstrip('#')
                    rgb=tuple(int(rgb[k:k+2], 16) for k in (0, 2, 4))
                    if rgb!=(176, 255, 184):
                        print(rgb)
                        break
                elif c-i in color_id_ref:
                    rgb=ano.find(c-i)['color_hex_triplet'].lstrip('#')
                    rgb=tuple(int(rgb[k:k+2], 16) for k in (0, 2, 4))
                    if rgb!=(176, 255, 184):
                        print(rgb)
                        break
        print(rgb)


    except:
        print('unreferenced pix val')
        print('try neighbours values')
        for i in range(10):
            rgb=(0,0,0)
            if c+i in color_id_ref:
                rgb=ano.find(c+i)['color_hex_triplet'].lstrip('#')
                rgb=tuple(int(rgb[k:k+2], 16) for k in (0, 2, 4))
                print(rgb)
                break
            elif c-i in color_id_ref:
                rgb=ano.find(c-i)['color_hex_triplet'].lstrip('#')
                rgb=tuple(int(rgb[k:k+2], 16) for k in (0, 2, 4))
                print(rgb)
                break

    print(rgb)
    where=np.where(atlas==c+32768)
    atlas_colored[where[0],where[1],where[2], 0]=rgb[0]
    atlas_colored[where[0],where[1],where[2], 1]=rgb[1]
    atlas_colored[where[0],where[1],where[2], 2]=rgb[2]

# tifffile.imsave('/data_2to/pix/annotation_halfbrain_with_audbarmot_colored_swaped.tif', atlas_colored.swapaxes(0,1).astype('uint8'), photometric='rgb',imagej=True)

tifffile.imsave('/data_2to/alignement/atlases/atlas_full_P3_colored.tif', atlas_colored.swapaxes(0,1).astype('uint8'), photometric='rgb',imagej=True)


## atlas
atlasref = io.read('/data_2to/alignement/atlases/new_region_atlases/P21/atlasP21_full.nrrd')
# atlasref=atlasref-32768
atlas_colored=np.zeros((atlasref.shape[0],atlasref.shape[1],atlasref.shape[2],3))
color_id_ref=np.unique(atlasref)[1:]
for c in color_id_ref:
    print(c)
    try:
        rgb=ano.find(c)['color_hex_triplet'].lstrip('#')
        rgb=tuple(int(rgb[k:k+2], 16) for k in (0, 2, 4))
        print(rgb)
        print(rgb)
        where=np.where(atlasref==c)
        atlas_colored[where[0],where[1],where[2], 0]=rgb[0]
        atlas_colored[where[0],where[1],where[2], 1]=rgb[1]
        atlas_colored[where[0],where[1],where[2], 2]=rgb[2]
    except:
        print('unreferenced pix val')



# tifffile.imsave('/data_2to/pix/annotation_halfbrain_with_audbarmot_colored_swaped.tif', atlas_colored.swapaxes(0,1).astype('uint8'), photometric='rgb',imagej=True)
import tifffile
tifffile.imsave('/data_2to/alignement/atlases/new_region_atlases/P21/atlasP21_full_colored.tif', atlas_colored.swapaxes(0,1).astype('uint8'), photometric='rgb',imagej=True)


