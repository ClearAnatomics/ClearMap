# Created by Ricardo Azevedo
# group statistics as used for 2021 iDISCO workshop
# configure directories and files
import tifffile
import numpy as np
from scipy import stats

from ClearMap.IO import IO as io

# root directory of data
directory = '../cfos_group_data'
# voxelized image files for group a
group_a_files = ['2L/density_counts_wcrust.tif',
                 '3L/density_counts_wcrust.tif',
                 '8L/density_counts_wcrust.tif']
# voxelized image files for group b
group_b_files = ['4L/density_counts_wcrust.tif',
                 '5L/density_counts_wcrust.tif',
                 '10L/density_counts_wcrust.tif']
# p valye cutoff
pcutoff = 0.05
# rotate output. Useful if switching between axes orientations. Uses np.transpose notation.
transpose = (1, 2, 0)

# %%
# compute average for group a
images_a = [io.read(f'{directory}/{f}') for f in group_a_files]
condensed_a = np.stack(images_a, axis=3)
avg_file_a = np.mean(condensed_a, axis=3)
io.write(directory + '/a_average.tif', avg_file_a)

# %%
# compute average for group b
images_b = [io.read(f'{directory}/{f}') for f in group_b_files]
condensed_b = np.stack(images_b, axis=3)
avg_file_b = np.mean(condensed_b, axis=3)
io.write(directory + '/b_average.tif', avg_file_b)

# %%
# compute p value map
tvals, pvals = stats.ttest_ind(condensed_a,
                               condensed_b,
                               axis=3,
                               equal_var=False)

# create image
sink = np.zeros(pvals.shape + (3,), dtype=np.uint8)
significant = pvals <= pcutoff
sink[(tvals < 0) & significant] = [0, 255, 0]  # green
sink[(tvals > 0) & significant] = [255, 0, 0]  # red

out = sink.transpose(2, 1, 0, 3)  # reverse transpose from io.read
tifffile.imsave(directory + '/pvalues.tif',
                pvals.transpose(2, 1, 0),
                photometric='rgb')
tifffile.imsave(directory + '/significant.tif',
                out,
                photometric='rgb')
tifffile.imsave(directory + '/significant_transposed.tif',
                out.transpose(transpose + (3,)),  # add rgb axis
                photometric='rgb')
