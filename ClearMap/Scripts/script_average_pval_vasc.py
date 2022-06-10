from ClearMap.Environment import *  #analysis:ignorews

#directories and files
directory = '/raid/data/210720_5modep'

import numpy as np


list_file=['467_density_branches.tif', '468_density_branches.tif', '469_density_branches.tif']

for i , file in enumerate(list_file):
	file=io.read(directory+'/'+file)
	if i ==0:
		condensed_control=file[:, :, :, np.newaxis]
	else:
		condensed_control=np.concatenate((condensed_control, file[:, :, :, np.newaxis]), axis=3)

avg_file_control=np.mean(condensed_control, axis=3)
io.write(directory+'/'+'avgcont.tif', avg_file_control)	
io.write(directory+'/'+'condensed_control.tif', condensed_control)	



from ClearMap.Environment import *  #analysis:ignorews

#directories and filesb
directory = '/raid/data/210720_5modep'

import numpy as np




list_file=['433_density_branches.tif', '456_density_branches.tif', '457_density_branches.tif','458_density_branches.tif',]

for i , file in enumerate(list_file):
	file=io.read(directory+'/'+file)
	if i ==0:
		condensed_deprived=file[:, :, :, np.newaxis]
	else:
		condensed_deprived=np.concatenate((condensed_deprived, file[:, :, :, np.newaxis]), axis=3)

avg_file_deprived=np.mean(condensed_deprived, axis=3)
io.write(directory+'/'+'avgdep.tif', avg_file_deprived)	
io.write(directory+'/'+'condensed_deprived.tif', condensed_deprived)	

#%%
from scipy import stats

import numpy
def colorPValues(pvals, psign, positive = [1,0], negative = [0,1], pcutoff = None, positivetrend = [0,0,1,0], negativetrend = [0,0,0,1], pmax = None):
    
    pvalsinv = pvals.copy();
    if pmax is None:
        pmax = pvals.max();    
    pvalsinv = pmax - pvalsinv;    
    
    if pcutoff is None:  # color given p values
        
        d = len(positive);
        ds = pvals.shape + (d,);
        pvc = numpy.zeros(ds);
    
        #color
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
        
        if d != len(positive) or  d != len(negative) or  d != len(negativetrend) :
            raise RuntimeError('colorPValues: postive, negative, postivetrend and negativetrend option must be equal length!');
        
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
    
        #non significant postive
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
    
        #non significant postive
        ii = numpy.logical_and(numpy.negative(ids), numpy.negative(idc))
        pvalsi = pvalsinv[ii];
        w = negativetrend;
        for i in range(d):
            pvc[ii, i] = pvalsi * w[i];
        
        return pvc;
    


vox_control=io.read(directory+'/'+'condensed_control.tif')#io.read(work_dir + '/' +'vox_ori4_control_rad'+str(radius)+'.tif')
vox_mutant=io.read(directory+'/'+'condensed_deprived.tif')#io.read(work_dir + '/' +'vox_ori4_control_rad'+str(radius)+'.tif')
pcutoff = 0.05
tvals, pvals = stats.ttest_ind(vox_control[:, :, :,:], vox_mutant[:, :, :, :], axis = 3, equal_var = False);
pi = np.isnan(pvals);
pvals[pi] = 1.0;
tvals[pi] = 0;
pvals2 = pvals.copy();
pvals2[pvals2 > pcutoff] = pcutoff;
psign=np.sign(tvals)
## from sagital to coronal view
pvals2_f=np.swapaxes(np.swapaxes(pvals2, 0,2), 1,2)
psign_f=np.swapaxes(np.swapaxes(psign, 0,2), 1,2)
# pvals = self.cutoffPValues(pvals, pcutoff = pcutoff);
# pvals, psign = tTestVoxelization(vox_control_avg, vox_mutant_avg, pcutoff = None, signed = True);

from ClearMap.Analysis.Statistics.GroupStatistics import color_p_values
pvalscol = colorPValues(pvals2_f, psign_f, positive = [255,0,0], negative = [0,255,0])

pcutoff = 0.01
pvals2 = pvals.copy();
pvals2[pvals2 > pcutoff] = pcutoff;
psign=np.sign(tvals)
## from sagital to coronal view
pvals2_f=np.swapaxes(np.swapaxes(pvals2, 0,2), 1,2)
psign_f=np.swapaxes(np.swapaxes(psign, 0,2), 1,2)
pvalscol_01 = colorPValues(pvals2_f, psign_f, positive = [0,255,0], negative = [0,0,255])

pvalscol_f=np.maximum(pvalscol, pvalscol_01)

# io.write('/data_SSD_2to/191122Otof/pvalcolors.tif', np.moveaxis(pvalscol, -1, 0).astype('float32'))
# io.write('/data_SSD_2to/191122Otof/pvalcolors.tif', pvalscol.astype('uint8'), photometric='rgb')
import tifffile
tifffile.imsave(directory+'/pvalcolors_bicol_vasc1.tif', np.swapaxes(pvalscol_f, 2, 0).astype('uint8'), photometric='rgb',imagej=True)