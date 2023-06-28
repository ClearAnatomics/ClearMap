import os
import numpy as np
import pandas
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from numpy import sqrt, pi, exp, linspace
from scipy.signal import chirp, find_peaks, peak_widths


n=0
workdir='/mnt/vol00-renier/Nicolas/capillary_profile'
savedir='/mnt/data_SSD_2to/capillaries_profile'

def gaussian(x, amp, cen, wid, offset):
 return offset+(amp * exp(-(x-cen)**2 /wid))

for path, subdirs, files in os.walk(workdir):
    for file in files:
        if '12' in file:
            n=n+1
            print(n, file)
            df = pandas.read_csv(os.path.join(workdir, file))
            fig=plt.figure(n)
            x=df['X']
            y=df['Y']
            plt.plot(x,y)
            best_vals, covar = curve_fit(gaussian, x, y, maxfev=1000, method='lm',  p0=(8000, 30, 10, 200))
            amp, cen, wid, offset = best_vals
            print(best_vals)
            y_fit=gaussian(x, *best_vals)
            peaks, _ = find_peaks(y)
            results_half = peak_widths(y, peaks, rel_height=0.5)
            print(results_half[0])  # widths
            plt.plot(x, y_fit)
            plt.title(file[:-4])
            plt.plot(peaks, x[peaks], "x")
            plt.hlines(*results_half[1:], color="C2")
            plt.show()
            params=[amp, cen, wid, offset]
            # for r in results_half:
            rmax=np.max(np.array(results_half[0]))
            print(rmax)
            # if len(results_half)>0:
            params.append(rmax)
            np.save(os.path.join(savedir, file[:-4]+'_params_'+'.npy'),  np.array(params))


