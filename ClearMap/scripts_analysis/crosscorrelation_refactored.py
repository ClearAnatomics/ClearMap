from sklearn import linear_model
import numpy as np

import ClearMap.IO.IO as io

work_dir=''
vox_controls=''
vox_mutants=''


template_shape=(320,528,228)

x=vox_controls[:, :, :].flatten().reshape(-1, 1)
y=vox_mutants[:, :, :].flatten().reshape(-1, 1)

clf = linear_model.LinearRegression()
clf.fit(x, y)
score=clf.score(x, y)
print(score)
y_=clf.predict(x)
dif=y_-y
dif=dif.reshape(template_shape)


avg_std=np.std(dif)

io.write(work_dir+'/avg_std_otof_control_allbatch.tif',dif)