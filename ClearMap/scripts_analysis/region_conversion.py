import numpy as np
import ClearMap.IO.IO as io

ano_old=io.read('/home/sophie.skriabine/Documents/anoP3_v23456789abcdefghi.nrrd')

old_reg=[1,7,8,9,10,12,13,14,15,16,17,18,41,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,16001,16911, 16114,16382,
 16919,16921,16922,17738,17740,32775,11,36,37,38,39,40,32781,32782,42,43,44,45,46,47,48,49,50,51]

new_reg=[1002,322,170,378,795,7,661,672,385,247,698,631,278,429,262,797,6,908,1097,313,275,1065,595,301,1099,
         351,603,449,703,818,315,315, 1080,549,512,4,302,500,329,733,642,398,621,35,115,653,776,81,997,475,178,381,
         214,483,186,64,255,83]

unique=np.unique(ano_old)

ano_new=np.zeros(ano_old.shape)
for u in unique[1:]:
    if u in old_reg:
        new_val=new_reg[np.asarray(old_reg==u).nonzero()[0][0]]
        ano_new[np.where(ano_old==u)]=new_val
    else:
        ano_new[np.where(ano_old==u)]=997#"brain labeled region




io.write( '/home/sophie.skriabine/Documents/anoP3_v23456789abcdefghi_real_ano.nrrd',ano_new)








### 3d smoothing
import pyvista as pv

# anot=io.read('/home/sophie.skriabine/Documents/anoP3_v23456789abcdefghik_real_ano.nrrd')
# anot=io.read('/data_2to/alignement/atlases/new_region_atlases/P14/ano_P14_p10_corrected.tif')
anot=io.read('/data_2to/alignement/atlases/new_region_atlases/P10/ano_P10.nrrd')
anot=anot-32768
smoothed_anot=anot.copy()
# smoothed_anot=np.zeros(anot.shape)


unique=np.unique(anot)
unique=np.delete(unique, np.where(unique==997))
u=329
# u=703

isolated_reg=np.zeros(anot.shape)
where=np.where(anot==997)
smoothed_anot[where[0],where[1],where[2]]=997

for u in unique[1:][::-1]:
    print(u)
    if u!=997:


        isolated_reg=np.zeros(anot.shape)
        where=np.where(anot==u)
        isolated_reg[where[0],where[1],where[2]]=1

        # get contour
        from numpy import sin, cos, pi
        from skimage import measure
        import matplotlib.pyplot as plt
        import open3d as o3d
        from mpl_toolkits.mplot3d import Axes3D


        grid = pv.UniformGrid()
        grid.dimensions = np.array(isolated_reg.shape) + 1
        # grid.origin = (0, 0, 0)  # The bottom left corner of the data set
        grid.spacing = (1, 1, 1)  # These are the cell sizes along each axis
        grid.cell_data["values"] = isolated_reg.flatten(order="F")

        vol = grid.threshold_percent(90, invert=0)
        pv.set_plot_theme('default')
        # vol.plot(show_edges=True, background='white', cmap='coolwarm')


        # Get the out surface as PolyData
        surf = vol.extract_geometry()
        # Smooth the surface
        smooth = surf.smooth(edge_angle=90, feature_angle=90, feature_smoothing=False, n_iter=10000)
        smooth=smooth.clean()
        smooth = smooth.extract_largest()
        # cpos = [-2, 5, 3]
        # smooth.plot(show_edges=True)

        # clipped_smooth = smooth.clip_closed_surface()
        voxels = pv.voxelize(smooth,check_surface=False, density=smooth.length/500)
        voxels=voxels.clip_surface(smooth)
        # p = pv.Plotter()
        # p.add_mesh(voxels, color=True, show_edges=True, opacity=0.5)
        # p.add_mesh(smooth, color="lightblue", opacity=0.5)
        # p.show(cpos=cpos)

        # voxels["density"] = np.full(voxels.n_cells, 3.65) # g/cc
        # voxels.plot(scalars="density", cpos=cpos)
        smoothed_isolated_region=np.array(voxels.points.data.tolist())
        # voxels.plot(cpos=cpos)

        smoothed_anot[smoothed_isolated_region[:,0].astype(int),smoothed_isolated_region[:,1].astype(int),smoothed_isolated_region[:,2].astype(int)]=u
    else:
        isolated_reg=np.zeros(anot.shape)
        where=np.where(anot==u)
        smoothed_anot[where[0],where[1],where[2]]=997


import ClearMap.Visualization.Plot3d as p3d

io.write('/data_2to/alignement/atlases/new_region_atlases/P10/ano_P10_smoothed.tif', smoothed_anot.astype('float32'))

io.write('/data_2to/alignement/atlases/new_region_atlases/P14/ano_P14_p10_corrected_smoothed.tif', smoothed_anot.astype('float32'))
p3d.plot(smoothed_anot)


smoothed_anot=io.read('/home/sophie.skriabine/Documents/smoothed_anot_P3_V4.tif')
where=np.where(anot==997)
smoothed_anot[where[0],where[1],where[2]]=997