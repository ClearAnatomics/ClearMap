# -*- coding: utf-8 -*-
"""
Converts orientation information into image data for visulalization and analysis
"""
__author__    = 'Christoph Kirst <christoph.kirst.ck@gmail.com>'
__license__   = 'GPLv3 - GNU General Pulic License v3 (see LICENSE.txt)'
__copyright__ = 'Copyright © 2020 by Christoph Kirst'
__webpage__   = 'http://idisco.info'
__download__  = 'http://www.github.com/ChristophKirst/ClearMap2'

import numpy
import math

import pyximport;
pyximport.install(setup_args={"include_dirs":numpy.get_include()}, reload_support=True)

import ClearMap.IO.IO as io
import ClearMap.Analysis.OrientationCode as orc

def voxelizeOrientations(points, orientations, dataSize = None, sink = None, size = (5,5,5), weights = None):
    """Converts a list of points into an volumetric image array
    
    Arguments:
        points (array): point data array
        orientations (array): point data array
        dataSize (tuple): size of final image
        sink (str, array or None): the location to write or return the resulting voxelization image, if None return array    
    Returns:
        (array): volumetric data of orientation statistics
    """
    
    if dataSize is None:
      dataSize = tuple(int(math.ceil(points[:,i].max())) for i in range(points.shape[1]));
    elif isinstance(dataSize, str):
      dataSize = io.dataSize(dataSize);
        
    if weights is not None:
      orts = (orientations.T * weights).T;
    else:
      orts = orientations;
    
    data = orc.voxelizeOrientations(points, orts, dataSize[0], dataSize[1], dataSize[2], size[0], size[1], size[2]);
    
    return data;
    #return io.writeData(sink, data);



def test():
    """Test voxelization module"""
    import numpy as np
    
    import ClearMap.Analysis.Orientation as ort
    
    from importlib import reload
    reload(ort)
    
    
    n = 200;
    
    #points = np.random.rand(n,3) * 20;
    #orientations = np.random.rand(n,3);
    #orientations = (orientations.T / np.linalg.norm(orientations, axis = 1)).T;
    
    phi = np.linspace(0, 2 * np.pi, n);
    psi = np.linspace(0, 2 * np.pi, n);
    phi,psi = np.meshgrid(phi,psi);
    
    orientations = np.array([np.cos(psi) * np.cos(phi), np.sin(phi), np.sin(psi) * np.cos(phi)]);
    
    points = np.meshgrid(np.arange(n), np.arange(n));
    points = np.array(points);
    
    ox, oy, oz = orientations;
    px, py = points;
    
    orientations_f = np.array([ox.flatten(), oy.flatten(), oz.flatten()]).T
    points_f = np.array([px.flatten(), py.flatten()]).T
    
    
    import mayavi.mlab as mlab
    mlab.quiver3d(points_f[:,0], points_f[:,1], np.zeros(points_f.shape[0]), orientations_f[:,0], orientations_f[:,1], orientations_f[:,2], color = 'black', opacity = 0.2, scale_factor = 1.5)
    

    vo = ort.voxelizeOrientations(points_f, orientations_f, dataSize = (20,20,20), size = (5,5,5));
  
    #normalize
    nrm = np.linalg.norm(vo, axis  = -1);
    nrm[nrm < 10e-5] = 1.0;  
    vo = (vo.transpose((3,0,1,2)) / nrm).transpose((1,2,3,0))
    
    import ClearMap.Visualization.ColorMaps as cmaps;
    reload(cmaps)
    
    vo2 = cmaps.boys2rgb(vo.transpose([3,0,1,2]));
    
    import ClearMap.Visualization.Plot3d as p3d
    p3d.plot([[vo2[:,:,:,0], vo2[:,:,:,1], vo2[:,:,:,2]]])
    
   
    import ClearMap.Visualization.ColorMaps as cmaps;
    reload(cmaps)
    
    ob = cmaps.boys2rgb(orientations);
    
    
    ob.shape = ob.shape + (1,)
    
    
    p3d.plot([[ob[0], ob[1], ob[2]]])
    
    
    
    from mayavi import mlab
    import numpy as np
    
    import ClearMap.Visualization.ColorMaps as cmaps;
    reload(cmaps)
    
    # Make sphere, choose colors
    phi, theta = np.mgrid[0:np.pi:101j, 0:2*np.pi:101j]
    x, y, z = np.sin(phi) * np.cos(theta), np.sin(phi) * np.sin(theta), np.cos(phi)
    xr, yr, zr = x.ravel(), y.ravel(), z.ravel();
    
    
    rgb = cmaps.boys2rgb([xr,yr,zr]);
    rgb = np.abs(np.array([xr,yr,zr])).T;
    
    color = np.zeros((rgb.shape[0], 4), dtype = 'uint8');
    color[:,:3]= rgb * 255;
    color[:,3] = 255;
    
    #s = cmaps.rgbToLUT(rgb);
    #si = np.array(s, dtype = int);
    
    si = np.arange(len(xr));
    si.shape = x.shape;

    
    #lut = cmaps.rgbLUT();
    lut = 0.8 * color;
    lut[:,3] = 255;
    
    # Display
    mlab.figure(1, bgcolor=(1, 1, 1), fgcolor=(0, 0, 0), size=(600, 500))
    mm = mlab.mesh(x, y, z, scalars = si, colormap='Spectral', vmin = 0, vmax = lut.shape[0])
    
    #mm = mlab.points3d(xr, yr, zr, si, mode = 'point')
    
    #magic to modify lookup table 
    mm.module_manager.scalar_lut_manager.lut._vtk_obj.SetTableRange(0, lut.shape[0])
    mm.module_manager.scalar_lut_manager.lut.number_of_colors = lut.shape[0]
    mm.module_manager.scalar_lut_manager.lut.table = lut
    
    mlab.view()
    mlab.show()
    
    
    
    
    

    
    
    
    ### colors on sphere
#    import numpy as np
#    from mpl_toolkits.mplot3d import Axes3D
#    import matplotlib.pyplot as plt
#    import matplotlib.tri as mtri
#    
#    (n, m) = (250, 250)
#    
#    # Meshing a unit sphere according to n, m 
#    theta = np.linspace(0, 2 * np.pi, num=n, endpoint=False)
#    phi = np.linspace(np.pi * (-0.5 + 1./(m+1)), np.pi*0.5, num=m, endpoint=False)
#    theta, phi = np.meshgrid(theta, phi)
#    theta, phi = theta.ravel(), phi.ravel()
#    theta = np.append(theta, [0.]) # Adding the north pole...
#    phi = np.append(phi, [np.pi*0.5])
#    mesh_x, mesh_y = ((np.pi*0.5 - phi)*np.cos(theta), (np.pi*0.5 - phi)*np.sin(theta))
#    triangles = mtri.Triangulation(mesh_x, mesh_y).triangles
#    x, y, z = np.cos(phi)*np.cos(theta), np.cos(phi)*np.sin(theta), np.sin(phi)
#    
#    # Defining a custom color scalar field
#    vals = np.sin(6*phi) * np.sin(3*theta)
#    colors = np.mean(vals[triangles], axis=1)
#    
#    # Plotting
#    fig = plt.figure()
#    ax = fig.gca(projection='3d')
#    cmap = plt.get_cmap('Blues')
#    triang = mtri.Triangulation(x, y, triangles)
#    collec = ax.plot_trisurf(triang, z, cmap=cmap, shade=False, linewidth=0.)
#    collec.set_array(colors)
#    collec.autoscale()
#    plt.show()
    
    
    
if __name__ == "__main__":
    test();