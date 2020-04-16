from distutils.core import setup
from distutils.extension import Extension
import numpy as np
package_name = 'distance_transform'
module_name1 = 'geodesic_distance'
module1 = Extension(module_name1,
                    include_dirs = [np.get_include(),'./cpp'],
                    sources = ['./cpp/util.cpp', './cpp/geodesic_distance_2d.cpp', './cpp/geodesic_distance_3d.cpp', './cpp/geodesic_distance.cpp'])

setup(name=package_name,
      ext_modules = [module1])


# to build, run python stup.py build or python setup.py build_ext --inplace
# to install, run python setup.py install
