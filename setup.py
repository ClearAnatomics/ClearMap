import functools
import os
import platform
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor

from multiprocessing import cpu_count
from pathlib import Path

from setuptools import setup, find_packages, Extension

import numpy as np

from Cython.Build import cythonize

# from https://stackoverflow.com/a/3042436
# To modify the data install dir to match the source install dir...
# Pull the installation dir info from distutils with:
from distutils.command.install import INSTALL_SCHEMES
# Modify the data install dir to match the source install dir:
for scheme in INSTALL_SCHEMES.values():
    scheme['data'] = scheme['purelib']

requirements = [  # pip
    'lxml'
]

OPTIMISE_COMPILATION_FOR_TARGET = True  # Cython code will run on the machine where is was compiled
N_PROCS = cpu_count() - 2
DEFAULT_COMPILE_ARGS = ['-w', '-O3']
DEFAULT_LINK_ARGS = []

if len(sys.argv) > 2:
    USE_OPENMP = sys.argv[2].lower() in ('use_openmp', 'true')
else:
    os_name = platform.system().lower()
    if os_name.startswith('linux'):
        USE_OPENMP = True
    elif os_name.startswith('darwin'):
        cpp_compiler_version = subprocess.check_output(['c++', '--version']).decode('ascii').split('\n')[0]
        USE_OPENMP = 'g++' in cpp_compiler_version
    else:
        raise ValueError(f'Unknown OS {os_name}')

if OPTIMISE_COMPILATION_FOR_TARGET:
    DEFAULT_COMPILE_ARGS += ['-march=native', '-mtune=native']


def module_path_to_doted(ext_path):
    return os.path.splitext(ext_path)[0].replace(os.sep, '.')


def find_data_files(src_dir):
    out = {}
    for root, subdirs, files in os.walk(src_dir):
        if root not in out:
            out[root] = []
        for f in files:
            if os.path.splitext(f)[-1] not in ('.py', '.pyc'):
                out[root].append(os.path.join(root, f))
    return out


excluded_pyx = ['_Old', '_Todo', 'StatisticsPointListCode', 'flow']
extension_paths = [str(p) for p in Path('ClearMap').rglob('*.pyx') if not any([excl in str(p) for excl in excluded_pyx])]

extra_args = ['-fopenmp'] if USE_OPENMP else []
extensions = []
for ext_path in extension_paths:
    extension = Extension(
        name=module_path_to_doted(ext_path),
        sources=[ext_path],
        libraries=['m'],
        language='c++',  # WARNING: should be in file header
        include_dirs=[np.get_include(), os.path.dirname(os.path.abspath(ext_path))],
        extra_compile_args=DEFAULT_COMPILE_ARGS+extra_args,
        extra_link_args=DEFAULT_LINK_ARGS+extra_args,
    )
    extensions.append(extension)

# ext_modules = cythonize(extensions, nthreads=N_PROCS, quiet=True)
with ProcessPoolExecutor(max_workers=N_PROCS) as executor:
    parametrized_cythonize = functools.partial(cythonize, quiet=True)
    ext_modules = executor.map(parametrized_cythonize, extensions)
    ext_modules = [item for sublist in ext_modules for item in sublist]

data_dirs = [
    'ClearMap/External/elastix',
    'ClearMap/External/geodesic_distance',
    'ClearMap/Resources',
    'ClearMap/ImageProcessing/Skeletonization',   # .npy and .npy.zip
    'ClearMap/ImageProcessing/Binary',  # .npy and .npy.zip
    'ClearMap/ImageProcessing/machine_learning/vessel_filling/resources'  # .pth
]
data_files = [('licenses', ['LICENSE.txt', 'LICENSE'])]
for p in data_dirs:
    data_files.extend([(k, v) for k, v in find_data_files(p).items()])
data_files.extend([('', ['start_gui.sh'])])

packages = find_packages(exclude=('doc', 'tests*', 'pickle_python_2'))
setup(
    name='ClearMap2',
    version='2.1.0',
    description='3D cell counting and vasculature analysis for lightsheet microscopy',
    install_requires=requirements,
    packages=packages,
    ext_modules=ext_modules,
    entry_points={
        'gui_scripts': [
            'clearmap-ui = ClearMap.gui.run_gui:entry_point'
        ],
        'console_scripts': [
            'cell_map = ClearMap.Scripts.cell_map_new_api:main',
            'clearmap-utils = ClearMap.Scripts.clearmap_utils:main'
        ],
    },
    url='',
    license='MIT',
    author='Christoph Kirst, Sophie Skriabine, Charly Rousseau, Etienne Doumazane',
    author_email='',
    package_data={'ClearMap.config': ['*.cfg'],
                  'ClearMap.gui': ['creator/*.ui',
                                   'creator/*.qrc',
                                   'creator/graphics_resources/*.png',
                                   'creator/graphics_resources/*.jpg',
                                   'creator/graphics_resources/*.svg',
                                   'creator/icons/*.png',
                                   'creator/icons/*.jpg',
                                   'creator/icons/*.svg',
                                   ]},
    data_files=data_files,
    zip_safe=False
)
