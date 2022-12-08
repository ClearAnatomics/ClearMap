import os
from multiprocessing import cpu_count

import numpy as np
from setuptools import setup, find_packages, Extension

from Cython.Build import cythonize

# from https://stackoverflow.com/a/3042436
# To modify the data install dir to match the source install dir...
# Pull the install dir info from distutils with:
from distutils.command.install import INSTALL_SCHEMES
# Modify the data install dir to match the source install dir:
for scheme in INSTALL_SCHEMES.values():
    scheme['data'] = scheme['purelib']

n_procs = cpu_count() - 2

requirements = [
    'lxml'
]
extension_paths = [
    'ClearMap/ParallelProcessing/DataProcessing/ArrayProcessingCode.pyx',
    'ClearMap/ParallelProcessing/DataProcessing/ConvolvePointListCode.pyx',
    'ClearMap/ParallelProcessing/DataProcessing/DevolvePointListCode.pyx',
    'ClearMap/ParallelProcessing/DataProcessing/MeasurePointListCode.pyx',
    'ClearMap/ParallelProcessing/DataProcessing/StatisticsPointListCode.pyx',

    'ClearMap/ImageProcessing/Binary/FillingCode.pyx',
    'ClearMap/ImageProcessing/Clipping/ClippingCode.pyx',
    'ClearMap/ImageProcessing/Differentiation/HessianCode.pyx',
    'ClearMap/ImageProcessing/Filter/Rank/RankCode.pyx',
    'ClearMap/ImageProcessing/Filter/Rank/RankCoreCode.pyx',
    'ClearMap/ImageProcessing/Thresholding/ThresholdingCode.pyx',
    'ClearMap/ImageProcessing/Tracing/TraceCode.pyx',

    'ClearMap/ImageProcessing/Filter/Rank/PercentileCode.pyx',
    'ClearMap/ImageProcessing/Filter/Rank/BilateralCode.pyx',
    'ClearMap/ImageProcessing/Filter/Rank/ParametricCode.pyx',
]
extensions = []


def module_path_to_doted(ext_path):
    return os.path.splitext(ext_path)[0].replace(os.sep, '.')


for ext_path in extension_paths:
    extension = Extension(
        name=module_path_to_doted(ext_path),
        sources=[ext_path],
        libraries=['m'],
        language='c++',  # WARNING: should be in file header
        include_dirs=[np.get_include(), os.path.dirname(os.path.abspath(ext_path))],
        extra_compile_args=['-w']
    )
    extensions.append(extension)
# ext_modules = cythonize(extensions, nthreads=n_procs, quiet=True, language_level="3")
ext_modules = cythonize(extensions, nthreads=n_procs, quiet=True)


def find_data_files(src_dir):
    out = {}
    for root, subdirs, files in os.walk(src_dir):
        if root not in out:
            out[root] = []
        for f in files:
            if os.path.splitext(f)[-1] not in ('.py', '.pyc'):
                out[root].append(os.path.join(root, f))
    return out


data_files = [('licenses', ['LICENSE.txt', 'LICENSE'])]

data_files.extend([(k, v) for k, v in find_data_files('ClearMap/External/elastix').items()])
data_files.extend([(k, v) for k, v in find_data_files('ClearMap/External/geodesic_distance').items()])
data_files.extend([(k, v) for k, v in find_data_files('ClearMap/Resources').items()])
data_files.extend([(k, v) for k, v in find_data_files('ClearMap/ImageProcessing/Skeletonization').items()])  # .npy and .npy.zip
data_files.extend([(k, v) for k, v in find_data_files('ClearMap/ImageProcessing/Binary').items()])  # .npy and .npy.zip
data_files.extend([(k, v) for k, v in find_data_files(
    'ClearMap/ImageProcessing/MachineLearning/VesselFilling/Resources').items()])  # .pth

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
            'clear_map_ui = ClearMap.gui.run_gui:entry_point'
        ],
        'console_scripts': [
             'cell_map = ClearMap.Scripts.cell_map_new_api:main',
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
