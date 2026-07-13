import os
import sys
import subprocess
import platform

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


###############################################################################
# Parallel C/C++ compilation
###############################################################################

from setuptools.command.build_ext import build_ext as _build_ext

# N_PROCS must be defined before this class is used.
class parallel_build_ext(_build_ext):
    def initialize_options(self):
        super().initialize_options()
        self.parallel = N_PROCS


###############################################################################
# Platform helpers
###############################################################################

def _to_msvc_flags(flags):
    """Convert GCC-style flags to MSVC equivalents where possible."""
    out = []
    for arg in flags:
        if arg == '-fopenmp':
            out.append('/openmp')
        elif arg.startswith('-m'):
            # -march=native, -mtune=native → no MSVC equivalent, skip
            continue
        elif arg == '-O3':
            out.append('/O2')  # MSVC max optimisation
        elif arg == '-w':
            out.append('/w')   # disable warnings
        elif arg.startswith('-f'):
            # -flarge-source-files etc → no MSVC equivalent, skip
            continue
        else:
            out.append(arg)
    return out

requirements = [  # pip
    'lxml'
]
os_name = platform.system().lower()

OPTIMISE_COMPILATION_FOR_TARGET = os.environ.get('OPTIMISE_CLEARMAP_COMPILATION_FOR_TARGET', True) != 'False' # Cython code will run on the machine where is was compiled
N_PROCS = max(1, cpu_count() - 2)

DEFAULT_COMPILE_ARGS = []
DEFAULT_LIBRARIES = []
DEFAULT_LINK_ARGS = []

if not os_name.startswith('windows'):
    DEFAULT_COMPILE_ARGS += ['-w', '-O3']
    DEFAULT_LIBRARIES += ['m']

if os_name.startswith('linux'):
    out = subprocess.check_output(['gcc', '--help=common']).decode('utf-8')
    if '-flarge-source-files' in out:  # Add only if supported by the compiler
        DEFAULT_COMPILE_ARGS += ['-flarge-source-files']


if '--no-openmp' in sys.argv:
    sys.argv.remove('--no-openmp')
    USE_OPENMP = False
else:
    if os_name.startswith('linux') or os_name.startswith('windows'):
        USE_OPENMP = True
    elif os_name.startswith('darwin'):
        cpp_compiler_version = subprocess.check_output(['c++', '--version']).decode('ascii').split('\n')[0]
        USE_OPENMP = 'g++' in cpp_compiler_version
    else:
        raise ValueError(f'Unknown OS {os_name}')

if OPTIMISE_COMPILATION_FOR_TARGET and not os_name.startswith('windows'):
    DEFAULT_COMPILE_ARGS += ['-march=native', '-mtune=native']


def module_path_to_dotted(ext_path):
    return os.path.splitext(ext_path)[0].replace(os.sep, '.')


def find_data_files(src_dir):
    out = {}
    for root, subdirs, files in os.walk(src_dir):
        if root not in out:
            out[root.replace(os.sep, '/')] = []
        for f in files:
            if os.path.splitext(f)[-1] not in ('.py', '.pyc'):
                out[root.replace(os.sep, '/')].append(f"{root}/{f}".replace(os.sep, '/'))
    return out


excluded_pyx = ['_Old', '_Todo', 'StatisticsPointListCode', 'flow', 'OrientationCode']
extension_paths = [str(p) for p in Path('ClearMap').rglob('*.pyx')
                   if not any(excl in str(p) for excl in excluded_pyx)]

###############################################################################
# OpenMP flags (platform-conditional — NOT in file headers)
###############################################################################

extra_compile_args = list(DEFAULT_COMPILE_ARGS)
extra_link_args = list(DEFAULT_LINK_ARGS)

if USE_OPENMP:
    extra_compile_args += ['-fopenmp']
    extra_link_args += ['-fopenmp']

if os_name.startswith('windows'):
    extra_compile_args = _to_msvc_flags(extra_compile_args)
    extra_link_args = _to_msvc_flags(extra_link_args)

extensions = []
for ext_path in extension_paths:
    extension = Extension(
        name=module_path_to_dotted(ext_path),
        sources=[ext_path],
        libraries=DEFAULT_LIBRARIES,
        # language= intentionally omitted — read from #distutils: header
        include_dirs=[np.get_include(), os.path.dirname(os.path.abspath(ext_path))],
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
    )
    extensions.append(extension)

print(f'Building {len(extensions)} extensions with {N_PROCS} processes, '
      f'using OpenMP: {USE_OPENMP}, '
      f'libraries: {DEFAULT_LIBRARIES}, ',
      f'compile args: {extra_compile_args},'
      f'link args: {extra_link_args}')
if os_name.startswith('darwin') or os_name.startswith('windows'):
    ext_modules = cythonize(extensions, quiet=True)
else:
    ext_modules = cythonize(extensions, nthreads=N_PROCS, quiet=True)

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

packages = find_packages(exclude=('doc', 'tests*', 'pickle_python_2', 'deprecated',
                                  'ClearMap.External.elastix', 'ClearMap.External.geodesic_distance'))

setup(
    name='ClearMap',
    version='3.1.0',
    description='3D cell counting and vasculature analysis for lightsheet microscopy',
    install_requires=requirements,
    packages=packages,
    ext_modules=ext_modules,
    cmdclass={'build_ext': parallel_build_ext},
    entry_points={
        'gui_scripts': [
            'clearmap-ui = ClearMap.gui.app:entry_point'
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
    include_package_data=True,
    package_data={
        'ClearMap.config': [
            '*.cfg',
            'defaults/*/*.yml',
            'schemas/*/*.yaml',
            'defaults/*/*.cfg',
        ],
        'ClearMap.gui': [
            'creator/*.ui',
            'creator/*.qrc',
            'creator/graphics_resources/*.png',
            'creator/graphics_resources/*.jpg',
            'creator/graphics_resources/*.svg',
            'creator/icons/*.png',
            'creator/icons/*.jpg',
            'creator/icons/*.svg',
        ],
    },
    data_files=data_files,
    zip_safe=False
)
