# build_ext_custom.py
from setuptools.command.build_ext import build_ext as _build_ext
from setuptools.extension import Extension
from Cython.Build import cythonize
import numpy as np
import os
import subprocess
from pathlib import Path
import platform

os_name = platform.system().lower()

DEFAULT_COMPILE_ARGS = []
DEFAULT_LIBRARIES = []
DEFAULT_LINK_ARGS = []

if not os_name.startswith('windows'):
    DEFAULT_COMPILE_ARGS += ['-w', '-O3']
    DEFAULT_LIBRARIES += ['m']

if os_name.startswith('linux'):
    try:
        out = subprocess.check_output(['gcc', '--help=common']).decode('utf-8')
        if '-flarge-source-files' in out:
            DEFAULT_COMPILE_ARGS += ['-flarge-source-files']
    except Exception:
        pass

USE_OPENMP = True
if os_name.startswith('darwin'):
    try:
        cpp_version = subprocess.check_output(['c++', '--version']).decode('ascii')
        USE_OPENMP = 'g++' in cpp_version
    except Exception:
        USE_OPENMP = False

extra_args = ['-fopenmp'] if USE_OPENMP else []
if os_name.startswith('windows'):
    extra_args = [arg.replace('-f', '/') for arg in extra_args]
    DEFAULT_COMPILE_ARGS = [arg.replace('-m', '/') for arg in DEFAULT_COMPILE_ARGS]
    DEFAULT_LINK_ARGS = [arg.replace('-m', '/') for arg in DEFAULT_LINK_ARGS]

extra_link_args = extra_args
compile_args = DEFAULT_COMPILE_ARGS + extra_args
link_args = DEFAULT_LINK_ARGS + extra_link_args


extension_paths = [str(p) for p in Path('ClearMap').rglob('*.pyx')
                   if not any(excl in str(p) for excl in ['_Old', '_Todo', 'StatisticsPointListCode', 'flow', 'OrientationCode'])]

extensions = [
    Extension(
        name=os.path.splitext(p)[0].replace(os.sep, '.'),
        sources=[p],
        libraries=DEFAULT_LIBRARIES,
        language='c++',
        include_dirs=[np.get_include(), os.path.dirname(os.path.abspath(p))],
        extra_compile_args=compile_args,
        extra_link_args=link_args
    ) for p in extension_paths
]

ext_modules = cythonize(extensions, quiet=True)

class build_ext(_build_ext):
    pass
