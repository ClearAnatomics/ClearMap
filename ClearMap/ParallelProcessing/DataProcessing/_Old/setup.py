#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 16:42:21 2020

@author: ckirst
"""

from distutils.core import setup
from Cython.Build import cythonize

import numpy as np


from distutils.extension import Extension
    
ext = Extension(name = 'test',
                sources = ["ArrayProcessingCode_New.pyx"],
                include_dirs = [np.get_include()],
                extra_compile_args = ["-O3", "-march=native", "-fopenmp"],
                extra_link_args = ['-fopenmp'],
                language='c++')


setup(name='test',
      ext_modules=cythonize(ext))
