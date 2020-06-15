# -*- coding: utf-8 -*-
"""
ParallelProcessing
==================

This sub-package provides functions for parallel processing of data
for ClearMap.

The main functionality is to dsitribute 
:mod:`~ClearMap.ParallelProcessing.Blocks` of data to be processed in parallel
manged by :mod:`ClearMap.ParallelProcessing.BlockProcessing`.

The :mod:`~ClearMap.ParallelProcessing.DataProcessing` sub-module 
contains cython implementations for a set of functions to be applied to
large arrays in parallel with out sub-dividing the data into blocks
but using memory mapping and the buffer interface.
""" 
__author__    = 'Christoph Kirst <christoph.kirst.ck@gmail.com>'
__license__   = 'GPLv3 - GNU General Pulic License v3 (see LICENSE)'
__copyright__ = 'Copyright © 2020 by Christoph Kirst'
__webpage__   = 'http://idisco.info'
__download__  = 'http://www.github.com/ChristophKirst/ClearMap2'