#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Temporary
=========

Utility functions to create temporary files.
"""
__author__    = 'Christoph Kirst <christoph.kirst.ck@gmail.com>'
__license__   = 'GPLv3 - GNU General Pulic License v3 (see LICENSE.txt)'
__copyright__ = 'Copyright © 2020 by Christoph Kirst'
__webpage__   = 'http://idisco.info'
__download__  = 'http://www.github.com/ChristophKirst/ClearMap2'


import os
import tempfile


def temporary_filename(prefix = None, postfix = None):
  prefix = prefix + '_' if prefix is not None else '';
  postfix = postfix + '_' if postfix is not None else ''; 
  
  return os.path.join(tempfile.gettempdir(), prefix + next(tempfile._get_candidate_names()) + postfix)