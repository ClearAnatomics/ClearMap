#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Temporary
=========

Utility functions to create temporary files.
"""
__author__    = 'Christoph Kirst <ckirst@rockefeller.edu>'
__license__   = 'MIT License <http://www.opensource.org/licenses/mit-license.php>'
__copyright__ = 'Copyright (c) 2017 by Christoph Kirst, The Rockefeller University, New York City'


import os
import tempfile


def temporary_filename(prefix = None, postfix = None):
  prefix = prefix + '_' if prefix is not None else '';
  postfix = postfix + '_' if postfix is not None else ''; 
  
  return os.path.join(tempfile.gettempdir(), prefix + next(tempfile._get_candidate_names()) + postfix)