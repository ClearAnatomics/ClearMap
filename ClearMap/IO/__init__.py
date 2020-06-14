# -*- coding: utf-8 -*-
"""
IO
==

This module handels data reading and writing in ClearMap via sources and
sinks.

The base class for IO is a :mod:`~ClearMap.IO.Source` that automatically 
detects file type and gathers some basic infomration about the file.

>>> import ClearMap.IO.IO as io
>>> import numpy as np
>>> x = np.random.rand(30,40,50);
>>> s = io.as_source(x);
>>> print(s);
Numpy-Source(30, 40, 50)[float64]|C|

The first tuple is the shape of the array, the next list contains the datype 
or data structure, the |.| bracket denotes the order of this array, which 
can be |C| for c-contigous, |F| for Fortran contigous or '' if the array
is non-contigous.

File sources contain further information about the location of the files etc.

The :mod:`~ClearMap.IO.FileList` source allows to turn a list of files into a 
sinlge source, e.g. for microscope data in which each imageing plane comes
in a separate file.

Note
----
Sources can be sliced as numpy arrays. All sources can be made virtual,
containing only pointers to data in order to use them in parallel processing.

"""
__author__    = 'Christoph Kirst <christoph.kirst.ck@gmail.com>'
__license__   = 'GPLv3 - GNU General Pulic License v3 (see LICENSE.txt)'
__copyright__ = 'Copyright © 2020 by Christoph Kirst'
__webpage__   = 'http://idisco.info'
__download__  = 'http://www.github.com/ChristophKirst/ClearMap2'
