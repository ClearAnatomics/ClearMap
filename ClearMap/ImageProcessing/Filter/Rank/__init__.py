"""
Rank filter module.

This module provides 3d filters based on the local histogram / rank order of an image.
"""
__author__    = 'Christoph Kirst <christoph.kirst.ck@gmail.com>'
__license__   = 'GPLv3 - GNU General Pulic License v3 (see LICENSE.txt)'
__copyright__ = 'Copyright © 2020 by Christoph Kirst'
__webpage__   = 'http://idisco.info'
__download__  = 'http://www.github.com/ChristophKirst/ClearMap2'


from .Rank import *           #analysis:ignore
from .Percentile import *     #analysis:ignore
from .Bilateral import *      #analysis:ignore
from .Parametric import *     #analysis:ignore


from . import Rank as rnk
from . import Percentile as per
from . import Bilateral as bil
from . import Parametric as par

__all__ = rnk.__all__ + per.__all__ + bil.__all__ + par.__all__;