"""
Rank filter module.

This module provides 3d filters based on the local histogram / rank order of an image.
"""
__author__    = 'Christoph Kirst <ckirst@rockefeller.edu>'
__license__   = 'MIT License <http://www.opensource.org/licenses/mit-license.php>'
__copyright__ = 'Copyright 2018 by Christoph Kirst, The Rockefeller University, New York City'
#
#
from .Rank import *           #analysis:ignore
from .Percentile import *     #analysis:ignore
from .Bilateral import *      #analysis:ignore
from .Parametric import *     #analysis:ignore


from . import Rank as rnk
from . import Percentile as per
from . import Bilateral as bil
from . import Parametric as par

__all__ = rnk.__all__ + per.__all__ + bil.__all__ + par.__all__;