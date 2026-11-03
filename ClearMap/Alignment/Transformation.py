# -*- coding: utf-8 -*-
"""
Transformation
==============

Transformation classes to track transformations of data, points and shapes.

Note
----
These classes are used to keep track of location and scaling when cropping or resampling data
or mapping between raw data and reference atlases.
"""
__author__ = 'Christoph Kirst <christoph.kirst.ck@gmail.com>'
__license__ = 'GPLv3 - GNU General Public License v3 (see LICENSE)'
__copyright__ = 'Copyright © 2023 by Christoph Kirst'

from .Transformations.Transformation import Transformation, ttype_to_transformation
from .Transformations.LinearTransformation import AffineTransformation, ScalingTransformation
from .Transformations.SlicingTransformation import SlicingTransformation
from .Resampling import ResamplingTransformation, OrientationTransformation
from .Elastix import ElastixTransformation

# register transformations in factory

ttype_to_transformation.update(
    { transformation.ttype : transformation for transformation in
        (
            Transformation,
            AffineTransformation,
            ScalingTransformation,
            SlicingTransformation,
            OrientationTransformation,
            ResamplingTransformation,
            ElastixTransformation
        )
    }
)