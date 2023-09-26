# -*- coding: utf-8 -*-
"""
Transformation
==============

Abstract base class to handle transformation of coordinates and data
when cropping data or mapping between raw data and reference atlases.
"""
__author__ = 'Christoph Kirst <christoph.kirst.ck@gmail.com>'
__license__ = 'GPLv3 - GNU General Public License v3 (see LICENSE)'
__copyright__ = 'Copyright © 2020 by Christoph Kirst'


import io
import pickle
import numpy as np


# Base class

class TransformationBase(object):
    """Base class for transformations of data, points and shapes."""

    ttype=None

    def __init__(self, inverse=False):
        self.inverse = inverse

    def transform_points(self, source, inverse=False, **kwargs):
        raise NotImplementedError()

    def transform_data(self, source, inverse=False, **kwargs):
        raise NotImplementedError()

    def transform_shape(self, shape, inverse=False, **kwargs):
        raise NotImplementedError()

    def get_inverse(self, inverse=False):
        return self.inverse if not inverse else not self.inverse

    def to_dict(self) -> dict:
        return {'ttype': self.ttype,
                'inverse' : self.inverse}

    @classmethod
    def from_dict(cls, dictionary: dict):
        return cls(**dictionary)

    def write(self, filename):
        with open(filename, 'wb') as file:
            pickle.dump(self.to_dict(), file, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def read(cls, filename):
        with io.open(filename, 'rb') as file:
            dictionary = pickle.load(file)
        return cls.from_dict(dictionary)

    def __repr__(self):
        inverse = '|i|' if self.inverse else ''
        return '%s%s' % (self.ttype, inverse)


# Factory

ttype_to_transformation = {}


def transformation_from_dict(dictionary):
    ttype = dictionary.get('ttype', None)
    if ttype not in ttype_to_transformation.keys():
        raise ValueError('No transformation of type %r' % ttype)
    return ttype_to_transformation[ttype].from_dict(dictionary)


# Transformations

class Transformation(TransformationBase):

    ttype = 'Transformation'

    def __init__(self, transformations=None, inverse=False):
        super().__init__(inverse=inverse)
        if transformations is None:
            transformations = []
        self.transformations = transformations

    def get_transformations(self, inverse=False, start_at=None, stop_at=None) -> list:
        inverse = self.get_inverse(inverse)
        transformations = self.transformations
        if start_at is not None:
            types = [transformation.ttype for transformation in transformations]
            if start_at not in types:
                raise ValueError('Transformation has no sub-transformation %r (%r)' % (start_at, types))
            for i, ttype in enumerate(types):
                if ttype == start_at:
                    transformations = transformations[i:]
                    break
        if stop_at is not None:
            types = [transformation.ttype for transformation in transformations]
            if stop_at not in types:
                raise ValueError('Transformation has no sub-transformation %r (%r)' % (stop_at, types))
            for i, ttype in enumerate(types):
                if ttype == stop_at:
                    transformations = transformations[:i + 1]
                    break
        if inverse:
            transformations = transformations[::-1]
        return transformations

    def transform_points(self, source, inverse=False, start_at=None, stop_at=None, **kwargs):
        transformations = self.get_transformations(inverse=inverse, start_at=start_at, stop_at=stop_at)
        transformed_points = source
        for transformation in transformations:
            transformed_points = transformation.transform_points(transformed_points, inverse=inverse, **kwargs)
        return transformed_points

    def transform_shape(self, shape, inverse=False, start_at=None, stop_at=None, **kwargs):
        transformations = self.get_transformations(inverse=inverse, start_at=start_at, stop_at=stop_at)
        transformed_shape = shape
        for transformation in transformations:
            transformed_shape = transformation.transform_shape(transformed_shape, inverse=inverse, **kwargs)
        return transformed_shape

    def transform_data(self, source, inverse=False, start_at=None, stop_at=None, **kwargs):
        transformations = self.get_transformations(inverse=inverse, start_at=start_at, stop_at=stop_at)
        transformed_data = source
        for transformation in transformations:
            transformed_data = transformation.transform_points(transformed_data, inverse=inverse, **kwargs)
        return transformed_data

    def to_dict(self) -> dict:
        dictionary = super().to_dict()
        dictionary.update(transformations=[transformation.to_dict() for transformation in self.transformations])
        return dictionary

    @classmethod
    def from_dict(cls, dictionary: dict):
        transformations = [transformation_from_dict(transformation) for transformation in dictionary['transformations']]
        return cls(transformations=transformations)

    def __len__(self):
        return len(self.transformations)

    def __getitem__(self, item):
        if isinstance(item, int):
            return self.transformations.__getitem__(item)
        else:
            transformations = self.transformations.__getitem__(item)
            return Transformation(transformations=transformations)

    def __setitem__(self, *args):
        self.transformations.__setitem__(*args)

    def __repr__(self):
        return "%s[\n   %s]" % (super().__repr__(), '\n ->'.join([transformation.__repr__() for transformation in self.transformations]))




###############################################################################
# Test
###############################################################################

def _test():
    import numpy as np
    import ClearMap.Alignment.Transformation as tfm

    from importlib import reload
    reload(tfm)

    # combined transformations
    a1 = tfm.AffineTransformation(M=5 * np.identity(3))
    a2 = tfm.AffineTransformation(M=np.identity(3), b=3)
    t = tfm.Transformation(transformations=[a1,a2])
    print(t)

    points = np.random.rand(5,3)
    transformed = t.transform_points(points)
    print(np.allclose(points * 5 + 3, transformed))

    # slicing
    data = np.random.rand(30,50)
    slicing = (slice(5,15,3), slice(None,40))
    data_sliced = data[slicing]

    s = tfm.SlicingTransformation(slicing=slicing, shape=data.shape)
    print(s)
    print(np.all(s.transform_data(data) == data_sliced))

    points_sliced = np.array(np.where(data_sliced > 0.75)).T
    points = s.transform_points(points_sliced, inverse=True)
    print(np.all(data[points[:,0], points[:,1]] == data_sliced[points_sliced[:,0], points_sliced[:,1]]))


