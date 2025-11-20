import numpy as np

from .Transformation import TransformationBase


class AffineTransformation(TransformationBase):

    ttype = 'Affine'

    def __init__(self, m, b=None, inverse=False):
        super().__init__(inverse=inverse)
        self.m = m
        self.b = b or 0

    def transform_data(self, source, inverse=False, **kwargs):
        raise NotImplementedError()

    def transform_points(self, source, inverse=False, **kwargs):
        inverse = self.get_inverse(inverse)
        m = self.m
        if inverse:
            m = np.linalg.inv(self.m)
        return np.einsum('ij,...j->...i', m, source) + self.b

    def transform_shape(self, shape, inverse=False, **kwargs):
        raise NotImplementedError()

    def to_dict(self) -> dict:
        dictionary = super().to_dict()
        dictionary.update(m=self.m, b=self.b)
        return dictionary

    def __repr__(self):
        m = ','.join(['%r' % list(r) for r in self.m])
        b = self.b
        if not np.isscalar(b):
            b = list(b)
        return '%s[%s, %r]' % (super().__repr__(), m, b)


class ScalingTransformation(TransformationBase):

    ttype = 'Scaling'

    def __init__(self, scaling, inverse=False):
        super().__init__(inverse=inverse)
        self.scaling = scaling

    def get_scaling(self, ndim=None, inverse=False):
        inverse= self.get_inverse(inverse)
        scaling = self.scaling
        if np.isscalar(scaling):
            ndim = ndim or 1
            scaling = tuple(scaling for d in range(ndim))
        if inverse:
            scaling = tuple(1.0/scale for scale in scaling)
        return scaling

    def transform_data(self, source, inverse=False, **kwargs):
        raise NotImplementedError()

    def transform_points(self, source, inverse=False, **kwargs):
        scaling = self.get_scaling(source.shape[-1], inverse=inverse)
        return source * scaling

    def transform_shape(self, shape, inverse=False, **kwargs):
        scaling = self.get_scaling(len(shape), inverse=inverse)
        return tuple(int(np.ceil(scale * s)) for scale, s in zip(scaling, shape))

    def to_dict(self) -> dict:
        dictionary = super().to_dict()
        dictionary.update(scaling=self.scaling)
        return dictionary

    def __repr__(self):
        return '%s[%r]' % (super().__repr__(), self.scaling)
