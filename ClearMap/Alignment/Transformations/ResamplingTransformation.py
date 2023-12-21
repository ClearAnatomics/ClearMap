
from .Transformation import TransformationBase
from ClearMap.Alignment.Resampling import resample_information, resample_factor, resample, resample_inverse, resample_points, \
    resample_points_inverse, resample_shape
from ClearMap.Alignment.orientation import format_orientation, orient_shape, orient, orient_points

import ClearMap.IO.IO as io

########################################################################################
# Transformation interface
########################################################################################

class ResamplingTransformation(TransformationBase):
    def __init__(self, ttype='Resampling', inverse=False,
                 original_resolution=None, resampled_resolution=None,
                 original_shape=None, resampled_shape=None,
                 original=None, resampled=None,
                 orientation=None):

        super().__init__(ttype=ttype, inverse=inverse)

        original_shape, resampled_shape, original_resolution, resampled_resolution, orientation = \
            resample_information(original_shape, resampled_shape,
                                 original_resolution, resampled_resolution,
                                 original, resampled,
                                 orientation, discretize=True, consistent=True)

        self.original_shape = original_shape
        self.resampled_shape = resampled_shape
        self.original_resolution = original_resolution
        self.resampled_resolution = resampled_resolution
        self.orientation = orientation

    def resample_kwargs(self):
        return dict(original_shape=self.original_shape, resampled_shape=self.resampled_shape,
                    resampled_resolution=self.resampled_resolution, original_resolution=self.original_resolution,
                    orientation=self.orientation)

    def resample_factor(self):
        return resample_factor(**self.resample_kwargs())

    def transform_data(self, source, sink=None, inverse=False, **kwargs):
        inverse = self.get_inverse(inverse)
        resample_kwargs = self.resample_kwargs()
        resample_kwargs.update(kwargs)
        if not inverse:
            return resample(source, sink, **resample_kwargs)
        else:
            return resample_inverse(source, sink, **resample_kwargs)

    def transform_points(self, source, sink=None, inverse=False, **kwargs):
        inverse = self.get_inverse(inverse)
        resample_kwargs = self.resample_kwargs()
        resample_kwargs.update(kwargs)
        if not inverse:
            return resample_points(source, sink, **resample_kwargs)
        else:
            return resample_points_inverse(source, sink, **resample_kwargs)

    def transform_shape(self, shape, inverse=False, **kwargs):
        kwargs = self.resample_kwargs()
        kwargs.update(**kwargs)
        kwargs.update(original_shape=shape)
        return resample_shape(shape, **kwargs)

    def to_dict(self) -> dict:
        dictionary = super().to_dict()
        dictionary.update(self.resample_kwargs())
        return dictionary

    def __repr__(self):
        orientation = '' if self.orientation is None else ('%r ' % (self.orientation,))
        return '%s[%s%r->%r]' % (super().__repr__(), orientation, self.original_resolution, self.resampled_resolution)


class OrientationTransformation(TransformationBase):

    ttype = 'Orientation'

    def __init__(self, orientation=None, shape=None,inverse=None):
        super().__init__(inverse=inverse)
        self.orientation = format_orientation(orientation)
        self.shape = shape

    def transform_data(self, source, sink=None, inverse=False):
        inverse = self.get_inverse(inverse)
        return io.write(sink, orient(source, orientation=self.orientation, inverse=inverse))

    def transform_points(self, source, sink=None, inverse=False, **kwargs):
        inverse = self.get_inverse(inverse)
        return io.write(sink, orient_points(source, orientation=self.orientation, shape=self.shape, inverse=inverse))

    def transform_shape(self, shape, inverse=False, **kwargs):
        return orient_shape(self.orientation, shape, inverse=inverse)

    def to_dict(self) -> dict:
        dictionary = super().to_dict()
        dictionary.update(orientation=self.orientation,
                          shape=self.shape)
        return dictionary

    def __repr__(self):
        orientation = '' if self.orientation is None else ('%r' % (self.orientation,))
        shape = '' if self.shape is None else (' %r ' % (self.shape,))
        return '%s[%s%s]' % (super().__repr__(), orientation, shape)
