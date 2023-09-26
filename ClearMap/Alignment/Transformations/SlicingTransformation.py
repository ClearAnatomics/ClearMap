
import numpy as np

from .Transformation import TransformationBase

from ClearMap.IO.Slice import sliced_shape, sliced_slicing
#TODO: organize this systematically

class SlicingTransformation(TransformationBase):

    ttype = 'Slicing'

    def __init__(self, shape, slicing, inverse=False):
        super().__init__(inverse=inverse)
        self.shape = shape
        self.slicing = slicing

    def start(self):
        return tuple(sl.start if sl.start is not None else 0 for sl in self.slicing)

    def step(self):
        return tuple(sl.step if sl.step is not None else 1 for sl in self.slicing)

    def stop(self):
        return tuple(sl.stop for sl in self.slicing)

    def transform_data(self, source, inverse=False, **kwargs):
        inverse = self.get_inverse(inverse)
        if not inverse:
            return source[self.slicing]
        else:
            raise NotImplementedError()

    def transform_points(self, source, inverse=False, **kwargs):
        inverse = self.get_inverse(inverse)
        shift = np.array(self.start())
        step = np.array(self.step())
        if np.any(step < 0):   # TODO: account for negative steps
            return NotImplementedError()
        transformed = source
        if not inverse:
            if not np.all(shift == 0):
                transformed = transformed - shift[np.newaxis, :]
            if not np.all(step == 1):
                transformed = transformed / step[np.newaxis,:]
        else:
            if not np.all(step == 1):
                transformed = transformed * step[np.newaxis, :]
            if not np.all(shift == 0):
                transformed = transformed + shift[np.newaxis, :]

        return transformed

    def transform_shape(self, shape, inverse=False, **kwargs):
        inverse = self.get_inverse(inverse)
        if not inverse:
            return sliced_shape(self.slicing, shape)
        else:  # not invertible
            return NotImplementedError()

    def transform_slicing(self, slicing, inverse=False, **kwargs):
        inverse = self.get_inverse(inverse)
        if not inverse:
            return sliced_slicing(slicing, self.slicing, self.shape)
        else:  # not invertible
            return NotImplementedError()

    def to_dict(self) -> dict:
        dictionary = super().to_dict()
        dictionary.update(shape=self.shape, slicing=self.slicing)
        return dictionary

    def __repr__(self):
        return '%s[%r, %r]' % (super().__repr__(), self.shape, self.slicing)
