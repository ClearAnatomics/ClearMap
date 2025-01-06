from pathlib import Path
from typing import Optional
from dataclasses import dataclass

import itk
import numpy as np


# WARNING: required to have access to module
#  We execute two commands to force itk to load subpackages, so as to have the following type available.
#  types hidden by itk lazy loading:
#    - itk.itkImagePython
#    - itk.elxParameterObjectPython.elastixParameterObject
from itk.elxParameterObjectPython import elastixParameterObject
from itk import itkImagePython
itk.ParameterObject.New()


# Warning: below, the geometric dimension of the image is merely inferred from np.array shape,
#  this works only for 1-channel images.
# TODO: deal with multichannel images


def index_to_physical_matrix(origin, spacing, direction=None):
    if len(origin) != len(spacing):
        raise ValueError("origin and spacing should be coordinate lists of the same size.")
    if direction is None:
        direction = np.eye(len(origin))

    matrix_dim = len(origin) + 1
    if direction.shape != (matrix_dim - 1,) * 2:
        raise ValueError("direction shape must be  (d,d) where d is the length of origin")

    scaling_matrix = np.eye(matrix_dim, dtype="float64")
    translation_matrix = np.eye(matrix_dim, dtype="float64")
    rotation_matrix = np.eye(matrix_dim, dtype="float64")

    for i in range(matrix_dim - 1):
        scaling_matrix[i, i] = spacing[i]
        translation_matrix[i, -1] = origin[i]

    rotation_matrix[:-1, :-1] = direction
    print(rotation_matrix, translation_matrix, scaling_matrix)

    return np.linalg.inv(translation_matrix) @ rotation_matrix @ scaling_matrix


class ITKImage:
    def __init__(self, source: str | np.ndarray, perm=None):
        """_summary_

        Arguments:
        ----------
        source (str | np.ndarray):
            path to the image file or the image array
        perm (_type_, optional):
            permutation to apply to the coords (see code). Defaults to None.
             If None, the perm is the reversing of the coords

        Raises:
            ValueError: _description_
            ValueError: _description_
        """
        if isinstance(source, (str, Path)):
            source = str(source)
            if source.endswith('.npy'):
                self.image = np.load(source)  # TODO: see if same as below
            else:
                self.image = itk.imread(source, itk.F)
        else:
            self.image = itk.GetImageFromArray(np.ascontiguousarray(source))

        self.ndim = len(self.image.shape)
        if perm is not None:
            if len(perm) != self.ndim:
                raise ValueError("the provided perm list is not a permutation of range(ndim)")
        else:
            perm = list(range(self.ndim))[::-1]
        self.input_to_numpy_axis_perm = perm
        try:
            self.numpy_to_input_axis_perm = [perm.index(i) for i in range(self.ndim)]
        except:
            raise ValueError("the provided perm list is not a permutation of range(ndim)")
        self.meta_dict = dict(self.image)
        self._ipmatrix = None

    @property
    def index_to_physical_matrix(self):
        """Return the physical to index affine transformation, as a projective transform."""
        if self._ipmatrix is None:
            self._ipmatrix = index_to_physical_matrix(
                self.meta_dict["origin"],
                self.meta_dict["spacing"],
                self.meta_dict["direction"],
            )
        return self._ipmatrix.transpose()

    @property
    def index_to_physical_pair(self):
        """
        Return the physical to index affine transformation,
        as a pair (matrix, translation_vector).
        """
        dim = self.image.ndim
        translation_vector = np.array(
            list(self.image.TransformIndexToPhysicalPoint((0,) * dim))
        ).reshape(1, -1)
        matrix = (
            np.array(
                [
                    list(
                        self.image.TransformIndexToPhysicalPoint(
                            list((1 if i == j else 0 for j in range(dim)))
                        )
                    )
                    for i in range(dim)
                ]
            )
            - translation_vector
        )

        return matrix, translation_vector

    # def physical_to_index(self, points):
    #     return (points @ np.linalg.inv(self.index_to_physical_matrix))[:, :-1]

    def physical_to_index(self, points, perm=None):
        if perm is None:
            perm = self.input_to_numpy_axis_perm
        matrix, translation_vector = self.index_to_physical_pair
        return ((points - translation_vector) @ np.linalg.inv(matrix))[:, perm]

    def index_to_physical(self, points, perm=None):
        if perm is None:
            perm = self.numpy_to_input_axis_perm
        matrix, translation_vector = self.index_to_physical_pair
        return (points[:, perm]) @ matrix + translation_vector

    def nice_contrast_limits(self):
        return (0, np.percentile(self.image, 99.97))

    def brutal_rotate(self, rotation_axis, power=1):
        """Perform a 90° rotation of the 3d image without editing the metadata accordingly."""

        if self.image.ndim != 3:
            raise ValueError("this method is only for 3d images")

        power = power % 4
        moving = [i for i in range(3) if i != rotation_axis]

        for i in range(power):
            self.image = itk.GetImageFromArray(
                np.ascontiguousarray(
                    np.flip(np.swapaxes(self.image, *moving), moving[0])
                )
            )


@dataclass
class RegistrationData:
    """
    Class that retains the data of a registration attempt.

    Encapsulate fixed and moving image and points, chosen elastix registration parameters,
    computed transform parameters, transformed image, transformed fixed points, registration
    scores for various metrics. All points are stored in index coordinates (numpy order),
    as the data arrays of napari.
    """

    fixed_image: ITKImage
    moving_image: ITKImage
    fixed_points: Optional[np.array] = None
    moving_points: Optional[np.array] = None
    reg_params: Optional[elastixParameterObject] = None
    pullback_image: Optional[itkImagePython] = None
    transform_params: Optional[elastixParameterObject] = None
