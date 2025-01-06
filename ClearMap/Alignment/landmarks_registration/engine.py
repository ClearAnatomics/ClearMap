import re
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np

import itk

from ClearMap.Alignment.landmarks_registration.registration_data import ITKImage
from ClearMap.IO import IO as clmio


class ElastixPointFileManager:
    """
    A class to handle the reading and writing of points in elastix/transformix format.
    This is a text file with the following format:
    point  # this can be point or index
    n_points  # the number of points to be found below
    x1 y1 z1
    x2 y2 z2
    ...
    """
    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)  # FIXME: this is the base directory, not the file path

    def path(self, filename):
        if filename is None:
            return None
        else:
            if Path(filename).is_absolute():
                return filename
            else:
                return str(self.data_dir / filename)

    def write(self, points, filename=None, prefix=""):
        """
        Write points as elastix/transformix point file,  in physical coords

        Arguments
        ---------
        points : np.array
            array of coords in physical coords
        filename : str
            File name of the elastix point file.
        prefix : str
            Prefix to be added to the filename.

        Returns
        -------
        filename: str
            File name of the elastix point file.
        """

        if filename is None:
            filename = prefix + str(datetime.timestamp(datetime.now())).replace(".", "")
        with open(self.data_dir / filename, "w") as point_file:
            point_file.write("point\n")
            point_file.write(str(points.shape[0]) + "\n")
            np.savetxt(point_file, points, delimiter=" ", newline="\n", fmt="%.18e")

        return filename

    def read(self, filename):
        """
        Parses the output points from the output file of transformix

        Arguments
        ---------
        filename : str | Path
            File name of the transformix output file.


        Returns
        --------
        points : array
            The points coordinates in index coords.
        """

        my_regex = r"OutputIndexFixed = \[ ([^\]]*) \]"
        with open(self.path(filename)) as f:
            lines = f.readlines()
        try:
            for i, line in enumerate(lines):
                lines[i] = re.findall(my_regex, line)[0]
            points = np.loadtxt(lines)
        except IndexError:
            parsed_lines = [ln.strip() for ln in lines[2:]]
            if len(parsed_lines) != int(lines[1].strip()):
                raise ValueError("The number of points does not match the number of point information lines.")
            points = np.array([list(map(float, ln.split())) for ln in parsed_lines])

        return points

    def delete(self, filenames):
        for name in filenames:
            f_path = self.path(name)
            if f_path is None:
                continue
            f_path = Path(f_path)
            f_path.unlink(missing_ok=True)


class AlignmentTool:
    """A class to perform registration using elastix and landmarks (points) files."""

    def __init__(self, data_dir: str | Path, fixed_image: ITKImage = None, moving_image: ITKImage = None,
                 fixed_landmarks_name: str = "fixed_points.pts", moving_landmarks_name: str = "moving_points.pts"):
        """Instantiate object"""
        self.step = 0
        self.data_dir = Path(data_dir)
        self.point_manager = ElastixPointFileManager(data_dir)
        self.fixed_landmarks_name = fixed_landmarks_name
        self.moving_landmarks_name = moving_landmarks_name
        self.fixed_image = fixed_image
        self.moving_image = moving_image
        if self.fixed_image is not None and self.moving_image is not None:
            if self.fixed_image.ndim != self.moving_image.ndim:
                raise ValueError("Fixed and moving image must have the same number of dimensions.")
            self.ndim = self.fixed_image.ndim
        else:
            self.ndim = None
            warnings.warn("No registration data provided. You should set it before performing registration.")

        self.former_reg_params = None  # The params we just applied in the last transform  # Shouldn't these 2 match base name
        self.next_reg_params = None  # TODO: use queue ?

        self.transform_params = None  # The transform parameters (as in the transformation matrix computed) from the "former" registration

        self.pullback_image = None  # i.e. the transformed moving image

        self.fixed_points = None
        self.moving_points = None
        self.transformed_points = None  # registered points

    def try_load_points(self):
        if self.fixed_points is None:
            self.fixed_points = self.point_manager.read(self.fixed_landmarks_name)
        if self.moving_points is None:
            self.moving_points = self.point_manager.read(self.moving_landmarks_name)

    # In self's attributes, points are stored in index coords
    def get_points(self, as_index=False):
        """
        Get fixed and moving points arrays

        Arguments:
        ---------
        as_index (bool, optional):
            Whether the output should be in index coords. Defaults to False.
             If False, the output arrays will be in physical coords.

        Returns:
            tuple[np.array, np.array]: fixed_pts, moving_pts
        """
        self.try_load_points()
        fixed_pts = self.fixed_points
        moving_pts = self.moving_points
        if not as_index:
            fixed_pts = self.fixed_image.index_to_physical(fixed_pts)
            moving_pts = self.moving_image.index_to_physical(moving_pts)

        return fixed_pts, moving_pts

    @staticmethod
    def _convert_to_index_coords(image, points, as_index=False):
        if not as_index:
            points = image.physical_to_index(points)
        return points

    def set_points(self, fixed_points, moving_points, as_index=False):
        """Set corresponding points (index) position from n x dim point array

        Arguments:
        ---------
        fixed_points: np.array
            the array of moving points
        moving: np.array
            the array of moving points
        as_index (bool, optional):
            Whether the passed points are already in index coords. Defaults to False.
        """
        if fixed_points is not None:
            self.fixed_points = self._convert_to_index_coords(self.fixed_image, fixed_points, as_index=as_index)

        if moving_points is not None:
            self.moving_points = self._convert_to_index_coords(self.moving_image, moving_points, as_index=as_index)

    # TODO: tweak function below to allow interactive parameter choices (maybe a few choices only)
    def set_next_registration_parameters(self, parameters):  # itk.elxParameterObjectPython.elastixParameterObject
        """
        Set the parameters for the next registration and shift the current result image to the moving image

        Parameters
        ----------
        parameters: elastixParameterObject
            The parameters to be used in the next registration
        """
        self.next_reg_params = parameters
        self.former_reg_params = None  # Reset the former parameters

        if self.pullback_image is not None:
            self.moving_image = self.pullback_image
            self.pullback_image = None  # Reset the result image

    def perform_registration(self, parameters=None, delete_tmp=True, debug=False, write_transformed_points=False):
        """
        Perform the registration using the current fixed and moving images and the current registration parameters
        unless new parameters are passed.

        Parameters
        ----------
        parameters: elastixParameterObject, optional
            The parameters to be used in the next registration. Defaults to None.
        delete_tmp: bool, optional
            Whether to delete the temporary files created during the registration. Defaults to True.
        debug: bool, optional
            Whether to log the registration to the console. Defaults to False.
        write_transformed_points: bool, optional
            Whether to write the transformed points to a file. Defaults to False.
        """
        self.set_next_registration_parameters(parameters)
        fixed_points, moving_points = self.get_points(as_index=False)  # WARNING: always as physical coords
        if fixed_points.shape[0] != moving_points.shape[0]:
            raise ValueError("The two point lists do not have the same length.")

        # Write the temp files (convert to real coords)
        fixed_points_filename = self.point_manager.write(fixed_points, prefix="fixed")
        moving_points_filename = self.point_manager.write(moving_points, prefix="moving")

        reg_args = {'parameter_object': self.next_reg_params, 'log_to_console': debug}
        if fixed_points.size > 0:  # case with landmark points
            landmarks = {
                'fixed_point_set_file_name': self.point_manager.path(fixed_points_filename),
                'moving_point_set_file_name': self.point_manager.path(moving_points_filename)
            }
            reg_args.update(landmarks)
        pullback_image, self.transform_params = itk.elastix_registration_method(
            self.fixed_image.image,
            self.moving_image.image,
            **reg_args
        )
        self.pullback_image = pullback_image
        itk.imwrite(pullback_image, self.data_dir / f"result.{self.step}.mhd")  # REFACTOR: use self.result_template
        if not isinstance(self.pullback_image, ITKImage):
            self.pullback_image = ITKImage(self.pullback_image)

        # if some points were passed, compute the points transform to verify the registration
        # TODO: make optional
        if fixed_points.size > 0:
            point_paths_to_delete = [fixed_points_filename, moving_points_filename]
            if write_transformed_points:
                itk.transformix_pointset(
                    self.moving_image.image,
                    self.transform_params,
                    fixed_point_set_file_name=self.point_manager.path(fixed_points_filename),
                    output_directory=str(self.point_manager.data_dir),
                )
                output_points_path = Path(self.point_manager.path("outputpoints.txt"))
                if not output_points_path.exists():
                    raise FileNotFoundError("The output points file could not be found.")
                output_points_path = output_points_path.rename(self.data_dir / f"transformed_points.{self.step}.pts")  # REFACTOR: use self.transformed_points_template

                out_points = self.point_manager.read(output_points_path)
                self.transformed_points = out_points[:, self.moving_image.input_to_numpy_axis_perm]
                point_paths_to_delete.append(output_points_path)
            if delete_tmp:
                self.point_manager.delete(point_paths_to_delete)

        self.former_reg_params = self.next_reg_params
        self.step += 1

    # TODO: method to backup the current state with the three images, the used reg parameters, the alignment transform, etc.
