# -*- coding: utf-8 -*-
"""
Elastix
=======

Interface to Elastix for alignment of volumetric data.

The elastix documentation can be found `here <https://elastix.lumc.nl/>`_.

In essence, a transformation :math:`T(x)` is sought so that for a fixed image 
:math:`F(x)` and a moving image :math:`M(x)`:

.. math::
    F(x) = M(T(x))

Once the map :math:`T` is estimated via elastix, transformix maps an image
:math:`I(x)` from the moving image frame to the fixed image frame, i.e.:

.. math::
    I(x) \\rightarrow I(T(x)) 

To register an image onto a reference image, the fixed image is typically 
chosen to be the image to be registered, while the moving image is the
reference image. In this way a position :math:`x` in the data is mapped to the corresponding
reference position :math:`T(x)` via transformix as:

.. math::
    x \\rightarrow T(x)

Summary
-------
    * elastix finds a transformation :math:`T: \\mathrm{fixed image} \\rightarrow \\mathrm{moving image}`
    * the fixed image is typically the image to be registered
    * the moving image is typically the reference image
    * the result folder may contain an image (mhd file) that is :math:`T^{-1}(\\mathrm{moving})`,
      i.e. has the size of the fixed image
    * transformix applied to data gives :math:`T^{-1}(\\mathrm{data})` !
    * transformix applied to points gives :math:`T(\\mathrm{points})` !
    * point arrays are assumed to be in (x,y,z) coordinates consistent with (x,y,z) array representation
      of images in ClearMap

In terms of atlas alignment this reads as

Atlas alignment
---------------
    * elastix finds a transformation :math:`T: \\mathrm{data} \\rightarrow \\mathrm{atlas}`
    * the result folder may contain an image (mhd file) that is :math:`T^{-1}(\\mathrm{atlas})`,
      i.e. is the atlas transformed to the data.
    * transformix applied to the atlas gives :math:`T^{-1}(\\mathrm{atlas})`, i.e. the atlas mapped onto the data !
    * transformix applied to points gives :math:`T(\\mathrm{points})`, i.e. the points in the atlas reference frame !

Main routines are: :func:`align`, :func:`transform`.
    
See Also
--------
    `Elastix documentation <https://elastix.lumc.nl/doxygen/index.html>`_
    :mod:`~ClearMap.Alignment.Resampling`
"""
__author__ = 'Christoph Kirst <christoph.kirst.ck@gmail.com>'
__license__ = 'GPLv3 - GNU General Public License v3 (see LICENSE)'
__copyright__ = 'Copyright © 2020 by Christoph Kirst'
__webpage__ = 'https://idisco.info'
__download__ = 'https://github.com/ClearAnatomics/ClearMap'

import copy
import os
import sys
import subprocess
import platform
import tempfile
import shutil
import re
from io import UnsupportedOperation
from pathlib import Path
import multiprocessing as mp

import numpy as np


import ClearMap.Settings as settings
from ClearMap.Utils.exceptions import ClearMapException

import ClearMap.IO.IO as io
from ClearMap.IO.elastix_config import ElastixParser

from ClearMap.Alignment.Transformations.Transformation import TransformationBase

try:
    import itk
    dir(itk)  #  force read
    from ClearMap.Alignment.landmarks_registration.engine import AlignmentTool
    from ClearMap.Alignment.landmarks_registration.registration_data import ITKImage
    itk_imported = True
except ImportError:
    itk_imported = False

##############################################################################
# Initialization and Settings
##############################################################################

elastix_binary = None
"""The elastix executable.

Notes
-----
  - setup in :func:`initialize_elastix`
"""

elastix_lib = None
"""Path to the elastix library.

Notes
-----
  - setup in :func:`initialize_elastix`
"""

transformix_binary = None
"""The transformix executable.

Notes
-----
  - setup in :func:`initialize_elastix`
"""

initialized = False
"""Status of the elastix binaries and paths.

Notes
-----
  - setup in :func:`initialize_elastix`
"""


class ElastixConfigPatcher:
    def __init__(self, config_paths, modified_sections=None):
        self.config_paths = config_paths
        if modified_sections is None:
            raise ValueError('modified_sections must be a dictionary of sections to modify')
        self.modified_sections = modified_sections
        self.original_sections = {p: self.get_original_sections(p) for p in config_paths}

    def __enter__(self):
        for config_path in self.config_paths:
            if config_path:
                self.patch_elastix_cfg_landmarks(config_path)

    def __exit__(self, exc_type, exc_val, exc_tb):
        for config_path in self.config_paths:
            if config_path:
                self.restore_elastix_cfg_no_landmarks(config_path)

    def patch_elastix_cfg_landmarks(self, elastix_cfg_path):
        self.__patch_config(elastix_cfg_path, self.modified_sections)

    def restore_elastix_cfg_no_landmarks(self, elastix_cfg_path):
        self.__patch_config(elastix_cfg_path, self.original_sections)

    @staticmethod
    def __patch_config(config_path, sections):
        cfg = ElastixParser(config_path)
        for option, value in sections[config_path].items():
            cfg[option] = value
        cfg.write()

    def get_original_sections(self, config_path):
        original_sections = {}
        cfg = ElastixParser(config_path)
        for option in self.modified_sections[config_path].keys():
            value = cfg.get(option)
            if value is not None:
                original_sections[option] = value
        return original_sections


def print_settings():
    """Prints the current elastix configuration.

    See also
    --------
    :const:`elastix_binary`, :const:`elastix_lib`, :const:`transformix_binary`, :const:`initialized`
    """

    if initialized:
        print(f"elastix_binary     = {elastix_binary}")
        print(f"elastix_lib        = {elastix_lib}")
        print(f"transformix_binary = {transformix_binary}")
    else:
        print("Elastix not initialized")


def check_success(result_directory):
    result_directory = Path(result_directory)
    log_path = result_directory / 'elastix.log'
    with open(log_path) as f:
        if 'fail' in f.read():
            raise RuntimeError(f'Elastix failed, check log file: "{log_path}"')


def set_elastix_library_path(elastix_lib_path=None):
    """Add elastix library path to the LD_LIBRARY_PATH variable in linux.

    Arguments
    ---------
      elastix_lib_path : str or None
      Path to elastix root directory.
      If None :const:`ClearMap.Settings.elastix_path` is used.
    """
    os_name = platform.system().lower()
    if os_name.startswith('linux'):
        lib_var_name = 'LD_LIBRARY_PATH'
    elif os_name.startswith('darwin'):
        lib_var_name = 'DYLD_LIBRARY_PATH'
    else:
        raise ValueError(f'Unknown OS {os_name}')

    print(f'OS: {os_name}, library variable name: {lib_var_name}')

    if lib_var_name in os.environ:
        lib_path = os.environ[lib_var_name]
        print(f'Variable {lib_var_name} exists, patching with {lib_path}')
        if elastix_lib_path not in lib_path.split(':'):
            os.environ[lib_var_name] = f'{elastix_lib_path}:{lib_path}'
    else:
        print(f'Variable {lib_var_name} not found, adding elastix lib folder: {elastix_lib}')
        os.environ[lib_var_name] = elastix_lib_path


def initialize_elastix(path=None):
    """Initialize all paths and binaries of elastix

    Arguments
    ---------
      path (str or None): path to elastix root directory, if None
      :const:`ClearMap.Settings.elastix_path` is used.

    See also
    --------
      :const:`elastix_binary`, :const:`elastix_lib`, :const:`transformix_binary`,
      :const:`initialized`, :func:`set_elastix_library_path`
    """
    global elastix_binary, elastix_lib, transformix_binary, initialized

    if path is None and settings.elastix_path is not None:
        path = settings.elastix_path
    else:
        raise RuntimeError('Cannot find elastix path!')

    elastix_binary = search_elx_bin(path, 'elastix')
    transformix_binary = search_elx_bin(path, 'transformix')

    elastix_lib = search_elx_lib(path)
    set_elastix_library_path(elastix_lib)  # FIXME: check if needs global elastix_lib ??

    initialized = True
    print(f'Elastix successfully initialized from path: {path}')
    return path


def search_elx_lib(path):
    for sub_dir in ('lib', 'bin'):
        lib_path = os.path.join(path, sub_dir)
        if os.path.exists(lib_path):
            return lib_path
    else:
        raise ClearMapException(f'Cannot find elastix libs in {lib_path} set path in Settings.py accordingly!')


def search_elx_bin(path, bin_type):
    elx = os.path.join(path, 'bin', bin_type)
    if os.path.exists(elx):
        return elx
    else:
        raise RuntimeError(f'Cannot find elastix binary {elx}, set path in Settings.py accordingly!')


def get_elastix_version():
    """Returns the elastix version.

    Returns
    -------
    version : str
      The elastix version.
    """
    check_elastix_initialized()

    cmd = [elastix_binary, '--version']
    res = subprocess.check_output(cmd)
    res = res.decode('utf-8')
    version = float(res.splitlines()[0].split(':')[-1].strip())
    return version


if 'sphinx' in sys.modules.keys():
    print('Import for documentation: elastix not initialized!')
else:
    initialize_elastix()


def check_elastix_initialized():
    """Checks if elastix is initialized

    Returns
    -------
    initialized : bool
      True if elastix paths are set.
    """
    if not initialized:
        raise RuntimeError("Elastix not initialized: run initialize_elastix(path) with proper path to elastix first")

    return True


###############################################################################
# Basic interface routines
###############################################################################

def transform_file(result_directory):
    """Finds and returns the transformation parameter file.

    Arguments
    ---------
    result_directory : str
      Path to directory of elastix results.

    Returns
    -------
    filename : str
      File name of the first transformation parameter file.

    Notes
    -----
    In case of multiple transformation parameter files the top level file is
    returned.
    """

    files = sorted(Path(result_directory).glob('TransformParameters.*.txt'))

    if not files:
        raise RuntimeError(f'Cannot find a valid transformation file in {result_directory}!')

    return str(files[-1])


def transform_directory_and_file(transform_parameter_file=None, transform_directory=None):
    """Determines transformation directory and file from either.

    Arguments
    ---------
    transform_parameter_file : str or None
      File name of the transformation parameter file.
    transform_directory : str or None
      Directory to the transformation parameter.

    Returns
    -------
    transform_parameter_file : str
      File name of the transformation parameter file.
    transform_directory : str
      Directory to the transformation parameter.

    Notes
    -----
    Only one of the two arguments need to be specified.
    """

    if not transform_parameter_file:
        if not transform_directory:
            raise ValueError('Neither the alignment directory nor the transformation parameter file is specified!')
        transform_parameter_dir = transform_directory
        transform_parameter_file = transform_file(transform_parameter_dir)
    else:
        transform_parameter_dir = os.path.split(transform_parameter_file)
        transform_parameter_dir = transform_parameter_dir[0]
        transform_parameter_file = transform_parameter_file

    return transform_parameter_dir, transform_parameter_file


def set_path_transform_files(result_directory):
    """Replaces relative with absolute path in the parameter files in the result directory.

    Arguments
    ---------
    result_directory : str
      Path to directory of elastix results.

    Notes
    -----
    When elastix is not run in the directory of the transformation files
    the absolute path needs to be given in each transformation file
    to point to the subsequent transformation files. This is done via this
    routine.
    """

    files = os.listdir(result_directory)
    files = [x for x in files if re.match('TransformParameters.\d.txt', x)]
    files.sort()

    if not files:
        raise RuntimeError(f'Cannot find a valid transformation file in {result_directory}!')

    rec = re.compile("\(InitialTransformParametersFileName \"(?P<parname>.*)\"\)")

    for f in files:
        fh, tmpfn = tempfile.mkstemp()
        ff = os.path.join(result_directory, f)

        with open(tmpfn, 'w') as newfile, open(ff) as parfile:
            for line in parfile:
                # print line
                m = rec.match(line)
                if m is not None:
                    pn = m.group('parname')
                    if pn != 'NoInitialTransform':
                        pathn, filen = os.path.split(pn)
                        filen = os.path.join(result_directory, filen)
                        newfile.write(line.replace(pn, filen))
                    else:
                        newfile.write(line)
                else:
                    newfile.write(line)

        os.close(fh)
        os.remove(ff)
        shutil.move(tmpfn, ff)


def set_metric_parameter_file(parameter_file_path, new_metric):
    """Replaces the metric in the parameter file.

    Arguments
    ---------
    parameter_file_path : str
      The parameter file name.
    new_metric : str
      The metric to use.

    Notes
    -----
    Used to replace the metric when inverse transform is estimated.
    """
    metric_pattern = re.compile(r'\(Metric "(?P<metric_name>.*)"\)')
    metric_set = False

    try:
        with tempfile.NamedTemporaryFile('w', delete=False) as temp_file:
            temp_file_path = temp_file.name
            with open(parameter_file_path, 'r') as parameter_file:
                for line in parameter_file:
                    match = metric_pattern.match(line)
                    if match:
                        current_metric = match.group('metric_name')
                        temp_file.write(line.replace(current_metric, new_metric))
                        metric_set = True
                    else:
                        temp_file.write(line)
            if not metric_set:
                temp_file.write(f'(Metric "{new_metric}")\n')

        shutil.move(temp_file_path, parameter_file_path)

    except Exception as error:
        print(f"Error setting metric in parameter file: {error}")
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        raise error


def result_data_file(result_directory):
    """Returns the mhd result file in a result directory.

    Arguments
    ---------
    result_directory : str
      Path to elastix result directory.

    Returns
    -------
    result_file : str
      The mhd file in the result directory.
    """
    files = os.listdir(result_directory)
    files = [x for x in files if re.match('.*.mhd', x)]
    files.sort()

    if not files:
        raise RuntimeError('Cannot find a valid result data file in ' + result_directory)

    return os.path.join(result_directory, files[0])


def transform_file_size_and_spacing(transform_file):
    """Parse the image size and spacing from a transformation parameter file.

    Arguments
    ---------
    transform_file : str
      File name of the transformix parameter file.

    Returns
    -------
    size: tuple
      The image size.
    spacing : tuple
      The image spacing.
    """
    resi = re.compile("\(Size (?P<size>.*)\)")
    resp = re.compile("\(Spacing (?P<spacing>.*)\)")

    si = None
    sp = None
    with open(transform_file) as parfile:
        for line in parfile:
            m = resi.match(line)
            if m is not None:
                pn = m.group('size')
                si = pn.split()

            m = resp.match(line)
            if m is not None:
                pn = m.group('spacing')
                sp = pn.split()

            parfile.close()

    si = tuple(float(x) for x in si)
    sp = tuple(float(x) for x in sp)

    return si, sp


def set_transform_file_size_and_spacing(transform_file, size, spacing):
    """Replaces size and scale in the transformation parameter file

    Arguments
    ---------
    transform_file : str
      Transformation parameter file.
    size : tuple
      The new image size.
    spacing : tuple
      The new image spacing.
    """

    resi = re.compile("\(Size (?P<size>.*)\)")
    resp = re.compile("\(Spacing (?P<spacing>.*)\)")

    fh, tmpfn = tempfile.mkstemp()

    si = tuple(int(x) for x in size)

    with open(transform_file) as parfile, open(tmpfn, 'w') as newfile:
        for line in parfile:
            m = resi.match(line)
            if m is not None:
                newfile.write("(Size %d %d %d)" % si)
            else:
                m = resp.match(line)
                if m is not None:
                    newfile.write("(Spacing %d %d %d)" % spacing)
                else:
                    newfile.write(line)

    os.remove(transform_file)
    shutil.move(tmpfn, transform_file)


def rescale_size_and_spacing(size, spacing, scale):
    """Rescales the size and spacing

    Arguments
    ---------
    size : tuple
      The image size.
    spacing : tuple
      The image spacing.
    scale : tuple, number
      The scale factor.

    Returns
    -------
    size : tuple
      The scaled image size.
    spacing : tuple
      The scaled image spacing.
    """

    if not isinstance(scale, (tuple, list)):
      scale = (scale,) * len(size)

    si = tuple(int(x * s) for x, s in zip(size, scale))
    sp = tuple(sp / sc for sp, sc in zip(spacing, scale))

    return si, sp


##############################################################################
#  Elastix Runs
##############################################################################

def align_from_dict(align_parameters, landmarks_files, landmarks_weights=None):
    use_landmarks = all([str(f) and os.path.exists(f) for f in landmarks_files.values()])
    if use_landmarks:
        align_parameters.update(itk_mode=True, **landmarks_files)
        tmp_registration_params = {p: {
            'Registration': ['MultiMetricMultiResolutionRegistration'],
            'Metric': ["AdvancedMattesMutualInformation", "CorrespondingPointsEuclideanDistanceMetric"],
            'Metric0Weight': 1,
            'Metric1Weight': landmarks_weights[i],
        } for i, p in enumerate(align_parameters['parameter_files'])}  # FIXME: maybe not for all steps
    else:
        tmp_registration_params = {p: {} for p in align_parameters['parameter_files']}
    for k, v in align_parameters.items():
        if not v:
            raise ValueError(f'Registration missing parameter "{k}"')
    # TODO: check which files to patch
    with ElastixConfigPatcher(align_parameters['parameter_files'], tmp_registration_params):
        align(**align_parameters)  # WARNING: if using itk_mode, should use it to patch the files too


def align(fixed_image, moving_image, affine_parameter_file=None, bspline_parameter_file=None,
          result_directory=None, processes=None,
          workspace=None, moving_landmarks_path=None, fixed_landmarks_path=None,
          parameter_files=None,
          check_alignment_success=False, itk_mode=False):
    """
    Align images using elastix, estimates a transformation :math:`T:` fixed image :math:`\\rightarrow` moving image.

    Arguments
    ---------
    fixed_image : str
        Image source of the fixed image (typically the reference image).
    moving_image : str
        Image source of the moving image (typically the image to be registered).
    affine_parameter_file : str or None
        Elastix parameter file for the primary affine transformation.
    bspline_parameter_file : str or None
        Elastix parameter file for the secondary non-linear transformation.
    result_directory : str or None
        Elastic result directory.
    processes : int or None
        Number of threads to use.
    workspace : Workspace
        The workspace passed to the function to cath the executor to cancel the execution of the computation
    moving_landmarks_path : str
        Path to the moving landmarks file.
    fixed_landmarks_path : str
        Path to the fixed landmarks file.
    check_alignment_success : bool
        Check if the alignment was successful.

    Returns
    -------
    result_directory : str
      Path to elastix result directory.
    """

    if parameter_files is None:
        parameter_files = [affine_parameter_file, bspline_parameter_file]
        parameter_files = [f for f in parameter_files if f is not None]
    else:
        if affine_parameter_file or bspline_parameter_file:
            raise ValueError('Cannot specify both parameter_files and affine_parameter_file or bspline_parameter_file')

    processes = processes if processes is not None else mp.cpu_count()

    check_elastix_initialized()

    # result directory
    result_directory = result_directory if result_directory is not None else tempfile.gettempdir()

    if not os.path.exists(result_directory):
        os.mkdir(result_directory)

    if itk_mode:
        if not itk_imported:
            try:
                import itk
                dir(itk)  # WARNING: itk uses lazy loading
                from itk import ParameterObject
            except ImportError as err:
                raise ImportError(f'Could not import itk, please select itk_mode=False or install itk. '
                                  f'Details: {err}')
        from itk import ParameterObject
        from ClearMap.Alignment.landmarks_registration.engine import AlignmentTool
        from ClearMap.Alignment.landmarks_registration.registration_data import ITKImage
        tool = AlignmentTool(str(result_directory),
                             fixed_image=ITKImage(fixed_image),
                             moving_image=ITKImage(moving_image),
                             fixed_landmarks_name=fixed_landmarks_path.name,
                             moving_landmarks_name=moving_landmarks_path.name)
        for param_file in parameter_files:  # TODO: use option to combine the different transforms if no intermediate results are needed
            elx_params = ParameterObject.New()
            elx_params.AddParameterFile(str(param_file))
            tool.perform_registration(parameters=elx_params, debug=False, delete_tmp=False)
    else:
        check_spaces(*parameter_files, fixed_image, moving_image,
                     moving_landmarks_path, fixed_landmarks_path, result_directory)

        # run elastix
        cmd = [elastix_binary, '-threads', str(processes), '-m', f'{moving_image}', '-f', f'{fixed_image}']
        for parameter_file in parameter_files:
            cmd.extend(['-p', f'{parameter_file}'])
        if moving_landmarks_path is not None or fixed_landmarks_path is not None:
            cmd.extend(['-mp', f'{moving_landmarks_path}', '-fp', f'{fixed_landmarks_path}'])
        cmd.extend(['-out', f'{result_directory}'])

        try:
            # Check if sys.stdout has a valid file descriptor
            if hasattr(sys.stdout, 'fileno') and callable(sys.stdout.fileno):
                with subprocess.Popen(cmd, stdout=sys.stdout, stderr=sys.stdout) as proc:
                    if workspace is not None:
                        workspace.process = proc
            else:
                # If sys.stdout does not have a valid file descriptor, redirect output to subprocess.PIPE
                # This could happen e.g. in an IDE
                with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE) as proc:
                    if workspace is not None:
                        workspace.process = proc
        except UnsupportedOperation:
            try:
                with subprocess.Popen(cmd) as proc:
                    if workspace is not None:
                        workspace.process = proc
            except (subprocess.SubprocessError, OSError) as err:
                raise ClearMapException(f'Align: failed executing: {" ".join(cmd)}') from err
        except (subprocess.SubprocessError, OSError) as err:
            raise ClearMapException(f'Align: failed executing: {" ".join(cmd)}') from err
        finally:
            if workspace is not None:
                workspace.process = None

    if not itk_mode and check_alignment_success:
        check_success(result_directory)

    return result_directory


def check_spaces(*paths):
    for p in paths:
        if p is not None and ' ' in str(p):
            raise ValueError(f'Could not run elastix with path containing spaces: {p}')


def transform(source, sink='transformix', transform_parameter_file=None, transform_directory=None,
              result_directory=None):
    """Transform a raw data set to reference using the elastix alignment results.

    Arguments
    ---------
    source : str or array
      Image source to be transformed.
    sink : str or None
      Image sink to save transformed image to. If 'transformix' return the default name
      of the data file generated by transformix.
    transform_parameter_file : str or None
      Parameter file for the primary transformation.
      If None, the file is determined from the transform_directory.
    transform_directory : str or None
      Result directory of elastix alignment.
      If None the transform_parameter_file has to be given.
    result_directory : str or None
      The directory for the transformix results.

    Returns
    -------
    transformed : array or st
      Array or file name of the transformed data.

    Note
    ----
    If the map determined by elastix is
    :math:`T: \\mathrm{fixed} \\rightarrow \\mathrm{moving}`,
    transformix on data works as :math:`T^{-1}(\\mathrm{data})`.
    """
    check_elastix_initialized()

    # image
    source = io.as_source(source)
    if isinstance(source, io.tif.Source):
        img_name = source.location
        delete_image = None
    else:
        img_name = os.path.join(tempfile.gettempdir(), 'elastix_input.tif')
        io.write(img_name, source)
        delete_image = img_name

    # result directory
    delete_result_directory = None
    if result_directory is None:
        result_dirname = os.path.join(tempfile.gettempdir(), 'elastix_output')
        delete_result_directory = result_dirname
    else:
        result_dirname = result_directory

    if not os.path.exists(result_dirname):
        os.makedirs(result_dirname)

    # transformation parameter
    transform_parameter_dir, transform_parameter_file = transform_directory_and_file(
        transform_parameter_file=transform_parameter_file, transform_directory=transform_directory)

    set_path_transform_files(transform_parameter_dir)

    # transformix -in inputImage.ext -out outputDirectory -tp TransformParameters.txx
    check_spaces(img_name, result_dirname, transform_parameter_file)
    cmd = f'{transformix_binary} -in {img_name} -out {result_dirname} -tp {transform_parameter_file}'

    res = os.system(cmd)

    if res != 0:
        raise RuntimeError('transform_data: failed executing: ' + cmd)

    # read data and clean up
    if delete_image is not None:
        os.remove(delete_image)

    if sink == 'transformix':
        return result_data_file(result_dirname)
    elif sink is None:
        result_file = result_data_file(result_dirname)
        result = io.read(result_file)
    elif isinstance(sink, str):
        result_file = result_data_file(result_dirname)
        result = io.convert(result_file, sink)
    else:
        raise RuntimeError('transform_data: sink not valid!')

    if delete_result_directory is not None:
        shutil.rmtree(delete_result_directory)

    return result


def deformation_field(sink='transformix', transform_parameter_file=None, transform_directory=None, result_directory=None):
    """Create the deformation field T(x) - x.

    Arguments
    ---------
    sink : str or None
      Image sink to save the transformation field; if 'transformix' return the default name
      of the data file generated by transformix.
    transform_parameter_file : str or None
      Parameter file for the primary transformation, if None, the file is
      determined from the transform_directory.
    transform_directory : str or None
      Result directory of elastix alignment, if None the
      transform_parameter_file has to be given.
    result_directory : str or None
      The directory for the transformix results.

    Returns
    -------
    deformation_field : array or str
      Array or file name of the deformation field data.

    Note
    ----
    The map determined by elastix is
    :math:`T \\mathrm{fixed} \\rightarrow \\mathrm{moving}`.
    """
    check_elastix_initialized()

    # result directory
    delete_result_directory = None
    if result_directory is None:
        result_dirname = os.path.join(tempfile.gettempdir(), 'elastix_output')
        delete_result_directory = result_dirname
    else:
        result_dirname = result_directory

    if not os.path.exists(result_dirname):
        os.makedirs(result_dirname)

    # setup transformation
    transform_parameter_dir, transform_parameter_file = transform_directory_and_file(
        transform_parameter_file=transform_parameter_file, transform_directory=transform_directory)
    set_path_transform_files(transform_parameter_dir)

    # transformix -in inputImage.ext -out outputDirectory -tp TransformParameters.txt
    check_spaces(result_dirname, transform_parameter_file)
    cmd = f'{transformix_binary} -def all -out {result_dirname} -tp {transform_parameter_file}'

    res = os.system(cmd)

    if res != 0:
        raise RuntimeError(f'deformation_field: failed executing: {cmd}')

    # read result and clean up
    if sink == 'transformix':
        return result_data_file(result_dirname)
    elif sink is None:
        result_file = result_data_file(result_dirname)
        result = io.read(result_file)
    elif isinstance(sink, str):
        result_file = result_data_file(result_dirname)
        result = io.convert(result_file, sink)
    else:
        raise RuntimeError('deformation_field: sink not valid!')

    if delete_result_directory is not None:
        shutil.rmtree(delete_result_directory)

    return result


def deformation_distance(deformation_field, sink=None, scale=None):
    """Compute the distance field from a deformation vector field.

    Arguments
    ---------
    deformation_field : str or array
      Source of the deformation field determined by :func:`deformation_field`.
    sink : str or None
      Image sink to save the deformation field to.
    scale : tuple or None
      Scale factor for each dimension, if None = (1,1,1).

    Returns
    -------
    deformation_distannce : array or st
      Array or file name of the deformation distance data.
    """

    deformation_field = io.read(deformation_field)

    df = np.square(deformation_field)
    if scale is not None:
        for i in range(3):
            df[:, :, :, i] = df[:, :, :, i] * (scale[i] * scale[i])
    df = np.sqrt(np.sum(df, axis=3))

    return io.write(sink, df)


###############################################################################
# Point transformations
###############################################################################

def write_points(filename, points, indices=False, binary=True):
    """Write points as elastix/transformix point file

    Arguments
    ---------
    filename : str
      File name of the elastix point file.
    points : array or str
      Source of the points.
    indices : bool
      Write as pixel indices or physical coordinates.

    Returns
    -------
    filename: str
      File name of the elastix point file.
    """

    points = io.read(points)

    if binary:
        with open(filename, 'wb') as point_file:
            if get_elastix_version() < 5:
                dtype = np.int64
            else:
                dtype = np.uint64
            if indices:
                np.array(1, dtype=dtype).tofile(point_file)
            else:
                np.array(0, dtype=dtype).tofile(point_file)

            num_points = np.array(len(points), dtype=np.int64)
            num_points.tofile(point_file)

            points = np.asarray(points, dtype=np.double)
            points.tofile(point_file)
    else:
        with open(filename, 'w') as point_file:
            if indices:
                point_file.write('index\n')
            else:
                point_file.write('point\n')

            point_file.write(f'{points.shape[0]}\n')
            np.savetxt(point_file, points, delimiter=' ', newline='\n', fmt='%.5e')

    return filename


def read_points(filename, indices=False, binary=True):
    """Parses the output points from the output file of transformix

    Arguments
    ---------
    filename : str
      File name of the transformix output file.
    indices : bool
      If True return pixel indices otherwise float coordinates.

    Returns
    --------
    points : array
      The point coordinates.
    """

    if binary:
        with open(filename) as f:
            if get_elastix_version() < 5:
                dtype = np.int64
            else:
                dtype = np.uint64
            _ = np.fromfile(f, dtype=dtype, count=1)[0]  # Discard the index flag
            num_points = np.fromfile(f, dtype=dtype, count=1)[0]
            if num_points == 0:
                points = np.zeros((0, 3))
            else:
                points = np.fromfile(f, dtype=np.double)
                points = np.reshape(points, (num_points, 3))
    else:  # text file
        with open(filename) as f:
            lines = f.readlines()

        num_points = len(lines)
        if num_points == 0:
            points = np.zeros((0, 3))
        else:
            points = np.zeros((num_points, 3))
            for k, line in enumerate(lines):
                ls = line.split()
                if indices:
                    for i in range(0, 3):
                        points[k, i] = float(ls[i + 22])
                else:
                    for i in range(0, 3):
                        points[k, i] = float(ls[i + 30])
    return points


def transform_points(source, sink=None, transform_parameter_file=None, transform_directory=None, indices=False,
                     result_directory=None, temp_file=None, binary=True):
    """Transform coordinates math:`x` via elastix estimated transformation to :math:`T(x)`.

    Arguments
    ---------
    source : str
      Source of the points.
    sink : str or None
      Sink for transformed points.
    transform_parameter_file : str or None
      Parameter file for the primary transformation.
      If None, the file is determined from the transform_directory.
    transform_directory : str or None
      Result directory of elastix alignment.
      If None the transform_parameter_file has to be given.
    indices : bool
      If True use points as pixel coordinates otherwise spatial coordinates.
    result_directory : str or None
      Elastic result directory.
    temp_file : str or None
      Optional file name for the elastix point file.

    Returns
    -------
    points : array or st
      Array or file name of transformed points.

    Note
    ----
    The transformation is from the fixed image coordinates to the moving image coordinates.
    """
    check_elastix_initialized()

    # input point file
    if temp_file is None:
        extension = "bin" if binary else "txt"
        temp_file = os.path.join(tempfile.gettempdir(), f'elastix_input.{extension}')

    delete_point_file = None
    if isinstance(source, str):
        if len(source) > 3 and source[-3:] in ['txt', 'bin']:
            binary = source[-3:] == 'bin'
            point_file = source
        else:
            points = io.read(source)
            point_file = temp_file
            delete_point_file = temp_file
            write_points(point_file, points, indices=indices, binary=binary)
    elif isinstance(source, np.ndarray):
        point_file = temp_file
        delete_point_file = temp_file
        write_points(point_file, source, indices=indices, binary=binary)
    else:
        raise RuntimeError(f'transform_points: source not string or array but "{type(source)}"!')

    # result directory
    if result_directory is None:
        out_dirname = os.path.join(tempfile.gettempdir(), 'elastix_output')
        delete_result_directory = out_dirname
    else:
        out_dirname = result_directory
        delete_result_directory = None

    if not os.path.exists(out_dirname):
        os.makedirs(out_dirname)

    # transform
    transform_parameter_dir, transform_parameter_file = transform_directory_and_file(
        transform_parameter_file=transform_parameter_file, transform_directory=transform_directory)
    set_path_transform_files(transform_parameter_dir)

    # run transformix
    check_spaces(point_file, out_dirname, transform_parameter_file)
    cmd = f'{transformix_binary} -def {point_file} -out {out_dirname} -tp {transform_parameter_file}'
    print(cmd)

    res = os.system(cmd)

    if res != 0:
        raise RuntimeError(f'failed executing {cmd}, return code: {res}')

    # read data and clean up
    if delete_point_file is not None:
        os.remove(delete_point_file)

    # read data / file
    extension = 'bin' if binary else 'txt'
    output_file = f'outputpoints.{extension}'
    if sink == []:  # return sink as file name
        return os.path.join(out_dirname, output_file)
    else:
        transformed_points = read_points(os.path.join(out_dirname, output_file), indices=indices, binary=binary)

    if delete_result_directory:
        shutil.rmtree(delete_result_directory)

    return io.write(sink, transformed_points)


def inverse_transform(fixed_image, affine_parameter_file, bspline_parameter_file=None, transform_parameter_file=None,
                      transform_directory=None, result_directory=None, processes=None):
    """Estimate inverse transformation :math:`T^{-1}:` moving image :math:`\\rightarrow` fixed image.

    Arguments
    ---------
    fixed_image : str
      Image source of the fixed image (typically the reference image)
    affine_parameter_file : str
      The parameter file for the original affine transformation.
    bspline_parameter_file : str
      The parameter file for the original b-spline transformation.
    transform_directory : str or None
      Elastic result directory of the original transform.
    transform_parameter_file : str or None
      Elastic transformation parameter file of the original transform.
    result_directory : str or None
      Elastic result directory of the inverse transform.
    processes : int or None
      Number of processes to use.

    Returns
    -------
    result_directory : str
      Path to elastix result directory.
    """
    check_elastix_initialized()

    result_directory = result_directory or tempfile.gettempdir()

    if not os.path.exists(result_directory):
        os.mkdir(result_directory)

    # transformation files
    transform_parameter_dir, transform_parameter_file = transform_directory_and_file(
        transform_parameter_file=transform_parameter_file, transform_directory=transform_directory)
    set_path_transform_files(transform_parameter_dir)

    # set metric of the parameter files
    parameter_files = {
        'bspline': bspline_parameter_file,
        'affine': affine_parameter_file
    }
    for key, param_file in parameter_files.items():
        if param_file:
            param_file_path = Path(param_file)
            dest_file = Path(result_directory) / param_file_path.name
            shutil.copyfile(param_file_path, dest_file)
            set_metric_parameter_file(str(dest_file), 'DisplacementMagnitudePenalty')
        else:
            parameter_files[key] = None

    check_spaces(fixed_image, result_directory, transform_parameter_file, *parameter_files.values())

    # run elastix
    # The same image is used for fixed and moving image as per elastix documentation
    # 6.1.6 `DisplacementMagnitudePenalty: inverting transformations`
    # The DisplacementMagnitudePenalty is a cost function that penalises ||Tµ(x) − x||2. You can use this
    # to invert transforms, by setting the transform to be inverted as an initial transform (using -t0), setting
    # (HowToCombineTransforms "Compose"), and running elastix with this metric, using the original fixed
    # image set both as fixed (-f) and moving (-m) image.
    cmd = f'{elastix_binary} -threads {processes} -m {fixed_image} -f {fixed_image} -t0 {transform_parameter_file} '
    for k, v in parameter_files:
        if v:
            cmd += f'-p {v} '
    cmd += f'-out {result_directory}'

    res = os.system(cmd)

    if res != 0:
        raise RuntimeError(f'inverse_transform: failed executing: {cmd}')

    return result_directory


###############################################################################
# Transformation interface
###############################################################################

class ElastixTransformation(TransformationBase):
    """Elastix Transformation

    Note
    ----
    The transformation represented is :math:`T` as estimated by elastix.
    For data transformix applies :math:`T^{-1}` and for points :math:`T`
    """

    ttype = 'Elastix'

    def __init__(self, transform_directory=None, transform_parameter_file=None, inverse=False):
        super().__init__(inverse=inverse)

        transform_directory, transform_parameter_file = transform_directory_and_file(
            transform_parameter_file=transform_parameter_file, transform_directory=transform_directory)

        self.transform_directory = transform_directory
        self.transform_parameter_file = transform_parameter_file

    def transform_data(self, source, inverse=False, **kwargs):
        inverse = self.get_inverse(inverse)

        # Note: transformix maps atlas -> data via T^-1
        if not inverse:
            raise NotImplementedError()
        else:
            return transform(source, transform_parameter_file=self.transform_parameter_file,
                             transform_directory=self.transform_directory, **kwargs)

    def transform_points(self, source, inverse=False, **kwargs):
        inverse = self.get_inverse(inverse)

        if not inverse:
            return transform_points(source, transform_parameter_file=self.transform_parameter_file,
                                    transform_directory=self.transform_directory, **kwargs)
        else:
            raise NotImplementedError()

    def to_dict(self) -> dict:
        dictionary = super().to_dict()
        dictionary.update(transform_directory=self.transform_directory,
                          transform_parameter_file=self.transform_parameter_file)
        return dictionary

    def __repr__(self):
        return '%s[%s]' % (super().__repr__(), self.transform_directory)


###############################################################################
# Tests
###############################################################################


def _test():
    import os
    import numpy as np
    import ClearMap.Settings as settings
    import ClearMap.Alignment.Elastix as elx
    # reload(elx)

    path = os.path.join(settings.test_data_path, 'Elastix')

    result_directory = os.path.join(path, 'elastix_template_to_ref')

    elx.transform_file(result_directory)

    shape = np.array([432, 512, 229])
    points = np.random.rand(30, 3) * 0.25 * shape + 0.5 * shape
    points = np.round(points)
    points = np.array(points, dtype=int)

    test_txt = elx.transform_points(points, transform_directory=result_directory, binary=False, indices=False)

    test = elx.transform_points(points, transform_directory=result_directory, binary=True, indices=False)

    print(np.allclose(test_txt, test))
