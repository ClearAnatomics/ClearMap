# -*- coding: utf-8 -*-
"""
params
======

All the classes that define parameters or group thereof for the tabs of the graphical interface
"""
import os
import string
from itertools import permutations
from typing import List

import numpy as np

from ClearMap.config.atlas import ATLAS_NAMES_MAP
from ClearMap.gui.gui_utils import create_clearmap_widget

from ClearMap.gui.dialogs import get_directory_dlg
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import QInputDialog, QToolBox, QCheckBox

from ClearMap.config.config_loader import ConfigLoader


__author__ = 'Charly Rousseau <charly.rousseau@icm-institute.org>'
__license__ = 'GPLv3 - GNU General Public License v3 (see LICENSE.txt)'
__copyright__ = 'Copyright © 2022 by Charly Rousseau'
__webpage__ = 'https://idisco.info'
__download__ = 'https://www.github.com/ChristophKirst/ClearMap2'

from ClearMap.gui.params_interfaces import ParamLink, UiParameter, UiParameterCollection


class ParamsOrientationError(ValueError):
    pass


class AlignmentParams(UiParameterCollection):
    def __init__(self, tab, src_folder=None):
        super().__init__(tab, src_folder)
        self.stitching_general = GeneralStitchingParams(tab, src_folder)
        self.stitching_rigid = RigidStitchingParams(tab, src_folder)
        self.stitching_wobbly = WobblyStitchingParams(tab, src_folder)
        self.registration = RegistrationParams(tab, src_folder)

    def fix_cfg_file(self, f_path):
        # cfg = ConfigLoader.get_cfg_from_path(f_path)
        pipeline_name, ok = QInputDialog.getItem(self.tab, 'Please select pipeline type',
                                                 'Pipeline name:', ['CellMap', 'TubeMap', 'Both'], 0, False)
        if not ok:
            raise ValueError('Missing sample ID')
        self.config['pipeline_name'] = pipeline_name
        # WARNING: needs to be self.config
        #  to be sure that we are up to date
        #  (otherwise write but potentially no reload)
        self.write_config()

    @property
    def params(self):
        return self.stitching_general, self.stitching_rigid, self.stitching_wobbly, self.registration

    @property
    def all_stitching_params(self):
        return self.stitching_general, self.stitching_rigid, self.stitching_wobbly

    @property
    def pipeline_name(self):
        return self.config['pipeline_name']

    @pipeline_name.setter
    def pipeline_name(self, name):
        self.config['pipeline_name'] = name

    # @property
    # def pipeline_is_cell_map(self):
    #     return self.pipeline_name.lower().replace('_', '') == 'cellmap'
    #
    # @property
    # def pipeline_is_tube_map(self):
    #     return self.pipeline_name.lower().replace('_', '') == 'tubemap'
    #
    # @property
    # def pipeline_is_both(self):
    #     return self.pipeline_name.lower().replace('_', '') == 'both'


class SampleParameters(UiParameter):
    sample_id: str
    use_id_as_prefix: bool
    tile_extension: str
    raw_path: str
    autofluo_path: str
    arteries_path: str
    raw_resolution: List[float]
    arteries_resolution: List[float]
    autofluorescence_resolution: List[float]
    slice_x: List[int]
    slice_y: List[int]
    slice_z: List[int]
    orientation: List[int]

    def __init__(self, tab, src_folder=None):
        super().__init__(tab, src_folder)
        self.params_dict = {
            'sample_id': ['sample_id'],
            'use_id_as_prefix': ParamLink(['use_id_as_prefix'], self.tab.useIdAsPrefixCheckBox),
            'tile_extension': ParamLink(['src_paths', 'tile_extension'], self.tab.tileExtensionLineEdit),
            'raw_path': ['src_paths', 'raw'],
            'autofluo_path': ParamLink(['src_paths', 'autofluorescence'], self.tab.autofluoPathOptionalPlainTextEdit),
            'arteries_path': ParamLink(['src_paths', 'arteries'], self.tab.arteriesPathOptionalPlainTextEdit),
            'raw_resolution': ParamLink(['resolutions', 'raw'], self.tab.rawResolutionTriplet),
            'arteries_resolution': ParamLink(['resolutions', 'arteries'], self.tab.arteriesResolutionTriplet),
            'autofluorescence_resolution': ParamLink(['resolutions', 'autofluorescence'], self.tab.autofluorescenceResolutionTriplet),
            'slice_x': ParamLink(['slice_x'], self.tab.sliceXDoublet),
            'slice_y': ParamLink(['slice_y'], self.tab.sliceYDoublet),
            'slice_z': ParamLink(['slice_z'], self.tab.sliceZDoublet),
            'orientation': ['orientation']  # WARNING: Finish by orientation in case invalid,
        }
        self.connect()
        if self.sample_id:
            self.handle_sample_id_changed(self.sample_id)

    def connect(self):
        self.tab.sampleIdTxt.editingFinished.connect(self.handle_sample_id_changed)

        self.tab.rawPath.textChanged.connect(self.handle_raw_path_changed)

        self.tab.orient_x.currentTextChanged.connect(self.handle_orientation_changed)
        self.tab.orient_y.currentTextChanged.connect(self.handle_orientation_changed)
        self.tab.orient_z.currentTextChanged.connect(self.handle_orientation_changed)

        self.connect_simple_widgets()

    def _ui_to_cfg(self):
        self._config['base_directory'] = self.src_folder

    def cfg_to_ui(self):
        self.reload()
        super().cfg_to_ui()

    def fix_cfg_file(self, f_path):  # FIXME: seems wrong to pass f_path just for that usage
        # cfg = ConfigLoader.get_cfg_from_path(f_path)
        self.config['base_directory'] = os.path.dirname(f_path)  # WARNING: needs to be self.config
                                                                 #  to be sure that we are up to date
                                                                 #  (otherwise write but potentially no reload)
        if not self.sample_id:
            sample_id, ok = QInputDialog.getText(self.tab, 'Warning: missing ID',
                                                 '<b>Missing sample ID</b><br>Please input below')
            self.sample_id = sample_id
            if not ok:
                raise ValueError('Missing sample ID')
        self.config['sample_id'] = self.sample_id
        self.config['use_id_as_prefix'] = self.use_id_as_prefix
        self.config.write()

    # Sample params
    @property
    def sample_id(self):
        return self.tab.sampleIdTxt.text()

    @sample_id.setter
    def sample_id(self, id_):
        self.tab.sampleIdTxt.setText(id_)

    def handle_sample_id_changed(self, id_=None):
        if self.config is not None:
            self.config['sample_id'] = self.sample_id
            self.ui_to_cfg()   # FIXME: check

    @property
    def raw_path(self):
        f_path = self.tab.rawPath.toPlainText()
        f_path = f_path if f_path else None
        return f_path

    @raw_path.setter
    def raw_path(self, f_path):
        f_path = f_path if f_path is not None else ''
        self.tab.rawPath.setPlainText(f_path)

    def handle_raw_path_changed(self):
        self.config['src_paths']['raw'] = self.raw_path

    @property
    def orientation(self):
        x = int(self.tab.orient_x.currentText())
        y = int(self.tab.orient_y.currentText())
        z = int(self.tab.orient_z.currentText())
        n_axes = len(set([abs(e) for e in (x, y, z)]))
        if n_axes != 3:
            raise ParamsOrientationError('Number of different axis is only {} instead of 3. '
                                         'Please amend duplicate axes'.format(n_axes))
        return x, y, z

    @orientation.setter
    def orientation(self, orientation):
        n_axes = len(set([abs(e) for e in orientation]))
        if n_axes != 3:
            raise ParamsOrientationError('Number of different axis is only {} instead of 3. '
                                         'Please amend duplicate axes'.format(n_axes))
        self.tab.orient_x.setCurrentText(f'{orientation[0]}')
        self.tab.orient_y.setCurrentText(f'{orientation[1]}')
        self.tab.orient_z.setCurrentText(f'{orientation[2]}')

    def handle_orientation_changed(self, val):  # WARNING: does not seem to move up the stack because of pyqtsignals
        try:
            orientation = self.orientation
        except ParamsOrientationError as err:
            print('Invalid orientation, keeping current')
            return
        self._config['orientation'] = orientation


class RigidStitchingParams(UiParameter):
    skip: bool
    x_overlap: int
    y_overlap: int
    projection_thickness: List[int]
    max_shifts_x: List[int]
    max_shifts_y: List[int]
    max_shifts_z: List[int]
    background_level: int
    background_pixels: int

    def __init__(self, tab, src_folder=None):
        super().__init__(tab, src_folder)
        self.params_dict = {
            'skip': ParamLink(['skip'], self.tab.skipRigidCheckbox),
            'x_overlap': ParamLink(['overlap_x'], self.tab.xOverlapSinglet),
            'y_overlap': ParamLink(['overlap_y'], self.tab.yOverlapSinglet),
            'projection_thickness': ['project_thickness'],  # FIXME: change to projection
            'max_shifts_x': ParamLink(['max_shifts_x'], self.tab.rigidMaxShiftsXDoublet),
            'max_shifts_y': ParamLink(['max_shifts_y'], self.tab.rigidMaxShiftsYDoublet),
            'max_shifts_z': ParamLink(['max_shifts_z'], self.tab.rigidMaxShiftsZDoublet),
            'background_level': ParamLink(['background_level'], self.tab.rigidBackgroundLevel),
            'background_pixels': ParamLink(['background_pixels'], self.tab.rigidBackgroundPixels)
        }
        self.cfg_subtree = ['stitching', 'rigid']
        self.connect()

    def connect(self):
        self.tab.projectionThicknessDoublet.valueChangedConnect(self.handle_projection_thickness_changed)
        self.connect_simple_widgets()

    @property
    def projection_thickness(self):
        val = self.tab.projectionThicknessDoublet.getValue()
        if val is not None:
            val.append(None)
        return val

    @projection_thickness.setter
    def projection_thickness(self, thickness):
        if thickness is not None:
            thickness = thickness[:2]
        self.tab.projectionThicknessDoublet.setValue(thickness)

    def handle_projection_thickness_changed(self):
        self.config['project_thickness'] = self.projection_thickness


class WobblyStitchingParams(UiParameter):
    skip: bool
    max_shifts_x: List[int]
    max_shifts_y: List[int]
    max_shifts_z: List[int]
    valid_range: list  # FIXME: missing pixel size
    slice_range: List[int]
    slice_pixel_size: int

    def __init__(self, tab, src_folder=None):
        super().__init__(tab, src_folder)
        self.params_dict = {
            'skip': ParamLink(['skip'], self.tab.skipWobblyCheckBox),
            'max_shifts_x': ParamLink(['max_shifts_x'], self.tab.wobblyMaxShiftsXDoublet),
            'max_shifts_y': ParamLink(['max_shifts_y'], self.tab.wobblyMaxShiftsYDoublet),
            'max_shifts_z': ParamLink(['max_shifts_z'], self.tab.wobblyMaxShiftsZDoublet),
            'valid_range': ParamLink(['stack_valid_range'], self.tab.wobblyValidRangeDoublet),
            'slice_range': ParamLink(['slice_valid_range'], self.tab.wobblySliceRangeDoublet),
            'slice_pixel_size': ParamLink(['slice_pixel_size'], self.tab.wobblySlicePixelSizeSinglet)
        }
        self.cfg_subtree = ['stitching', 'wobbly']
        self.connect()

    def connect(self):
        self.connect_simple_widgets()


class GeneralStitchingParams(UiParameter):
    use_npy: bool
    run_raw: bool
    run_arteries: bool
    preview_raw: bool
    preview_arteries: bool
    convert_output: bool
    convert_raw: bool
    convert_arteries: bool
    conversion_fmt: str

    def __init__(self, tab, src_folder=None):
        super().__init__(tab, src_folder)
        self.params_dict = {
            'use_npy': ParamLink(['conversion', 'use_npy'], self.tab.stitchingUseNpyCheckBox),
            'run_raw': ParamLink(['stitching', 'run', 'raw'], self.tab.stitchingRunRawCheckBox),
            'run_arteries': ParamLink(['stitching', 'run', 'arteries'], self.tab.stitchingRunArteriesCheckBox),
            'preview_raw': ParamLink(['stitching', 'preview', 'raw'], self.tab.stitchingPreviewRawCheckBox),
            'preview_arteries': ParamLink(['stitching', 'preview', 'arteries'], self.tab.stitchingPreviewArteriesCheckBox),
            'convert_output': ['stitching', 'output_conversion', 'skip'],
            'convert_raw': ParamLink(['stitching', 'output_conversion', 'raw'], self.tab.stitchingConvertRawCheckBox),
            'convert_arteries': ParamLink(['stitching', 'output_conversion', 'arteries'], self.tab.stitchingConvertArteriesCheckBox),
            'conversion_fmt': ParamLink(['stitching', 'output_conversion', 'format'], self.tab.outputConversionFormat)
        }
        self.attrs_to_invert = ['convert_output']  # FIXME: check
        self.connect()

    def connect(self):
        self.tab.skipOutputConversioncheckBox.stateChanged.connect(self.handle_convert_output_changed)
        self.connect_simple_widgets()

    @property
    def convert_output(self):
        return not self.tab.skipOutputConversioncheckBox.isChecked()

    @convert_output.setter
    def convert_output(self, skip):
        self.set_check_state(self.tab.skipOutputConversioncheckBox, not skip)

    def handle_convert_output_changed(self, state):
        self.config['stitching']['output_conversion']['skip'] = not self.convert_output


class RegistrationParams(UiParameter):
    atlas_id_changed = pyqtSignal(str)
    atlas_structure_tree_id_changed = pyqtSignal(str)

    skip_resampling: bool
    atlas_resolution: List[float]
    atlas_id: str
    structure_tree_id: str
    atlas_folder: str
    channel_affine_file_path: str
    ref_affine_file_path: str
    ref_bspline_file_path: str

    def __init__(self, tab, src_folder=None):
        super().__init__(tab, src_folder)
        self.params_dict = {
            'skip_resampling': ParamLink(['resampling', 'skip'], self.tab.skipRegistrationResamplingCheckBox),
            'atlas_resolution': ['resampling', 'raw_sink_resolution'],
            'atlas_id': ['atlas', 'id'],
            'structure_tree_id': ['atlas', 'structure_tree_id'],
            'atlas_folder': ParamLink(['atlas', 'align_files_folder'], self.tab.atlasFolderPath, connect=False),  # FIXME: ensure that set correctly by picking
            'channel_affine_file_path': ParamLink(['atlas', 'align_channels_affine_file'], self.tab.channelAffineFilePath),
            'ref_affine_file_path': ParamLink(['atlas', 'align_reference_affine_file'], self.tab.refAffineFilePath),
            'ref_bspline_file_path': ParamLink(['atlas', 'align_reference_bspline_file'], self.tab.refBsplineFilePath),
        }
        self.atlas_info = ATLAS_NAMES_MAP
        self.cfg_subtree = ['registration']
        self.connect()

    def connect(self):
        self.tab.atlasResolutionTriplet.valueChangedConnect(self.handle_atlas_resolution_changed)
        self.tab.atlasIdComboBox.currentTextChanged.connect(self.handle_atlas_id_changed)
        self.tab.structureTreeIdComboBox.currentTextChanged.connect(self.handle_structure_tree_id_changed)
        self.connect_simple_widgets()

    @property
    def atlas_base_name(self):
        return self.atlas_info[self.atlas_id]['base_name']
        
    @property
    def atlas_resolution(self):
        return self.tab.atlasResolutionTriplet.getValue()

    @atlas_resolution.setter
    def atlas_resolution(self, res):
        self.tab.atlasResolutionTriplet.setValue(res)

    def handle_atlas_resolution_changed(self, state):
        self.config['resampling']['raw_sink_resolution'] = self.atlas_resolution
        self.config['resampling']['autofluo_sink_resolution'] = self.atlas_resolution

    @property
    def atlas_id(self):
        return self.tab.atlasIdComboBox.currentText()

    @atlas_id.setter
    def atlas_id(self, value):
        self.tab.atlasIdComboBox.setCurrentText(value)

    def handle_atlas_id_changed(self):
        self.config['atlas']['id'] = self.atlas_id
        self.atlas_resolution = self.atlas_info[self.atlas_id]['resolution']
        self.ui_to_cfg()
        self.atlas_id_changed.emit(self.atlas_base_name)

    @property
    def structure_tree_id(self):
        return self.tab.structureTreeIdComboBox.currentText()

    @structure_tree_id.setter
    def structure_tree_id(self, value):
        self.tab.structureTreeIdComboBox.setCurrentText(value)

    def handle_structure_tree_id_changed(self):
        self.config['atlas']['structure_tree_id'] = self.structure_tree_id
        self.ui_to_cfg()   # TODO: check if required
        self.atlas_structure_tree_id_changed.emit(self.structure_tree_id)


class CellMapParams(UiParameter):
    background_correction_diameter: List[int]
    maxima_shape: int
    detection_threshold: int
    cell_filter_size: List[int]
    cell_filter_intensity: int
    voxelization_radii: List[int]
    plot_when_finished: bool
    plot_detected_cells: bool
    crop_x_min: int
    crop_x_max: int  # TODO: if 99.9 % source put to 100% (None)
    crop_y_min: int
    crop_y_max: int  # TODO: if 99.9 % source put to 100% (None)
    crop_z_min: int
    crop_z_max: int  # TODO: if 99.9 % source put to 100% (None)

    def __init__(self, tab, sample_params=None, preprocessing_params=None, src_folder=None):
        super().__init__(tab, src_folder)
        self.params_dict = {
            'background_correction_diameter': ['detection', 'background_correction', 'diameter'],
            'maxima_shape': ParamLink(['detection', 'maxima_detection', 'shape'], self.tab.maximaShape),
            'detection_threshold': ParamLink(['detection', 'shape_detection', 'threshold'], self.tab.detectionThreshold),
            'cell_filter_size': ParamLink(['cell_filtration', 'thresholds', 'size'], self.tab.cellFilterThresholdSizeDoublet),
            'cell_filter_intensity': ParamLink(['cell_filtration', 'thresholds', 'intensity'], self.tab.voxelizationRadiusTriplet),
            'voxelization_radii': ParamLink(['voxelization', 'radii'], self.tab.voxelizationRadiusTriplet),
            'plot_when_finished': ParamLink(['run', 'plot_when_finished'], self.tab.runCellMapPlotCheckBox),
            'plot_detected_cells': ParamLink(['detection', 'plot_cells'], self.tab.cellDetectionPlotCheckBox),
            'crop_x_min': ParamLink(['detection', 'test_set_slicing', 'dim_0', 0], self.tab.detectionSubsetXRangeMin),
            'crop_x_max': ParamLink(['detection', 'test_set_slicing', 'dim_0', 1], self.tab.detectionSubsetXRangeMax),
            'crop_y_min': ParamLink(['detection', 'test_set_slicing', 'dim_1', 0], self.tab.detectionSubsetYRangeMin),
            'crop_y_max': ParamLink(['detection', 'test_set_slicing', 'dim_1', 1], self.tab.detectionSubsetYRangeMax),
            'crop_z_min': ParamLink(['detection', 'test_set_slicing', 'dim_2', 0], self.tab.detectionSubsetZRangeMin),
            'crop_z_max': ParamLink(['detection', 'test_set_slicing', 'dim_2', 1], self.tab.detectionSubsetZRangeMax)
        }
        self.sample_params = sample_params
        self.preprocessing_params = preprocessing_params
        self.connect()

    def connect(self):
        self.tab.backgroundCorrectionDiameter.valueChanged.connect(self.handle_background_correction_diameter_changed)
        self.connect_simple_widgets()  # |TODO: automatise in parent class

    def cfg_to_ui(self):
        self.reload()
        super().cfg_to_ui()

    @property
    def ratios(self):
        raw_res = np.array(self.sample_params.raw_resolution)
        atlas_res = np.array(self.preprocessing_params.registration.atlas_resolution)
        ratios = atlas_res / raw_res  # to original
        return ratios

    @property
    def background_correction_diameter(self):
        return [self.tab.backgroundCorrectionDiameter.value()] * 2

    @background_correction_diameter.setter
    def background_correction_diameter(self, shape):
        if isinstance(shape, (list, tuple)):
            shape = shape[0]
        self.tab.backgroundCorrectionDiameter.setValue(shape)

    def handle_background_correction_diameter_changed(self, val):
        self.config['detection']['background_correction']['diameter'] = self.background_correction_diameter

    @property
    def cell_filter_intensity(self):
        intensities = self.tab.cellFilterThresholdIntensityDoublet.getValue()
        if intensities is None:
            return
        else:
            intensities = list(intensities)
            if intensities[-1] == -1:
                intensities[-1] = 65536  # FIXME: hard coded, should read max
            return intensities

    @cell_filter_intensity.setter
    def cell_filter_intensity(self, intensity):
        self.tab.cellFilterThresholdIntensityDoublet.setValue(intensity)

    def handle_filter_intensity_changed(self, _):
        self.config['cell_filtration']['thresholds']['intensity'] = self.cell_filter_intensity

    def scale_axis(self, val, axis='x'):
        axis_ratio = self.ratios['xyz'.index(axis)]
        return round(val * axis_ratio)

    def reverse_scale_axis(self, val, axis='x'):
        axis_ratio = self.ratios['xyz'.index(axis)]
        return round(val / axis_ratio)

    @property
    def slice_tuples(self):
        return ((self.crop_x_min, self.crop_x_max),
                (self.crop_y_min, self.crop_y_max),
                (self.crop_z_min, self.crop_z_max))

    @property
    def slicing(self):
        return tuple([slice(ax[0], ax[1]) for ax in self.slice_tuples])


class VesselParams(UiParameterCollection):
    def __init__(self, tab, sample_params, preprocessing_params, src_folder=None):
        super().__init__(tab, src_folder)
        # self.sample_params = sample_params  # TODO: check if required
        # self.preprocessing_params = preprocessing_params  # TODO: check if required
        self.binarization_params = VesselBinarizationParams(tab, src_folder)
        self.graph_params = VesselGraphParams(tab, src_folder)
        self.visualization_params = VesselVisualizationParams(tab, sample_params, preprocessing_params, src_folder)

    @property
    def params(self):
        return self.binarization_params, self.graph_params, self.visualization_params


class VesselBinarizationParams(UiParameter):
    run_raw_binarization: bool
    raw_binarization_clip_range: List[int]
    raw_binarization_threshold: int
    smooth_raw: bool
    binary_fill_raw: bool
    fill_main_channel: bool
    run_arteries_binarization: bool
    arteries_binarization_clip_range: List[int]
    arteries_binarization_threshold: int
    smooth_arteries: bool
    binary_fill_arteries: bool
    fill_secondary_channel: bool
    fill_combined: bool
    plot_step_1: str
    plot_step_2: str
    plot_channel_1: str
    plot_channel_2: str

    def __init__(self, tab, src_folder=None):
        super().__init__(tab, src_folder)
        self.params_dict = {
            'run_raw_binarization': ParamLink(['raw', 'binarization', 'run'], self.tab.runRawBinarizationCheckBox),
            'raw_binarization_clip_range': ParamLink(['raw', 'binarization', 'clip_range'],
                                                     self.tab.rawBinarizationClipRangeDoublet),
            'raw_binarization_threshold': ['raw', 'binarization', 'threshold'],
            'smooth_raw': ParamLink(['raw', 'smoothing', 'run'], self.tab.rawBinarizationSmoothingCheckBox),
            'binary_fill_raw': ParamLink(['raw', 'binary_filling', 'run'], self.tab.rawBinarizationBinaryFillingCheckBox),
            'fill_main_channel': ParamLink(['raw', 'deep_filling', 'run'], self.tab.binarizationRawDeepFillingCheckBox),
            'run_arteries_binarization': ParamLink(['arteries', 'binarization', 'run'],
                                                   self.tab.runArteriesBinarizationCheckBox),
            'arteries_binarization_clip_range': ParamLink(['arteries', 'binarization', 'clip_range'],
                                                          self.tab.arteriesBinarizationClipRangeDoublet),
            'arteries_binarization_threshold': ['arteries', 'binarization', 'threshold'],
            'smooth_arteries': ParamLink(['arteries', 'smoothing', 'run'],
                                         self.tab.arteriesBinarizationSmoothingCheckBox),
            'binary_fill_arteries': ParamLink(['arteries', 'binary_filling', 'run'],
                                              self.tab.arteriesBinarizationBinaryFillingCheckBox),
            'fill_secondary_channel': ParamLink(['arteries', 'deep_filling', 'run'],
                                                self.tab.binarizationArteriesDeepFillingCheckBox),
            'fill_combined': ParamLink(['combined', 'binary_fill'], self.tab.binarizationConbineBinaryFillingCheckBox),
            'plot_step_1': ParamLink(None, self.tab.binarizationPlotStep1ComboBox, connect=False),
            'plot_step_2': ParamLink(None, self.tab.binarizationPlotStep2ComboBox, connect=False),
            'plot_channel_1': ParamLink(None, self.tab.binarizationPlotChannel1ComboBox, connect=False),
            'plot_channel_2': ParamLink(None, self.tab.binarizationPlotChannel2ComboBox, connect=False),

        }
        self.cfg_subtree = ['binarization']
        self.connect()

    def connect(self):
        self.tab.rawBinarizationThresholdSpinBox.valueChanged.connect(self.handle_raw_binarization_threshold_changed)

        self.tab.arteriesBinarizationThresholdSpinBox.valueChanged.connect(
            self.handle_arteries_binarization_threshold_changed)
        self.connect_simple_widgets()

    @property
    def n_steps(self):
        n_steps = self.run_raw_binarization
        n_steps += self.smooth_raw or self.binary_fill_raw
        n_steps += self.fill_main_channel
        n_steps += self.run_arteries_binarization
        n_steps += self.smooth_arteries or self.binary_fill_arteries
        n_steps += self.fill_secondary_channel
        n_steps += self.fill_combined
        return

    @property
    def raw_binarization_threshold(self):
        return self.sanitize_neg_one(self.tab.rawBinarizationThresholdSpinBox.value())

    @raw_binarization_threshold.setter
    def raw_binarization_threshold(self, value):
        self.tab.rawBinarizationThresholdSpinBox.setValue(self.sanitize_nones(value))

    def handle_raw_binarization_threshold_changed(self):
        self.config['raw']['binarization']['threshold'] = self.raw_binarization_threshold

    @property
    def arteries_binarization_threshold(self):
        return self.sanitize_neg_one(self.tab.arteriesBinarizationThresholdSpinBox.value())

    @arteries_binarization_threshold.setter
    def arteries_binarization_threshold(self, value):
        self.tab.arteriesBinarizationThresholdSpinBox.setValue(self.sanitize_nones(value))

    def handle_arteries_binarization_threshold_changed(self):
        self.config['arteries']['binarization']['threshold'] = self.arteries_binarization_threshold


class VesselGraphParams(UiParameter):
    skeletonize: bool
    build: bool
    clean: bool
    reduce: bool
    transform: bool
    annotate: bool
    use_arteries: bool
    vein_intensity_range_on_arteries_channel: List[int]
    restrictive_min_vein_radius: float
    permissive_min_vein_radius: float
    final_min_vein_radius: float
    arteries_min_radius: float
    max_arteries_tracing_iterations: int
    max_veins_tracing_iterations: int
    min_artery_size: int
    min_vein_size: int

    def __init__(self, tab, src_folder=None):
        super().__init__(tab, src_folder)
        self.params_dict = {
            'skeletonize': ParamLink(['graph_construction', 'skeletonize'], self.tab.buildGraphSkeletonizeCheckBox),
            'build': ParamLink(['graph_construction', 'build'], self.tab.buildGraphBuildCheckBox),
            'clean': ParamLink(['graph_construction', 'clean'], self.tab.buildGraphCleanCheckBox),
            'reduce': ParamLink(['graph_construction', 'reduce'], self.tab.buildGraphReduceCheckBox),
            'transform': ParamLink(['graph_construction', 'transform'], self.tab.buildGraphTransformCheckBox),
            'annotate':  ParamLink(['graph_construction', 'annotate'], self.tab.buildGraphRegisterCheckBox),
            'use_arteries': ParamLink(['graph_construction', 'use_arteries'], self.tab.buildGraphUseArteriesCheckBox),
            'vein_intensity_range_on_arteries_channel': ParamLink(['vessel_type_postprocessing', 'pre_filtering', 'vein_intensity_range_on_arteries_ch'],
                                                                  self.tab.veinIntensityRangeOnArteriesChannelDoublet),
            'restrictive_min_vein_radius': ParamLink(['vessel_type_postprocessing', 'pre_filtering', 'restrictive_vein_radius'],
                                                     self.tab.restrictiveMinVeinRadiusSpinBox),
            'permissive_min_vein_radius': ParamLink(['vessel_type_postprocessing', 'pre_filtering', 'permissive_vein_radius'],
                                                    self.tab.permissiveMinVeinRadiusSpinBox),
            'final_min_vein_radius': ParamLink(['vessel_type_postprocessing', 'pre_filtering', 'final_vein_radius'],
                                               self.tab.finalMinVeinRadiusSpinBox),
            'arteries_min_radius': ParamLink(['vessel_type_postprocessing', 'pre_filtering', 'arteries_min_radius'],
                                             self.tab.arteriesMinRadiusSpinBox),
            'max_arteries_tracing_iterations': ParamLink(['vessel_type_postprocessing', 'tracing', 'max_arteries_iterations'],
                                                         self.tab.maxArteriesTracingIterationsSpinBox),
            'max_veins_tracing_iterations': ParamLink(['vessel_type_postprocessing', 'tracing', 'max_veins_iterations'],
                                                      self.tab.maxVeinsTracingIterationsSpinBox),
            'min_artery_size': ParamLink(['vessel_type_postprocessing', 'capillaries_removal', 'min_artery_size'],
                                         self.tab.minArterySizeSpinBox),
            'min_vein_size': ParamLink(['vessel_type_postprocessing', 'capillaries_removal', 'min_vein_size'],
                                       self.tab.minVeinSizeSpinBox)
        }
        self.connect()

    def connect(self):
        self.connect_simple_widgets()


class VesselVisualizationParams(UiParameter):
    crop_x_min: int
    crop_x_max: int
    crop_y_min: int
    crop_y_max: int
    crop_z_min: int
    crop_z_max: int
    graph_step: str
    plot_type: str
    voxelization_size: List[int]
    vertex_degrees: str
    weight_by_radius: bool

    def __init__(self, tab, sample_params=None, preprocessing_params=None, src_folder=None):
        super().__init__(tab, src_folder)
        self.params_dict = {  # TODO: if 99.9 % source put to 100% (None)
            'crop_x_min': ParamLink(['slicing', 'dim_0', 0], self.tab.graphConstructionSlicerXRangeMin),
            'crop_x_max': ParamLink(['slicing', 'dim_0', 1], self.tab.graphConstructionSlicerXRangeMax),
            'crop_y_min': ParamLink(['slicing', 'dim_1', 0], self.tab.graphConstructionSlicerYRangeMin),
            'crop_y_max': ParamLink(['slicing', 'dim_1', 1], self.tab.graphConstructionSlicerYRangeMax),
            'crop_z_min': ParamLink(['slicing', 'dim_2', 0], self.tab.graphConstructionSlicerZRangeMin),
            'crop_z_max': ParamLink(['slicing', 'dim_2', 1], self.tab.graphConstructionSlicerZRangeMax),
            'graph_step': ParamLink(None, self.tab.graphSlicerStepComboBox, connect=False),
            'plot_type': ParamLink(None, self.tab.graphPlotTypeComboBox, connect=False),
            'voxelization_size': ParamLink(['voxelization', 'size'], self.tab.vasculatureVoxelizationRadiusTriplet),
            'vertex_degrees': ParamLink(None, self.tab.vasculatureVoxelizationFilterDegreesSinglet, connect=False),
            'weight_by_radius': ParamLink(None, self.tab.voxelizationWeightByRadiusCheckBox, connect=False)
            # 'filter_name': ParamLink(None, self.tab.voxelizationFitlerNameComboBox, connect=False),
            # 'weight_name': ParamLink(None, self.tab.voxelizationWeightNameComboBox, connect=False),
        }
        self.structure_id = None
        self.cfg_subtree = ['visualization']
        self.sample_params = sample_params
        self.preprocessing_params = preprocessing_params
        self.connect()

    def connect(self):
        self.connect_simple_widgets()

    def set_structure_id(self, structure_widget):
        self.structure_id = int(structure_widget.text(1))

    @property
    def ratios(self):
        raw_res = np.array(self.sample_params.raw_resolution)
        atlas_res = np.array(self.preprocessing_params.registration.atlas_resolution)
        ratios = atlas_res / raw_res  # to original
        return ratios

    def scale_axis(self, val, axis='x'):
        return round(val * self.ratios['xyz'.index(axis)])

    def reverse_scale_axis(self, val, axis='x'):
        axis_ratio = self.ratios['xyz'.index(axis)]
        return round(val / axis_ratio)

    @property
    def slice_tuples(self):
        return ((self.crop_x_min, self.crop_x_max),
                (self.crop_y_min, self.crop_y_max),
                (self.crop_z_min, self.crop_z_max))

    @property
    def slicing(self):
        return tuple([slice(ax[0], ax[1]) for ax in self.slice_tuples])


class PreferencesParams(UiParameter):
    verbosity: str
    n_processes_file_conv: int
    n_processes_resampling: int
    n_processes_stitching: int
    n_processes_cell_detection: int
    n_processes_binarization: int
    chunk_size_min: int
    chunk_size_max: int
    chunk_size_overlap: int
    start_folder: str
    start_full_screen: bool
    lut: str
    font_size: int
    pattern_finder_min_n_files: int
    three_d_plot_bg: str

    def __init__(self, tab, src_folder=None):
        super().__init__(tab, src_folder)
        self.params_dict = {
            'verbosity': ['verbosity'],
            'n_processes_file_conv': ['n_processes_file_conv'],
            'n_processes_resampling': ['n_processes_resampling'],
            'n_processes_stitching': ['n_processes_stitching'],
            'n_processes_cell_detection': ['n_processes_cell_detection'],
            'n_processes_binarization': ['n_processes_binarization'],
            'chunk_size_min': ParamLink(['detection_chunk_size_min'], self.tab.chunkSizeMinSpinBox, connect=False),
            'chunk_size_max': ParamLink(['detection_chunk_size_max'], self.tab.chunkSizeMaxSpinBox, connect=False),
            'chunk_size_overlap': ParamLink(['detection_chunk_overlap'], self.tab.chunkSizeOverlapSpinBox, connect=False),
            'start_folder': ParamLink(['start_folder'], self.tab.startFolderLineEdit, connect=False),
            'start_full_screen': ParamLink(['start_full_screen'], self.tab.startFullScreenCheckBox, connect=False),
            'lut': ['default_lut'],
            'font_size': ParamLink(['font_size'], self.tab.fontSizeSpinBox, connect=False),
            'pattern_finder_min_n_files': ParamLink(['pattern_finder_min_n_files'],
                                                    self.tab.patternFinderMinFilesSpinBox, connect=False),
            'three_d_plot_bg': ['three_d_plot_bg']
        }
        self.connect()

    def _ui_to_cfg(self):  # TODO: check if live update (i.e. connected handlers) or only on save
        cfg = self._config
        cfg['verbosity'] = self.verbosity
        cfg['n_processes_file_conv'] = self.n_processes_file_conv
        cfg['n_processes_resampling'] = self.n_processes_resampling
        cfg['n_processes_stitching'] = self.n_processes_stitching
        cfg['n_processes_cell_detection'] = self.n_processes_cell_detection
        cfg['n_processes_binarization'] = self.n_processes_binarization
        cfg['detection_chunk_size_min'] = self.chunk_size_min
        cfg['detection_chunk_size_max'] = self.chunk_size_max
        cfg['detection_chunk_overlap'] = self.chunk_size_overlap
        cfg['start_folder'] = self.start_folder
        cfg['start_full_screen'] = self.start_full_screen
        cfg['default_lut'] = self.lut
        cfg['font_size'] = self.font_size
        cfg['pattern_finder_min_n_files'] = self.pattern_finder_min_n_files
        cfg['three_d_plot_bg'] = self.three_d_plot_bg

    def cfg_to_ui(self):
        self.reload()
        super().cfg_to_ui()

    @property
    def three_d_plot_bg(self):
        return self.tab.threeDPlotsBackgroundComboBox.currentText().lower()

    @three_d_plot_bg.setter
    def three_d_plot_bg(self, value):
        self.tab.threeDPlotsBackgroundComboBox.setCurrentText(value)

    @property
    def verbosity(self):
        return self.tab.verbosityComboBox.currentText().lower()

    @verbosity.setter
    def verbosity(self, lvl):
        self.tab.verbosityComboBox.setCurrentText(lvl.capitalize())

    @property
    def n_processes_file_conv(self):
        return self.sanitize_neg_one(self.tab.nProcessesFileConversionSpinBox.value())

    @n_processes_file_conv.setter
    def n_processes_file_conv(self, n_procs):
        self.tab.nProcessesFileConversionSpinBox.setValue(self.sanitize_nones(n_procs))

    @property
    def n_processes_stitching(self):
        return self.sanitize_neg_one(self.tab.nProcessesStitchingSpinBox.value())

    @n_processes_stitching.setter
    def n_processes_stitching(self, value):
        self.tab.nProcessesStitchingSpinBox.setValue(self.sanitize_nones(value))

    @property
    def n_processes_resampling(self):
        return self.sanitize_neg_one(self.tab.nProcessesResamplingSpinBox.value())

    @n_processes_resampling.setter
    def n_processes_resampling(self, value):
        self.tab.nProcessesResamplingSpinBox.setValue(self.sanitize_nones(value))

    @property
    def n_processes_cell_detection(self):
        return self.sanitize_neg_one(self.tab.nProcessesCellDetectionSpinBox.value())

    @n_processes_cell_detection.setter
    def n_processes_cell_detection(self, n_procs):
        self.tab.nProcessesCellDetectionSpinBox.setValue(self.sanitize_nones(n_procs))

    @property
    def n_processes_binarization(self):
        return self.sanitize_neg_one(self.tab.nProcessesBinarizationSpinBox.value())

    @n_processes_binarization.setter
    def n_processes_binarization(self, value):
        self.tab.nProcessesBinarizationSpinBox.setValue(self.sanitize_nones(value))

    @property
    def lut(self):
        return self.tab.lutComboBox.currentText().lower()

    @lut.setter
    def lut(self, lut_name):
        self.tab.lutComboBox.setCurrentText(lut_name)

    @property
    def font_family(self):
        return self.tab.fontComboBox.currentFont().family()

    # @font_family.setter
    # def font_family(self, font):
    #     self.tab.fontComboBox.setCurrentFont(font)


class BatchParams(UiParameter):

    def __init__(self, tab, src_folder=None, preferences=None):
        super().__init__(tab, src_folder)
        self.group_concatenator = ' vs '
        self.preferences = preferences
        self.tab.sampleFoldersToolBox = QToolBox(parent=self.tab)
        self.tab.sampleFoldersPageLayout.addWidget(self.tab.sampleFoldersToolBox, 3, 0)

        self.comparison_checkboxes = []

    def _ui_to_cfg(self):
        self.config['paths']['results_folder'] = self.results_folder
        self.config['groups'] = self.groups
        self.config['comparisons'] = {letter: pair for letter, pair in zip(string.ascii_lowercase,
                                                                           self.selected_comparisons)}

    def cfg_to_ui(self):
        self.reload()
        self.results_folder = self.config['paths']['results_folder']
        for i in range(len(self.config['groups'].keys())):
            self.add_group()
        self.group_names = self.config['groups'].keys()  # FIXME: check that ordered
        for i, paths in enumerate(self.config['groups'].values()):
            self.set_paths(i+1, paths)
        self.update_comparisons()
        for chk_bx in self.comparison_checkboxes:
            if chk_bx.text().split(self.group_concatenator) in self.config['comparisons'].values():
                self.set_check_state(chk_bx, True)

    def connect(self):
        self.tab.addGroupPushButton.clicked.connect(self.add_group)
        self.tab.removeGroupPushButton.clicked.connect(self.remove_group)
        self.tab.resultsFolderLineEdit.textChanged.connect(self.handle_results_folder_changed)
        # self.connect_simple_widgets()

    def connect_groups(self):
        for btn in self.gp_add_folder_buttons:
            self.__connect_btn(btn, self.handle_add_src_folder_clicked)
        for btn in self.gp_remove_folder_buttons:
            self.__connect_btn(btn, self.handle_remove_src_folder_clicked)
        for ctrl in self.gp_group_name_ctrls:
            self.__connect_line_edit(ctrl, self.update_comparisons)

    def __connect_btn(self, btn, callback):
        try:
            btn.clicked.connect(callback, type=Qt.UniqueConnection)
        except TypeError as err:
            if err.args[0] == 'connection is not unique':
                btn.clicked.disconnect()
                btn.clicked.connect(callback, type=Qt.UniqueConnection)
            else:
                raise err

    def __connect_line_edit(self, ctrl, callback):
        try:
            ctrl.editingFinished.connect(callback, type=Qt.UniqueConnection)
        except TypeError as err:
            if err.args[0] == 'connection is not unique':
                ctrl.editingFinished.disconnect()
                ctrl.editingFinished.connect(callback, type=Qt.UniqueConnection)
            else:
                raise err

    @property
    def comparisons(self):
        """

        Returns
        -------
            The list of all possible pairs of groups
        """
        # return list(combinations(self.group_names, 2))
        return list(permutations(self.group_names, 2))

    @property
    def selected_comparisons(self):
        return [box.text().split(self.group_concatenator) for box in self.comparison_checkboxes if box.isChecked()]

    def update_comparisons(self):
        self.comparison_checkboxes = []
        for i in range(self.tab.comparisonsVerticalLayout.count(), -1, -1):  # Clear
            item = self.tab.comparisonsVerticalLayout.takeAt(i)
            if item is not None:
                widg = item.widget()
                widg.setParent(None)
                widg.deleteLater()
        for pair in self.comparisons:
            chk = QCheckBox(self.group_concatenator.join(pair))
            chk.setChecked(False)
            self.tab.comparisonsVerticalLayout.addWidget(chk)
            self.comparison_checkboxes.append(chk)

    def add_group(self):  # REFACTOR: better in tab object
        new_gp_id = self.n_groups + 1
        group_controls = create_clearmap_widget('sample_group_controls.ui', patch_parent_class='QWidget')
        self.tab.sampleFoldersToolBox.addItem(group_controls, f'Group {new_gp_id}')

        self.connect_groups()

    def remove_group(self):
        last_idx = self.tab.sampleFoldersToolBox.count() - 1
        widg = self.tab.sampleFoldersToolBox.widget(last_idx)
        self.tab.sampleFoldersToolBox.removeItem(last_idx)
        widg.setParent(None)
        widg.deleteLater()

    @property
    def n_groups(self):
        return self.tab.sampleFoldersToolBox.count()

    @property
    def group_names(self):
        return [lbl.text() for lbl in self.gp_group_name_ctrls]

    @group_names.setter
    def group_names(self, names):
        for w, name in zip(self.gp_group_name_ctrls, names):
            w.setText(name)

    @property
    def gp_group_name_ctrls(self):
        return self.get_gp_ctrls('NameLineEdit')

    @property
    def gp_add_folder_buttons(self):
        return self.get_gp_ctrls('AddSrcFolderBtn')
    
    @property
    def gp_remove_folder_buttons(self):
        return self.get_gp_ctrls('RemoveSrcFolderBtn')

    @property
    def gp_list_widget(self):
        return self.get_gp_ctrls('ListWidget')

    def get_gp_ctrls(self, ctrl_name):
        return [getattr(self.tab.sampleFoldersToolBox.widget(i), f'gp{ctrl_name}') for i in range(self.n_groups)]

    def set_paths(self, gp, paths):
        list_widget = self.gp_list_widget[gp-1]
        list_widget.clear()
        list_widget.addItems(paths)

    def get_paths(self, gp):  # TODO: should exist from group name
        list_widget = self.gp_list_widget[gp-1]
        return [list_widget.item(i).text() for i in range(list_widget.count())]

    def get_all_paths(self):
        return [self.get_paths(gp+1) for gp in range(self.n_groups)]

    @property
    def groups(self):
        return {gp: paths for gp, paths in zip(self.group_names, self.get_all_paths())}

    def handle_add_src_folder_clicked(self):
        gp = self.tab.sampleFoldersToolBox.currentIndex()
        folder_path = get_directory_dlg(self.preferences.start_folder, 'Select sample folder')
        if folder_path:
            self.gp_list_widget[gp].addItem(folder_path)

    def handle_remove_src_folder_clicked(self):
        # print('call')
        gp = self.tab.sampleFoldersToolBox.currentIndex()
        sample_idx = self.gp_list_widget[gp].currentRow()
        _ = self.gp_list_widget[gp].takeItem(sample_idx)
        # print(_.text())

    @property
    def results_folder(self):
        return self.tab.resultsFolderLineEdit.text()

    @results_folder.setter
    def results_folder(self, value):
        self.tab.resultsFolderLineEdit.setText(value)

    def handle_results_folder_changed(self):
        self.config['paths']['results_folder'] = self.results_folder

    @property
    def align(self):
        return self.tab.batchAlignCheckBox.isChecked()

    @align.setter
    def align(self, value):
        self.tab.batchAlignCheckBox.setChecked(value)

    # def handle_align_changed(self):
    #     self.

    @property
    def count_cells(self):
        return self.tab.batchCountCellsCheckBox.isChecked()

    @count_cells.setter
    def count_cells(self, value):
        self.tab.batchCountCellsCheckBox.isChecked()

    # def handle_count_cells_changed(self):

    @property
    def run_vaculature(self):
        return self.tab.batchVasculatureCheckBox.isChecked()

    @run_vaculature.setter
    def run_vaculature(self, value):
        self.tab.batchVasculatureCheckBox.setChecked(value)
