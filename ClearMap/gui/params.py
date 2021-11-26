import os

import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QInputDialog

from ClearMap.config.config_loader import get_configobj_cfg


class ConfigNotFoundError(Exception):
    pass


class UiParameter(object):
    def __init__(self, tab, src_folder=None):
        self.tab = tab
        self.src_folder = src_folder
        self.config = None

    def get_config(self, cfg_path):
        self.config = get_configobj_cfg(cfg_path)  # FIXME:
        if not self.config:
            raise ConfigNotFoundError

    def write_config(self):
        self.config.write()

    def reload(self):
        self.config.reload()

    def _translate_state(self, state):
        if state is True:
            state = Qt.Checked
        elif state is False:
            state = Qt.Unchecked
        else:
            raise NotImplementedError('Unknown state {}'.format(state))
        return state

    def ui_to_cfg(self):
        self._ui_to_cfg()
        self.write_config()

    def _ui_to_cfg(self):
        raise NotImplementedError

    def cfg_to_ui(self):
        raise NotImplementedError

    def is_checked(self, check_box):
        return check_box.checkState() == Qt.Checked

    def set_check_state(self, check_box, state):
        state = self._translate_state(state)
        check_box.setCheckState(state)

    def sanitize_nones(self, val):
        return val if val is not None else -1

    def sanitize_neg_one(self, val):
        return val if val != -1 else None


class SampleParameters(UiParameter):

    def _ui_to_cfg(self):
        self.config['base_directory'] = self.src_folder
        self.config['sample_id'] = self.sample_id
        self.config['slice_x'] = self.slice_x
        self.config['slice_y'] = self.slice_y
        self.config['slice_z'] = self.slice_z
        self.config['orientation'] = self.orientation
        self.config['src_paths']['raw'] = self.raw_path
        self.config['src_paths']['arteries'] = self.arteries_path
        self.config['src_paths']['autofluorescence'] = self.autofluo_path

        for img_type in ('raw', 'autofluorescence', 'arteries'):
            res = getattr(self, '{}_resolution'.format(img_type))
            self.config['resolutions'][img_type] = res

    def cfg_to_ui(self):
        self.sample_id = self.config['sample_id']
        self.orientation = self.config['orientation']
        self.raw_path = self.config['src_paths']['raw']
        self.autofluo_path = self.config['src_paths']['autofluorescence']
        self.arteries_path = self.config['src_paths']['arteries']
        for k, v in self.config['resolutions'].items():
            if v is not None and v != 'auto':
                ctrl = getattr(self.tab, '{}ResolutionTriplet'.format(k))
                ctrl.enableControls()
                ctrl.setValue(v)

    def fix_sample_cfg_file(self, f_path):
        cfg = get_configobj_cfg(f_path)
        cfg['base_directory'] = os.path.dirname(f_path)
        if not self.sample_id:
            sample_id, ok = QInputDialog.getText(self.tab, 'Warning: missing ID',
                                                 '<b>Missing sample ID</b><br>Please input below')
            self.sample_id = sample_id
            if not ok:
                raise ValueError('Missing sample ID')
        cfg['sample_id'] = self.sample_id
        cfg.write()

    # Sample params
    @property
    def sample_id(self):
        return self.tab.sampleIdTxt.text()

    @sample_id.setter
    def sample_id(self, id):
        self.tab.sampleIdTxt.setText(id)

    @property
    def raw_path(self):
        return self.tab.rawPath.text()

    @raw_path.setter
    def raw_path(self, f_path):
        self.tab.rawPath.setText(f_path)

    @property
    def autofluo_path(self):
        return self.tab.autofluoPathOptionalLineEdit.text()

    @autofluo_path.setter
    def autofluo_path(self, f_path):
        self.tab.autofluoPathOptionalLineEdit.setText(f_path)

    @property
    def arteries_path(self):
        return self.tab.arteriesPathOptionalLineEdit.text()

    @arteries_path.setter
    def arteries_path(self, f_path):
        self.tab.arteriesPathOptionalLineEdit.setText(f_path)

    @property
    def raw_resolution(self):
        return self.tab.rawResolutionTriplet.getValue()

    @raw_resolution.setter
    def raw_resolution(self, res):
        self.tab.rawResolutionTriplet.setValue(res)

    @property
    def autofluorescence_resolution(self):
        return self.tab.autofluorescenceResolutionTriplet.getValue()

    @autofluorescence_resolution.setter
    def autofluorescence_resolution(self, res):
        self.tab.autofluorescenceResolutionTriplet.setValue(res)

    @property
    def arteries_resolution(self):
        return self.tab.arteriesResolutionTriplet.getValue()

    @arteries_resolution.setter
    def arteries_resolution(self, res):
        self.tab.arteriesResolutionTriplet.setValue(res)

    @property
    def slice_x(self):
        return self.tab.sliceXDoublet.getValue()

    @slice_x.setter
    def slice_x(self, slc):
        self.tab.sliceXDoublet.setValue(slc)

    @property
    def slice_y(self):
        return self.tab.sliceYDoublet.getValue()

    @slice_y.setter
    def slice_y(self, slc):
        self.tab.sliceYDoublet.setValue(slc)

    @property
    def slice_z(self):
        return self.tab.sliceZDoublet.getValue()

    @slice_z.setter
    def slice_z(self, slc):
        self.tab.sliceZDoublet.setValue(slc)

    @property
    def orientation(self):
        x = int(self.tab.orient_x.currentText())
        y = int(self.tab.orient_y.currentText())
        z = int(self.tab.orient_z.currentText())
        n_axes = len(set([abs(e) for e in (x, y, z)]))
        if n_axes != 3:
            raise ValueError('Number of different axis is only {} instead of 3. '
                             'Please amend duplicate axes'.format(n_axes))
        return x, y, z

    @orientation.setter
    def orientation(self, orientation):
        n_axes = len(set([abs(e) for e in orientation]))
        if n_axes != 3:
            raise ValueError('Number of different axis is only {} instead of 3. '
                             'Please amend duplicate axes'.format(n_axes))
        self.tab.orient_x.setCurrentText('{}'.format(orientation[0]))
        self.tab.orient_y.setCurrentText('{}'.format(orientation[1]))
        self.tab.orient_z.setCurrentText('{}'.format(orientation[2]))

    @property
    def auto_fluo_path_checked(self):
        return self.is_checked(self.tab.autofluoCheckBox)

    @auto_fluo_path_checked.setter
    def auto_fluo_path_checked(self, state):
        self.set_check_state(self.tab.autofluoCheckBox, state)

    @property
    def arteries_path_checked(self):
        return self.is_checked(self.tab.arteriesPathCheckBox)

    @arteries_path_checked.setter
    def arteries_path_checked(self, state):
        self.set_check_state(self.tab.arteriesPathCheckBox, state)


class RigidStitchingParams(UiParameter):

    def _ui_to_cfg(self):
        cfg = self.config['stitching']['rigid']
        cfg['overlap_x'] = self.x_overlap
        cfg['overlap_y'] = self.y_overlap
        cfg['project_thickness'] = self.projection_thickness
        cfg['max_shifts_x'] = self.max_shifts_x
        cfg['max_shifts_y'] = self.max_shifts_y
        cfg['max_shifts_z'] = self.max_shifts_z
        cfg['background_level'] = self.background_level
        cfg['background_pixels'] = self.background_pixels

    def cfg_to_ui(self):
        cfg = self.config['stitching']['rigid']
        self.x_overlap = cfg['overlap_x']
        self.y_overlap = cfg['overlap_y']
        self.projection_thickness = cfg['project_thickness']
        self.max_shifts_x = cfg['max_shifts_x']
        self.max_shifts_y = cfg['max_shifts_y']
        self.max_shifts_z = cfg['max_shifts_z']
        self.background_level = cfg['background_level']
        self.background_pixels = cfg['background_pixels']

    @property
    def skip(self):
        return self.is_checked(self.tab.skipRigidCheckbox)

    @skip.setter
    def skip(self, state):
        self.set_check_state(self.tab.skipRigidCheckbox, state)

    @property
    def x_overlap(self):
        return self.tab.xOverlapSinglet.getValue()

    @x_overlap.setter
    def x_overlap(self, overlap):
        self.tab.xOverlapSinglet.setValue(overlap)

    @property
    def y_overlap(self):
        return self.tab.yOverlapSinglet.getValue()

    @y_overlap.setter
    def y_overlap(self, overlap):
        self.tab.yOverlapSinglet.setValue(overlap)

    @property
    def projection_thickness(self):
        val = self.tab.projectionThicknessDoublet.getValue()
        if val is not None:
            val.append(None)
        return val

    @projection_thickness.setter
    def projection_thickness(self, thickness):
        if thickness is None:
            self.tab.projectionThicknessDoublet.setValue(thickness)
        else:
            self.tab.projectionThicknessDoublet.setValue(thickness[: 2])

    @property
    def max_shifts_x(self):
        return self.tab.rigidMaxShiftsXDoublet.getValue()

    @max_shifts_x.setter
    def max_shifts_x(self, max_shift):
        self.tab.rigidMaxShiftsXDoublet.setValue(max_shift)

    @property
    def max_shifts_y(self):
        return self.tab.rigidMaxShiftsYDoublet.getValue()

    @max_shifts_y.setter
    def max_shifts_y(self, max_shift):
        self.tab.rigidMaxShiftsYDoublet.setValue(max_shift)

    @property
    def max_shifts_z(self):
        return self.tab.rigidMaxShiftsZDoublet.getValue()

    @max_shifts_z.setter
    def max_shifts_z(self, max_shift):
        self.tab.rigidMaxShiftsZDoublet.setValue(max_shift)

    @property
    def background_level(self):
        return self.tab.rigidBackgroundLevel.value()

    @background_level.setter
    def background_level(self, lvl):
        self.tab.rigidBackgroundLevel.setValue(lvl)

    @property
    def background_pixels(self):
        return self.tab.rigidBackgroundPixels.value()

    @background_pixels.setter
    def background_pixels(self, n_pix):
        self.tab.rigidBackgroundPixels.setValue(n_pix)


class WobblyStitchingParams(UiParameter):

    def _ui_to_cfg(self):
        cfg = self.config['stitching']['wobbly']
        cfg['skip'] = self.skip
        cfg['max_shifts_x'] = self.max_shifts_x
        cfg['max_shifts_y'] = self.max_shifts_y
        cfg['max_shifts_z'] = self.max_shifts_z
        cfg['valid_range'] = self.valid_range
        cfg['slice_range'] = self.slice_range
        cfg['slice_pixel_size'] = self.slice_pixel_size

    def cfg_to_ui(self):
        cfg = self.config['stitching']['wobbly']
        self.skip = cfg['skip']
        self.max_shifts_x = cfg['max_shifts_x']
        self.max_shifts_y = cfg['max_shifts_y']
        self.max_shifts_z = cfg['max_shifts_z']
        self.valid_range = cfg['valid_range']
        self.slice_range = cfg['slice_range']
        self.slice_pixel_size = cfg['slice_pixel_size']
        
    @property
    def skip(self):
        return self.is_checked(self.tab.skipWobblyCheckBox)

    @skip.setter
    def skip(self, state):
        self.set_check_state(self.tab.skipWobblyCheckBox, state)

    @property
    def max_shifts_x(self):
        return self.tab.wobblyMaxShiftsXDoublet.getValue()

    @max_shifts_x.setter
    def max_shifts_x(self, max_shift):
        self.tab.wobblyMaxShiftsXDoublet.setValue(max_shift)

    @property
    def max_shifts_y(self):
        return self.tab.wobblyMaxShiftsYDoublet.getValue()

    @max_shifts_y.setter
    def max_shifts_y(self, max_shift):
        self.tab.wobblyMaxShiftsYDoublet.setValue(max_shift)

    @property
    def max_shifts_z(self):
        return self.tab.wobblyMaxShiftsZDoublet.getValue()

    @max_shifts_z.setter
    def max_shifts_z(self, max_shift):
        self.tab.wobblyMaxShiftsZDoublet.setValue(max_shift)

    @property
    def valid_range(self):
        rng = self.tab.wobblyValidRangeDoublet.getValue()
        # rng = minus_1_to_none(rng)
        return rng

    @valid_range.setter
    def valid_range(self, rng):
        # rng = none_to_minus_1(rng)
        self.tab.wobblyValidRangeDoublet.setValue(rng)

    @property
    def slice_range(self):
        rng = self.tab.wobblySliceRangeDoublet.getValue()
        # rng = minus_1_to_none(rng)
        return rng

    @slice_range.setter
    def slice_range(self, rng):
        # rng = none_to_minus_1(rng)
        self.tab.wobblySliceRangeDoublet.setValue(rng)
        
    @property
    def slice_pixel_size(self):
        return self.tab.wobblySlicePixelSizeSinglet.getValue()

    @slice_pixel_size.setter
    def slice_pixel_size(self, size):
        self.tab.wobblySlicePixelSizeSinglet.setValue(size)


class GeneralStitchingParams(UiParameter):

    def _ui_to_cfg(self):
        cfg = self.config['stitching']
        cfg['run']['raw'] = self.run_raw
        cfg['run']['arteries'] = self.run_arteries
        cfg['preview']['raw'] = self.preview_raw
        cfg['preview']['arteries'] = self.preview_arteries
        cfg['output_conversion']['skip'] = not self.convert_output
        cfg['output_conversion']['raw'] = self.convert_raw
        cfg['output_conversion']['arteries'] = self.convert_arteries
        cfg['output_conversion']['format'] = self.conversion_fmt

    def cfg_to_ui(self):
        cfg = self.config['stitching']
        self.run_raw = cfg['run']['raw']
        self.run_arteries = cfg['run']['arteries']
        self.preview_raw = cfg['preview']['raw']
        self.preview_arteries = cfg['preview']['arteries']
        self.convert_output = not cfg['output_conversion']['skip']
        self.convert_raw = cfg['output_conversion']['raw']
        self.convert_arteries = cfg['output_conversion']['arteries']
        self.conversion_fmt = cfg['output_conversion']['format']

    @property
    def run_raw(self):
        return self.is_checked(self.tab.stitchingRunRawCheckBox)

    @run_raw.setter
    def run_raw(self, state):
        self.set_check_state(self.tab.stitchingRunRawCheckBox, state)

    @property
    def run_arteries(self):
        return self.is_checked(self.tab.stitchingRunArteriesCheckBox)

    @run_arteries.setter
    def run_arteries(self, state):
        self.set_check_state(self.tab.stitchingRunArteriesCheckBox, state)

    @property
    def preview_raw(self):
        return self.is_checked(self.tab.stitchingPreviewRawCheckBox)

    @preview_raw.setter
    def preview_raw(self, state):
        self.set_check_state(self.tab.stitchingPreviewRawCheckBox, state)

    @property
    def preview_arteries(self):
        return self.is_checked(self.tab.stitchingPreviewArteriesCheckBox)

    @preview_arteries.setter
    def preview_arteries(self, state):
        self.set_check_state(self.tab.stitchingPreviewArteriesCheckBox, state)

    @property
    def convert_output(self):
        return self.tab.skipOutputConversioncheckBox.checkState() == Qt.Unchecked  # unchecked to invert

    @convert_output.setter
    def convert_output(self, skip):
        self.set_check_state(self.tab.skipOutputConversioncheckBox, not skip)

    @property
    def convert_raw(self):
        return self.is_checked(self.tab.stitchingConvertRawCheckBox)

    @convert_raw.setter
    def convert_raw(self, state):
        self.set_check_state(self.tab.stitchingConvertRawCheckBox, state)

    @property
    def convert_arteries(self):
        return self.is_checked(self.tab.stitchingConvertArteriesCheckBox)

    @convert_arteries.setter
    def convert_arteries(self, state):
        self.set_check_state(self.tab.stitchingConvertArteriesCheckBox, state)

    @property
    def conversion_fmt(self):
        return self.tab.outputConversionFormat.currentText()

    @conversion_fmt.setter
    def conversion_fmt(self, fmt):
        self.tab.outputConversionFormat.setCurrentText(fmt)


class RegistrationParams(UiParameter):
    def _ui_to_cfg(self):
        cfg = self.config['registration']
        cfg['resampling']['skip'] = self.skip
        cfg['resampling']['raw_sink_resolution'] = self.raw_atlas_resolution
        cfg['resampling']['autofluo_sink_resolution'] = self.autofluo_atlas_resolution
        cfg['atlas']['align_files_folder'] = self.atlas_folder
        cfg['atlas']['align_channels_affine_file'] = self.channel_affine_file_path
        cfg['atlas']['align_reference_affine_file'] = self.ref_affine_file_path
        cfg['atlas']['align_reference_bspline_file'] = self.ref_bspline_file_path

    def cfg_to_ui(self):
        cfg = self.config['registration']
        self.skip = cfg['resampling']['skip']
        self.raw_atlas_resolution = cfg['resampling']['raw_sink_resolution']
        self.autofluo_atlas_resolution = cfg['resampling']['autofluo_sink_resolution']
        self.atlas_folder = cfg['atlas']['align_files_folder']
        self.channel_affine_file_path = cfg['atlas']['align_channels_affine_file']
        self.ref_affine_file_path = cfg['atlas']['align_reference_affine_file']
        self.ref_bspline_file_path = cfg['atlas']['align_reference_bspline_file']

    @property
    def skip(self):
        return self.is_checked(self.tab.skipRegistrationCheckBox)

    @skip.setter
    def skip(self, state):
        self.set_check_state(self.tab.skipRegistrationCheckBox, state)
        
    @property
    def raw_atlas_resolution(self):
        return self.tab.rawAtlasResolutionTriplet.getValue()

    @raw_atlas_resolution.setter
    def raw_atlas_resolution(self, res):
        self.tab.rawAtlasResolutionTriplet.setValue(res)

    @property
    def autofluo_atlas_resolution(self):
        return self.tab.autofluoAtlasResolutionTriplet.getValue()

    @autofluo_atlas_resolution.setter
    def autofluo_atlas_resolution(self, res):
        self.tab.autofluoAtlasResolutionTriplet.setValue(res)

    @property
    def atlas_folder(self):
        return self.tab.atlasFolderPath.text()

    @atlas_folder.setter
    def atlas_folder(self, folder):
        self.tab.atlasFolderPath.setText(folder)

    @property
    def channel_affine_file_path(self):
        return self.tab.channelAffineFilePath.text()

    @channel_affine_file_path.setter
    def channel_affine_file_path(self, f_path):
        self.tab.channelAffineFilePath.setText(f_path)

    @property
    def ref_affine_file_path(self):
        return self.tab.refAffineFilePath.text()

    @ref_affine_file_path.setter
    def ref_affine_file_path(self, f_path):
        self.tab.refAffineFilePath.setText(f_path)

    @property
    def ref_bspline_file_path(self):
        return self.tab.refBsplineFilePath.text()

    @ref_bspline_file_path.setter
    def ref_bspline_file_path(self, f_path):
        self.tab.refBsplineFilePath.setText(f_path)


class CellMapParams(UiParameter):
    def _ui_to_cfg(self):
        cfg = self.config
        cfg['detection']['background_correction']['diameter'] = self.background_correction_diameter
        cfg['detection']['shape_detection']['threshold'] = self.detection_threshold
        # self.crop_values_to_cfg()
        cfg['cell_filtration']['thresholds']['size'] = self.cell_filter_size
        cfg['voxelization']['radii'] = self.voxelization_radii
        cfg['run']['plot_when_finished'] = self.plot_when_finished

    def crop_values_to_cfg(self, ratios=None):
        if ratios is not None:
            self.scale_crop_values(ratios)
        cfg = self.config['detection']['test_set_slicing']  # TODO: if 99.9 % source put to 100% (None)
        cfg['dim_0'] = self.crop_x_min, self.crop_x_max
        cfg['dim_1'] = self.crop_y_min, self.crop_y_max
        cfg['dim_2'] = self.crop_z_min, self.crop_z_max

    def scale_crop_values(self, ratios):  # FIXME: ensure scaling when reading and writing
        crop_values = []
        ui_crops = self.crop_x_min, self.crop_x_max, self.crop_y_min, self.crop_y_max, self.crop_z_min, self.crop_z_max
        for ratio, val in zip(np.repeat(ratios, 2), ui_crops):
            crop_values.append(round(ratio * val))
        return crop_values

    def _crop_values_from_cfg(self, ratios=None):
        if ratios is not None:
            self.scale_crop_values(ratios)
        cfg = self.config['detection']['test_set_slicing']  # TODO: if 99.9 % source put to 100% (None)
        self.crop_x_min, self.crop_x_max = cfg['dim_0']
        self.crop_y_min, self.crop_y_max = cfg['dim_1']
        self.crop_z_min, self.crop_z_max = cfg['dim_2']

    def cfg_to_ui(self):
        cfg = self.config
        self.background_correction_diameter = cfg['detection']['background_correction']['diameter']
        self.detection_threshold = cfg['detection']['shape_detection']['threshold']
        self.cell_filter_size = cfg['cell_filtration']['thresholds']['size']
        self.voxelization_radii = cfg['voxelization']['radii']
        self.plot_when_finished = cfg['run']['plot_when_finished']
        try:
            self._crop_values_from_cfg()
        except KeyError as err:
            print('Could not load crop values from CellMap config, {}'.format(err))

    @property
    def plot_when_finished(self):
        return self.is_checked(self.tab.runCellMapPlotCheckBox)

    @plot_when_finished.setter
    def plot_when_finished(self, state):
        self.set_check_state(self.tab.runCellMapPlotCheckBox, state)

    @property
    def background_correction_diameter(self):
        return [self.tab.backgroundCorrectionDiameter.value()] * 2

    @background_correction_diameter.setter
    def background_correction_diameter(self, shape):
        if isinstance(shape, (list, tuple)):
            shape = shape[0]
        self.tab.backgroundCorrectionDiameter.setValue(shape)

    @property
    def detection_threshold(self):
        return self.tab.detectionThreshold.value()

    @detection_threshold.setter
    def detection_threshold(self, thrsh):
        self.tab.detectionThreshold.setValue(thrsh)
        
    @property
    def cell_filter_size(self):
        return self.tab.cellFilterThresholdSizeDoublet.getValue()

    @cell_filter_size.setter
    def cell_filter_size(self, size):
        self.tab.cellFilterThresholdSizeDoublet.setValue(size)

    @property
    def voxelization_radii(self):
        return self.tab.voxelizationRadiusTriplet.getValue()

    @voxelization_radii.setter
    def voxelization_radii(self, radii):
        self.tab.voxelizationRadiusTriplet.setValue(radii)

    @property
    def crop_x_min(self):
        return self.tab.detectionSubsetXRangeMin.value()

    @crop_x_min.setter
    def crop_x_min(self, val):
        self.tab.detectionSubsetXRangeMin.setValue(val)

    @property
    def crop_x_max(self):
        return self.tab.detectionSubsetXRangeMax.value()

    @crop_x_max.setter
    def crop_x_max(self, val):
        self.tab.detectionSubsetXRangeMax.setValue(val)

    @property
    def crop_y_min(self):
        return self.tab.detectionSubsetYRangeMin.value()

    @crop_y_min.setter
    def crop_y_min(self, val):
        self.tab.detectionSubsetYRangeMin.setValue(val)

    @property
    def crop_y_max(self):
        return self.tab.detectionSubsetYRangeMax.value()

    @crop_y_max.setter
    def crop_y_max(self, val):
        self.tab.detectionSubsetYRangeMax.setValue(val)

    @property
    def crop_z_min(self):
        return self.tab.detectionSubsetZRangeMin.value()

    @crop_z_min.setter
    def crop_z_min(self, val):
        self.tab.detectionSubsetZRangeMin.setValue(val)

    @property
    def crop_z_max(self):
        return self.tab.detectionSubsetZRangeMax.value()

    @crop_z_max.setter
    def crop_z_max(self, val):
        self.tab.detectionSubsetZRangeMax.setValue(val)

    @property
    def plot_detected_cells(self):
        return self.is_checked(self.tab.cellDetectionPlotCheckBox)

    @plot_detected_cells.setter
    def plot_detected_cells(self, state):
        self.set_check_state(self.tab.cellDetectionPlotCheckBox, state)


class PreferencesParams(UiParameter):
    def _ui_to_cfg(self):
        cfg = self.config
        cfg['verbosity'] = self.verbosity
        cfg['n_processes_file_conv'] = self.n_processes_file_conv
        cfg['n_processes_cell_detection'] = self.n_processes_cell_detection
        cfg['detection_chunk_size_min'] = self.chunk_size_min
        cfg['detection_chunk_size_max'] = self.chunk_size_max
        cfg['detection_chunk_overlap'] = self.chunk_size_overlap
        cfg['start_folder'] = self.start_folder
        cfg['start_full_screen'] = self.start_full_screen

    def cfg_to_ui(self):
        cfg = self.config
        self.verbosity = cfg['verbosity']
        self.n_processes_file_conv = cfg['n_processes_file_conv']
        self.n_processes_cell_detection = cfg['n_processes_cell_detection']
        self.chunk_size_min = cfg['detection_chunk_size_min']
        self.chunk_size_max = cfg['detection_chunk_size_max']
        self.chunk_size_overlap = cfg['detection_chunk_overlap']
        self.start_folder = cfg['start_folder']
        self.start_full_screen = cfg['start_full_screen']

    @property
    def start_folder(self):
        return self.tab.startFolderLineEdit.text()

    @start_folder.setter
    def start_folder(self, dir_path):
        self.tab.startFolderLineEdit.setText(dir_path)

    @property
    def start_full_screen(self):
        return self.is_checked(self.tab.startFullScreenCheckBox)

    @start_full_screen.setter
    def start_full_screen(self, state):
        self.set_check_state(self.tab.startFullScreenCheckBox, state)

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
    def n_processes_cell_detection(self):
        return self.sanitize_neg_one(self.tab.nProcessesCellDetectionSpinBox.value())

    @n_processes_cell_detection.setter
    def n_processes_cell_detection(self, n_procs):
        self.tab.nProcessesCellDetectionSpinBox.setValue(self.sanitize_nones(n_procs))

    @property
    def chunk_size_min(self):
        return self.tab.chunkSizeMinSpinBox.value()

    @chunk_size_min.setter
    def chunk_size_min(self, size):
        self.tab.chunkSizeMinSpinBox.setValue(size)

    @property
    def chunk_size_max(self):
        return self.tab.chunkSizeMaxSpinBox.value()

    @chunk_size_max.setter
    def chunk_size_max(self, size):
        self.tab.chunkSizeMaxSpinBox.setValue(size)

    @property
    def chunk_size_overlap(self):
        return self.tab.chunkSizeOverlapSpinBox.value()

    @chunk_size_overlap.setter
    def chunk_size_overlap(self, size):
        self.tab.chunkSizeOverlapSpinBox.setValue(size)
