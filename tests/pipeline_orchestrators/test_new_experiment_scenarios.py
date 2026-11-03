"""
What a user does in a new experiment, headless, one test per scenario.

Every scenario is a bug report that was fixed: an incomplete channel (a data type but no path yet, or no
resolution yet) must be hydrated, never processed, and must not break the workers. See the "Contract" in the
docstring of :mod:`ClearMap.pipeline_orchestrators.generic_orchestrators`.
"""
import warnings
from pathlib import Path

import numpy as np
import pytest
import tifffile

import ClearMap.pipeline_orchestrators.colocalization as colocalization_module
import ClearMap.pipeline_orchestrators.tube_map as tube_map_module
from ClearMap.Utils.exceptions import (ClearMapConfigError, ClearMapRuntimeError, ClearMapVRamException,
                                       MissingAssetError)
from ClearMap.config.config_adjusters.dict_ops import get_nested
from ClearMap.config.config_adjusters.policy import INSTANCE_SPECS_REGISTRY
from ClearMap.config.config_repository import ConfigRepository

from .conftest import RAW_FOLDER_NAME, ChannelEditor

DATA_TYPE = 'nuclei'
RESOLUTION = (1.625, 1.625, 3.0)
CHANNELS_OF_EVERY_PIPELINE = {'nuclei_a': 'nuclei', 'nuclei_b': 'nuclei', 'myelin': 'myelin', 'vessels': 'vessels'}
STACK_SHAPE = (3, 8, 8)  # z, y, x: a tiny stack, enough for the file handling
TILE_EXPRESSION = 'tile_Y<Y,2>_X<X,2>.tif'
TEMPLATES_SANITIZER_WARNING = '[runtime-templates]'  # What the last resort sanitizer says when it removes templates


# ----------------------------------------------------------------------------------------------------------------
# The sample manager: when does a channel join the workspace
# ----------------------------------------------------------------------------------------------------------------

def test_new_experiment_has_a_workspace_without_channels(experiment, sample_manager):
    assert sample_manager.workspace_ready
    assert sample_manager.channels == []
    assert sample_manager.setup_complete


def test_channel_with_a_data_type_only_is_incomplete(sample_manager, channel_editor):
    channel_editor.add('test', DATA_TYPE)

    assert sample_manager.channels == ['test']
    assert sample_manager.incomplete_channels == ['test']
    assert sample_manager.complete_channels == []
    assert not sample_manager.setup_complete
    assert not sample_manager.has_channels_in_workspace(['test'])


def test_channel_with_a_data_type_only_is_not_pipeline_ready(sample_manager, channel_editor):
    """Entering the data type before the path used to make callers ask the workspace for a channel it lacks"""
    channel_editor.add('test', DATA_TYPE)

    assert sample_manager.pipeline_ready_channels == []
    channel_editor.set_path('test')
    assert sample_manager.pipeline_ready_channels == ['test']


def test_channel_joins_the_workspace_when_it_has_a_path_and_a_data_type(sample_manager, channel_editor):
    channel_editor.add('test', DATA_TYPE)
    channel_editor.set_path('test')

    assert sample_manager.complete_channels == ['test']
    assert sample_manager.incomplete_channels == []
    assert sample_manager.setup_complete
    assert sample_manager.has_channels_in_workspace(['test'])


def test_channel_with_a_path_only_is_incomplete(sample_manager, channel_editor):
    channel_editor.add('test', with_path=True)

    assert 'test' not in sample_manager.complete_channels


# ----------------------------------------------------------------------------------------------------------------
# Registration: "KeyError: 'Channel_0'" / "KeyError: 'test'" when adding a channel in a new experiment
# ----------------------------------------------------------------------------------------------------------------

def test_registration_ignores_a_channel_without_path(experiment, channel_editor):
    channel_editor.add('test', DATA_TYPE)

    registration = experiment.get_worker('registration')
    registration.setup_if_needed()

    assert registration.all_channels == ['test']
    assert registration.channels == []
    assert registration.setup_complete


def test_registration_follows_a_channel_which_becomes_complete(experiment, channel_editor, monkeypatch):
    channel_editor.add('test', DATA_TYPE)
    registration = experiment.get_worker('registration')
    registration.setup_if_needed()
    setups = []
    original_setup = registration.setup

    def recording_setup(*args, **kwargs):
        setups.append(1)
        return original_setup(*args, **kwargs)

    monkeypatch.setattr(registration, 'setup', recording_setup)

    channel_editor.set_path('test')  # No call to the worker: it follows the workspace

    assert setups == [1]  # The setup is redone once, to include the channel
    assert registration.channels == ['test']
    assert registration.setup_complete


def test_registration_lists_are_subsets_of_the_lists_of_all_channels(experiment, channel_editor):
    channel_editor.add('complete', DATA_TYPE, with_path=True)
    channel_editor.add('incomplete', DATA_TYPE)
    registration = experiment.get_worker('registration')

    assert registration.all_channels == ['complete', 'incomplete']
    assert registration.channels == ['complete']
    assert set(registration.channels_to_register) <= set(registration.all_channels_to_register)
    assert set(registration.channels_to_resample) <= set(registration.channels)


def test_workspace_names_the_unknown_channel(experiment, channel_editor):
    channel_editor.add('test', DATA_TYPE)
    workspace = experiment.sample_manager.workspace

    with pytest.raises(MissingAssetError, match='Unknown channel "test"') as error:
        workspace['test']
    assert isinstance(error.value, KeyError)  # For the callers which catch it


# ----------------------------------------------------------------------------------------------------------------
# Colocalization: "unknown ['channel_2-channel_3']", then "TypeError: 'NoneType' object is not iterable"
# ----------------------------------------------------------------------------------------------------------------

def test_two_nuclei_channels_make_a_valid_colocalization_pair(experiment, channel_editor):
    channel_editor.add('channel_2', DATA_TYPE, with_path=True)
    channel_editor.add('channel_3', DATA_TYPE, with_path=True)  # Was rejected by the validation of the config

    assert list(experiment.get_config_view()['colocalization']['channels']) == ['channel_2-channel_3']


def test_colocalization_needs_two_complete_channels(sample_manager, channel_editor):
    channel_editor.add('channel_2', DATA_TYPE, with_path=True)
    channel_editor.add('channel_3', DATA_TYPE)

    assert sample_manager.is_colocalization_compatible  # The config follows the data types
    assert not sample_manager.is_colocalization_ready  # The tab needs the channels to be complete
    assert sample_manager.complete_colocalization_pairs() == []

    channel_editor.set_path('channel_3')

    assert sample_manager.is_colocalization_ready
    assert sample_manager.complete_colocalization_pairs() == [('channel_2', 'channel_3')]


def test_resolution_of_a_new_channel_is_not_set(sample_manager, channel_editor):
    channel_editor.add('test', DATA_TYPE, with_path=True)

    with pytest.raises(ClearMapConfigError, match='test'):  # Was TypeError: 'NoneType' object is not iterable
        sample_manager.get_channel_resolution('test')

    channel_editor.set_resolution('test', RESOLUTION)

    assert sample_manager.get_channel_resolution('test') == RESOLUTION


def test_colocalization_worker_waits_for_the_resolution(experiment, channel_editor):
    channel_editor.add('channel_2', DATA_TYPE, with_path=True)
    channel_editor.add('channel_3', DATA_TYPE, with_path=True)

    worker = experiment.get_worker('colocalization', ('channel_2', 'channel_3'))  # Was TypeError
    worker.setup_if_needed()  # Was TypeError: setup() missing 2 required positional arguments

    assert worker.setup_complete
    assert not worker.setup_finalised
    with pytest.raises(ClearMapRuntimeError, match='channel_2'):
        worker.colocalization_channel('channel_2')


def test_colocalization_worker_waits_for_the_detected_cells(experiment, channel_editor):
    channel_editor.add('channel_2', DATA_TYPE, with_path=True, resolution=RESOLUTION)
    channel_editor.add('channel_3', DATA_TYPE, with_path=True, resolution=RESOLUTION)

    worker = experiment.get_worker('colocalization', ('channel_2', 'channel_3'))
    worker.setup_if_needed()

    assert worker.setup_complete
    assert not worker.setup_finalised  # No cells detected yet
    with pytest.raises(ClearMapRuntimeError, match='detected'):
        worker.colocalization_channel('channel_3')


def _no_cells_table(self, channel):
    return None


class _FakeColocalizationChannel:
    """The colocalization package is not what is tested here: it computes on the detected cells"""

    def __init__(self, cells_shape_path, cells, voxel_dims):
        self.voxel_dims = voxel_dims


def test_colocalization_worker_is_finalised_when_the_resolution_and_the_cells_are_there(experiment, channel_editor,
                                                                                        monkeypatch):
    monkeypatch.setattr(colocalization_module, 'ColocalizationChannel', _FakeColocalizationChannel)
    monkeypatch.setattr(colocalization_module.ColocalizationProcessor, 'get_cells_df', _no_cells_table)
    channel_editor.add('channel_2', DATA_TYPE, with_path=True)
    channel_editor.add('channel_3', DATA_TYPE, with_path=True)
    worker = experiment.get_worker('colocalization', ('channel_2', 'channel_3'))
    for channel in ('channel_2', 'channel_3'):  # The cells are detected
        cells_shape = Path(worker.get('cells', channel=channel, asset_sub_type='shape').path)
        cells_shape.parent.mkdir(parents=True, exist_ok=True)
        cells_shape.touch()
    worker.setup_if_needed()
    assert not worker.setup_finalised  # Still no resolution

    channel_editor.set_resolution('channel_2', RESOLUTION)
    channel_editor.set_resolution('channel_3', RESOLUTION)
    worker.setup_if_needed()  # What the colocalization tab does when it is set up again

    assert worker.setup_finalised
    assert worker.colocalization_channel('channel_2').voxel_dims == RESOLUTION


# ----------------------------------------------------------------------------------------------------------------
# The stitching worker: it prepares the raw data of all the channels, it stitches the tiled ones
# ----------------------------------------------------------------------------------------------------------------

def _write_stack(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(path, np.zeros(STACK_SHAPE, dtype=np.uint16), photometric='minisblack')
    return path


def test_raw_data_of_a_pre_stitched_channel_is_prepared_without_the_stitching_config(experiment, channel_editor,
                                                                                    tmp_path):
    """No tiled channel: no stitching section, but the pre-stitched channel is imported all the same"""
    stack = _write_stack(tmp_path / RAW_FOLDER_NAME / 'pre_stitched.tif')
    channel_editor.add('test', DATA_TYPE)
    experiment.apply_ui_patch({'sample': {'channels': {'test': {'path': str(stack)}}}})
    assert 'stitching' not in experiment.cfg_coordinator.active_sections

    stitching_worker = experiment.get_worker('stitching')
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        stitching_worker.setup_if_needed()
        prepared = stitching_worker.prepare_all_channels_raw_data()

    assert prepared == ['test']
    assert experiment.sample_manager.get('stitched', channel='test').exists
    assert not stitching_worker.setup_complete  # Nothing to stitch
    assert [str(w.message) for w in caught] == []  # Nothing to warn about either


def test_stitching_worker_is_set_up_when_a_channel_is_tiled(experiment, channel_editor, tmp_path):
    tiles_folder = tmp_path / RAW_FOLDER_NAME / 'tiles'
    for y in range(2):
        for x in range(2):
            _write_stack(tiles_folder / f'tile_Y{y:02d}_X{x:02d}.tif')
    channel_editor.add('test', DATA_TYPE)
    experiment.apply_ui_patch({'sample': {'channels': {'test': {'path': str(tiles_folder / TILE_EXPRESSION)}}}})

    stitching_worker = experiment.get_worker('stitching')
    stitching_worker.setup_if_needed()

    assert 'stitching' in experiment.cfg_coordinator.active_sections
    assert stitching_worker.channels == ['test']
    assert stitching_worker.setup_complete


# ----------------------------------------------------------------------------------------------------------------
# The vasculature worker
# ----------------------------------------------------------------------------------------------------------------

def _no_free_v_ram() -> int:
    return 0


def test_deep_fill_raises_when_the_vram_is_insufficient(experiment, channel_editor, monkeypatch):
    """Headless, no dialog: the worker raises, the caller decides (the GUI offers to retry)"""
    channel_editor.add('vessels', 'vessels', with_path=True)
    worker = experiment.get_worker('vasculature', substep='binary')
    worker.setup_if_needed()
    monkeypatch.setattr(tube_map_module, 'get_free_v_ram', _no_free_v_ram)

    with pytest.raises(ClearMapVRamException, match='"vessels"'):
        worker.deep_fill_channel('vessels')


# ----------------------------------------------------------------------------------------------------------------
# The config: the templates of the defaults stay in the defaults
# ----------------------------------------------------------------------------------------------------------------

def _templates_reservoirs_in(sections: dict) -> list[str]:
    """The templates reservoirs (of the defaults) found in these config sections, as dotted paths"""
    return ['.'.join(spec.abs_templates_path()) for spec in INSTANCE_SPECS_REGISTRY.specs
            if spec.abs_templates_path() and get_nested(sections, spec.abs_templates_path()) is not None]


def test_templates_of_the_defaults_never_reach_a_new_experiment(experiment, tmp_path):
    """
    The templates only live in the defaults, where the templates resolver reads them. A new experiment, and
    the sections its channels bring in, are written without them, so the sanitizer has nothing to remove.
    """
    experiment_dir = tmp_path / 'new_experiment'
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        experiment.boot_new(experiment_dir)
        channel_editor = ChannelEditor(experiment, experiment_dir / RAW_FOLDER_NAME)
        for name, data_type in CHANNELS_OF_EVERY_PIPELINE.items():
            channel_editor.add(name, data_type, with_path=True)

    sanitized = [str(w.message) for w in caught if TEMPLATES_SANITIZER_WARNING in str(w.message)]
    assert sanitized == []

    working_config = experiment.cfg_coordinator.get_config_view()
    assert _templates_reservoirs_in(working_config) == []
    sections = {spec.section for spec in INSTANCE_SPECS_REGISTRY.specs}
    experiment_files = ConfigRepository(base_dir=experiment_dir, known_names=sections).load_all()
    assert _templates_reservoirs_in(experiment_files) == []

    # The templates were used all the same, from the defaults, for the instances of the channels
    assert {'nuclei_a', 'nuclei_b'} <= set(working_config['cell_map']['channels'])
    assert 'myelin' in working_config['tract_map']['channels']
