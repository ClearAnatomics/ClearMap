"""
Fixtures to run the orchestrators headless, on a real new experiment.

Nothing is mocked: the event bus, the config coordinator, the sample manager and the experiment controller
are the ones the GUI uses, only Qt is absent. The experiment is created in an empty folder from the default
configuration ("Load default config" in the GUI), and edited through the same patches as the GUI
(:meth:`~ClearMap.pipeline_orchestrators.experiment_controller.ExperimentController.apply_ui_patch`).
"""
from pathlib import Path
from typing import Iterator, Optional

import pytest

from ClearMap.Utils.event_bus import EventBus
from ClearMap.config.config_handler import ConfigHandler
from ClearMap.config.update_config import update_default_config
from ClearMap.pipeline_orchestrators.experiment_controller import ExperimentController
from ClearMap.pipeline_orchestrators.sample_info_management import build_sample_manager

RAW_FOLDER_NAME = 'raw_data'
ATLAS_CACHE_NAME = 'clearmap_atlas_cache'  # In the pytest cache (.pytest_cache/d/)


@pytest.fixture(scope='session', autouse=True)
def isolated_user_config(tmp_path_factory, pytestconfig) -> Iterator[Path]:
    """
    Never read (or write) the real ``~/.clearmap``: the tests must not depend on the machine they run on,
    nor leave anything in the home of whoever runs them. The user level config folders are new ones,
    initialised from the package defaults like the application does on its first launch, once per session.

    The atlas cache (reoriented atlases, about 1 GB, derived from the package atlases only) is kept in the
    pytest cache of the repository instead, so that it is built once, not on every run.
    ``pytest --cache-clear`` removes it.
    """
    user_dir = tmp_path_factory.mktemp('clearmap_user_config')
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(ConfigHandler, 'user_defaults_dir', user_dir)
        monkeypatch.setattr(ConfigHandler, 'user_global_dir', user_dir / 'config')
        update_default_config()
        _set_machine_config('atlas_cache_folder', str(pytestconfig.cache.mkdir(ATLAS_CACHE_NAME)))
        yield user_dir


def _set_machine_config(key: str, value) -> None:
    """Edit the (isolated) machine config file, which the experiments read when they are opened"""
    machine_cfg_path = ConfigHandler.get_global_canonical_path('machine')
    machine_cfg = dict(ConfigHandler.get_cfg_from_path(machine_cfg_path))
    if key not in machine_cfg:
        raise KeyError(f'Unknown machine config key "{key}" (in {machine_cfg_path})')
    machine_cfg[key] = value
    ConfigHandler.dump(machine_cfg_path, machine_cfg)


@pytest.fixture()
def experiment(tmp_path: Path) -> ExperimentController:
    """A new experiment, created in an empty folder from the default config, without any channel"""
    bus = EventBus()
    sample_manager = build_sample_manager(bus=bus)
    controller = ExperimentController(cfg_coordinator=sample_manager.cfg_coordinator,
                                      sample_manager=sample_manager, evt_bus=bus)
    controller.set_experiment_dir(tmp_path)
    controller.boot_new()
    return controller


@pytest.fixture()
def sample_manager(experiment: ExperimentController):
    return experiment.sample_manager


@pytest.fixture()
def channel_editor(experiment: ExperimentController, tmp_path: Path) -> 'ChannelEditor':
    return ChannelEditor(experiment, tmp_path / RAW_FOLDER_NAME)


class ChannelEditor:
    """What the user does in the channels sub-tab of the sample tab: one patch per edit, like the GUI"""

    def __init__(self, experiment: ExperimentController, raw_folder: Path):
        self._experiment = experiment
        self._raw_folder = raw_folder

    def add(self, name: str, data_type: Optional[str] = None, *, with_path: bool = False,
            resolution: Optional[tuple[float, float, float]] = None) -> None:
        """Create the channel, then set what is given, one edit at a time (the order of a user)"""
        self._patch(name, {})
        if data_type is not None:
            self.set_data_type(name, data_type)
        if with_path:
            self.set_path(name)
        if resolution is not None:
            self.set_resolution(name, resolution)

    def set_data_type(self, name: str, data_type: str) -> None:
        self._patch(name, {'data_type': data_type})

    def set_path(self, name: str) -> None:
        """Point the channel to an (existing, empty) folder"""
        folder = self._raw_folder / name
        folder.mkdir(parents=True, exist_ok=True)
        self._patch(name, {'path': str(folder)})

    def set_resolution(self, name: str, resolution: tuple[float, float, float]) -> None:
        self._patch(name, {'resolution': list(resolution)})

    def _patch(self, name: str, channel_patch: dict) -> None:
        self._experiment.apply_ui_patch({'sample': {'channels': {name: channel_patch}}})

