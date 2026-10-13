"""Fake experiment with two nuclei channels, to test the colocalization tab of the GUI without real data.

Usage, from the root of the repository::

    python -m tests.fake_experiment <root>

then open ``<root>/colocalisation_v3_1`` in the GUI. Or, in a test, ``make_fake_experiment(tmp_path)``
(the user level config is the one of the machine unless the test isolates it, see
``tests/pipeline_orchestrators/conftest.py``).

The experiment is created like "Load default config" does, and the channels are set through the same patches as
the Channels sub-tab, so the sample folder contains the sample, registration, cell_map, colocalization...
config files next to the data of each channel. Open ``root / SAMPLE_FOLDER_NAME`` in ClearMap.
"""
from pathlib import Path

import numpy as np
import tifffile

SAMPLE_FOLDER_NAME = 'colocalisation_v3_1'
CHANNEL_NAMES = ('channel_2', 'channel_3')
N_PLANES = 5
PLANE_SHAPE_YX = (64, 64)


RESOLUTION = (1.625, 1.625, 3.0)
DATA_TYPE = 'nuclei'


def make_fake_experiment(root, channel_names=CHANNEL_NAMES, n_planes=N_PLANES, plane_shape_yx=PLANE_SHAPE_YX):
    """
    Create a new experiment (the sample folder is ``root / SAMPLE_FOLDER_NAME``) with one nuclei channel per name,
    each pointing to a folder of small z-plane tifs in the sample folder. The experiment writes the config files
    in the sample folder.

    Parameters
    ----------
    root : str | Path
        The folder which will contain the sample folder (created if missing).
    channel_names : Sequence[str]
        The channels to create (one folder each).
    n_planes : int
        The number of z planes (one tif each).
    plane_shape_yx : tuple[int, int]
        The shape of each plane.

    Returns
    -------
    dict[str, Path]
        The folder of each channel (already set as its path in the config).
    """
    from ClearMap.Utils.event_bus import EventBus
    from ClearMap.pipeline_orchestrators.experiment_controller import ExperimentController
    from ClearMap.pipeline_orchestrators.sample_info_management import build_sample_manager

    sample_dir = Path(root) / SAMPLE_FOLDER_NAME
    sample_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    folders = {}
    for name in channel_names:
        folder = sample_dir / name
        folder.mkdir(parents=True, exist_ok=True)
        for z in range(n_planes):  # One tif per z plane, like a non-tiled acquisition
            tifffile.imwrite(folder / f'{name}_Z{z:04d}.tif', rng.integers(0, 2000, plane_shape_yx, dtype=np.uint16))
        folders[name] = folder

    bus = EventBus()
    sample_manager = build_sample_manager(bus=bus)
    experiment = ExperimentController(cfg_coordinator=sample_manager.cfg_coordinator,
                                      sample_manager=sample_manager, evt_bus=bus)
    experiment.set_experiment_dir(sample_dir)
    experiment.boot_new()  # What "Load default config" does
    for name, folder in folders.items():  # One edit at a time, like the Channels sub-tab
        for channel_patch in ({}, {'data_type': DATA_TYPE}, {'path': str(folder)}, {'resolution': list(RESOLUTION)}):
            experiment.apply_ui_patch({'sample': {'channels': {name: channel_patch}}})
    return folders


if __name__ == '__main__':
    import sys

    for channel, channel_folder in make_fake_experiment(sys.argv[1]).items():
        print(f'{channel}: {channel_folder}')
