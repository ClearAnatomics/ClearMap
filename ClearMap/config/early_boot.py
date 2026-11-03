"""
early_boot.py
--------------
Run this **FIRST**, before anything else (even logging).

- Reads machine params (machine_params_v<major>_<minor>.cfg) from the configuration folder (~/.clearmap, or $CLEARMAP_CFG_DIR)
- Patches tempfile’s default directory ASAP (if configured)


.. warning::

    Any module imported here should not read tempfile.tempdir to
    avoid a race condition in setting it.
"""

import os
import pathlib
import platform
import shutil
import subprocess
import tempfile
from functools import cached_property
from typing import Optional, TypedDict, Any, Dict, TypeAlias

from ClearMap import __version__
from ClearMap.config.config_handler import ConfigHandler, CLEARMAP_CFG_DIR

from ClearMap.config.utils import parse_flat_yaml, set_flat_yaml_value
from ClearMap.Utils.exceptions import ClearMapConfigError
from ClearMap.Utils.utilities import is_headless

SENTINEL_DIR = CLEARMAP_CFG_DIR / '.first_run_sentinels'

ConfigValueType: TypeAlias = str | int | float | bool | None


class MachineConfig:
    """
    Minimal, side-effect-free reader for machine config.
    Caches the parsed file; exposes DEFAULT section as a mapping.
    """
    _version = __version__  # required early for filename

    def __init__(self):
        self.min_temp_folder_gb = 5
        self._env_path = os.environ.get('CLEARMAP_MACHINE_CFG')  # optional override

    @cached_property
    def cfg_path(self) -> pathlib.Path:
        if self._env_path:
            return pathlib.Path(self._env_path).expanduser()
        return self.machine_cfg_path()

    @staticmethod
    def machine_cfg_path() -> pathlib.Path:
        return ConfigHandler.get_global_path('machine', must_exist=False)

    @staticmethod
    def _package_machine_cfg_path() -> pathlib.Path:
        return ConfigHandler.get_default_path('machine', must_exist=True, from_package=True)

    def _ensure_cfg_exists(self) -> pathlib.Path:
        """Write the packaged default machine config to the user folder."""
        user_path = self.cfg_path
        if user_path.exists():
            return user_path  # Early return if already present

        default_path = self._package_machine_cfg_path()
        if not default_path.exists():
            raise FileNotFoundError(f'Packaged default machine config not found: {default_path}.'
                                    f'Please check your ClearMap installation.')

        return ConfigHandler.copy_if_missing(default_path, user_path)

    def _patch_temp_dir_cfg(self, path_str: str):
        self._ensure_cfg_exists()
        set_flat_yaml_value(self.cfg_path, 'temp_folder', path_str)  # edit specific line only. keep comments

    def _patch_temp_dir_env(self, path_str: str):
        for var in ('TMP', 'TEMP', 'TMPDIR'):
            os.environ[var] = path_str
        tempfile.tempdir = path_str  # critical: must happen before anything else uses tempfile

    def prompt_tmp_folder(self):
        # Not at the top: a headless run imports this module and must not need Qt (nor a display)
        from PyQt5.QtWidgets import QFileDialog, QMessageBox, QInputDialog
        from ClearMap.gui.gui_utils_base import ensure_qapp

        app = ensure_qapp()

        # Plan A, proper folder using QT directory dialog
        tmp_folder = QFileDialog.getExistingDirectory(
            None, f'Select temporary folder (>={self.min_temp_folder_gb} GB free recommended)',
            str(pathlib.Path.home()),
            QFileDialog.ShowDirsOnly | QFileDialog.DontResolveSymlinks,
        )
        if tmp_folder:
            tmp_folder = pathlib.Path(tmp_folder).expanduser()
            if self._is_valid_tmp(tmp_folder):
                return str(tmp_folder)
            else:
                QMessageBox.warning(None, 'Folder not suitable',
                    f'"{tmp_folder}" is not writeable or does not have enough free space.\n'
                    'Please choose another.',
                )
        # Plan B, pure string for e.g. non existing folders (or cancel)
        text, ok = QInputDialog.getText(None, 'Enter temporary folder',
            f'No valid folder selected.\n'
            f'Enter an absolute path with ≥ {self.min_temp_folder_gb} GB free:',
        )
        if ok and text.strip():
            tmp_folder = pathlib.Path(text.strip()).expanduser()
            if self._is_valid_tmp(tmp_folder, self.min_temp_folder_gb):
                return str(tmp_folder)
            else:
                QMessageBox.critical(None, "Invalid folder",
                    f"‘{tmp_folder}’ is not writeable or does not have enough free space.",
                )
        return None

    def _read_cfg(self):
        self._ensure_cfg_exists()
        with open(self.cfg_path, 'r', encoding='utf-8') as f:
            data = f.read()
        cfg = parse_flat_yaml(data)
        return cfg

    @cached_property
    def defaults(self) -> Dict[str, Any]:
        return self._read_cfg()

    def get(self, key: str, default: ConfigValueType = None) -> str | None:
        return self.defaults.get(key, default)

    # -------- early side-effect (tempdir patch)

    def _is_valid_tmp(self, path: pathlib.Path, min_free_gb: Optional[int] = None) -> bool:
        if min_free_gb is None:
            min_free_gb = self.min_temp_folder_gb
        try:
            path.mkdir(parents=True, exist_ok=True)
            # write test_
            tst_path = path / '.clearmap_tmp_test'
            tst_path.write_text('ok', encoding='utf-8')
            tst_path.unlink(missing_ok=True)
            total, used, free = shutil.disk_usage(str(path))
            return free >= min_free_gb * (1024 ** 3)
        except Exception:
            return False

    def patch_tmp(self, *, allow_prompt: bool = True) -> str:
        """
        Set the temporary folder (the TMP, TEMP and TMPDIR variables, and tempfile.tempdir) to the first valid one of:

        1. the environment variable CLEARMAP_TMP: it belongs to this run (e.g. the node-local folder of a cluster job),
           so it comes before the machine config, which can be shared by the runs;
        2. the ``temp_folder`` of the machine config;
        3. the folder that the user chooses in a dialog (not when headless), which is saved in the machine config.

        Only the folder of the dialog is written in the machine config: the other two are read, never saved. There is
        no default folder: ClearMap writes hundreds of GB there, and the system one is often small.

        Returns
        -------
        str
            The folder.

        Raises
        ------
        ClearMapConfigError
            If no folder is valid (writeable, with enough free space) and the user cannot or does not choose one.
        """
        self._ensure_cfg_exists()

        rejected = []
        for source, raw_folder in (('CLEARMAP_TMP', os.environ.get('CLEARMAP_TMP')),
                                   ('temp_folder of the machine config', self.get('temp_folder', None))):
            if not raw_folder or raw_folder.strip().lower() == 'null':
                continue
            folder = pathlib.Path(raw_folder).expanduser()
            if self._is_valid_tmp(folder):
                self._patch_temp_dir_env(str(folder))
                return str(folder)
            rejected.append(f'{source} ("{raw_folder}")')
        rejected_msg = f' Not writeable or too little free space: {", ".join(rejected)}.' if rejected else ''

        # Plan C: prompt user
        if allow_prompt and not is_headless():
            prompt_result = self.prompt_tmp_folder()
            if prompt_result:
                self._patch_temp_dir_cfg(prompt_result)
                self._patch_temp_dir_env(prompt_result)
                return prompt_result
            raise ClearMapConfigError('A valid temporary folder is required. Startup aborted by the user.')
        raise ClearMapConfigError(
            f'No valid temporary folder.{rejected_msg} Set CLEARMAP_TMP, or temp_folder in {self.cfg_path}, to a '
            f'writeable folder with at least {self.min_temp_folder_gb} GB free (ClearMap writes much more there: '
            f'use a large, fast disk, not the system /tmp).')

    @cached_property
    def verbosity(self) -> str:
        return (self.get('verbosity', 'info')).lower()

    @cached_property
    def start_full_screen(self) -> bool:
        return self.get('start_full_screen', False)


class BootResult(TypedDict):
    first_run: bool
    desktop_entry_ok: bool
    cfg_ensured: bool
    tmp_dir: Optional[str]


def prompt_machine_params() -> bool:
    """
    Try to open the Preferences dialog so the user can fill machine params.
    Returns True if the dialog opened and was accepted, False otherwise.
    Falls back gracefully in headless/CI.
    """
    try:
        from ClearMap.gui.gui_utils_base import ensure_qapp

        app = ensure_qapp()
        if app is None:
            return False  # headless or PyQt not installed

        # Lazy imports avoid circulars on early boot
        from ClearMap.config.config_handler import ConfigHandler
        from ClearMap.gui.preferences import PreferenceUi  # uses PreferencesParams & GenericDialog under the hood

        class _MiniMainWindow:
            # Minimal surface needed by PreferenceUi
            def __init__(self):
                self.config_loader = ConfigHandler(CLEARMAP_CFG_DIR)
            def file_exists(self, p):
                from pathlib import Path
                return Path(p).expanduser().exists()
            def print_error_msg(self, msg):
                print(msg)
            def reload_prefs(self):
                pass

        mw = _MiniMainWindow()
        dlg = PreferenceUi(mw)
        dlg.setup(12)  # FIXME: event_bus + view_provider
        result = dlg.open()  # modal (returns 1 on accept)
        return bool(result)
    except Exception as e:
        # Don’t block boot in non-GUI environments
        print(f"[early_boot] Preferences dialog unavailable: {e}")
        return False


def _sentinel_path(version_tag: str) -> pathlib.Path:
    SENTINEL_DIR.mkdir(parents=True, exist_ok=True)
    return SENTINEL_DIR / f'first_run_done_{version_tag}'


def _is_first_run(version_tag: str) -> bool:
    return not _sentinel_path(version_tag).exists()


def _mark_first_run_done(version_tag: str) -> None:
    _sentinel_path(version_tag).write_text('ok', encoding='utf-8')


def _ensure_desktop_entry(icon_path: Optional[str] = None) -> bool:
    # Keep this very conservative; no errors on headless/unsupported OS.
    if is_headless():
        return False
    if platform.system().lower() != "linux":
        return False
    xdg = pathlib.Path.home() / ".local/share/applications"
    xdg.mkdir(parents=True, exist_ok=True)
    desktop_file = xdg / "clearmap.desktop"
    if desktop_file.exists():
        return True
    try:
        content = f"""[Desktop Entry]
Type=Application
Name=ClearMap
Comment=ClearMap GUI
Exec=clearmap-ui
Icon={icon_path or "clearmap"}
Terminal=false
Categories=Science;Graphics;
StartupWMClass=clearmap
"""
        desktop_file.write_text(content, encoding='utf-8')
        return True
    except Exception:
        return False

def first_boot(*, allow_prompt: bool = True) -> BootResult:
    """
    Call this at the very beginning of your entry point.
    It will:
    - ensure the per-user config exists
    - patch temp dir (may prompt unless headless or allow_prompt=False)
    - run migrations (if version changed)
    - ensure desktop entry on Linux (best-effort)
    - persist a small boot state stamp
    """
    mc = MachineConfig()

    # ALWAYS patch tmp first
    tmp_dir = mc.patch_tmp(allow_prompt=allow_prompt)  # may prompt, or use cfg/env

    major, minor = mc._version.split('.')[:2]
    version_tag = f'v{major}_{minor}'

    # TODO: move into build instead
    creator_folder = pathlib.Path(__file__).parent.parent / 'gui' / 'creator'
    qrc_path = creator_folder / 'icons.qrc'
    py_rc_path = creator_folder / 'icons_rc.py'
    if not py_rc_path.exists() or qrc_path.stat().st_mtime > py_rc_path.stat().st_mtime:
        subprocess.run(['pyrcc5', str(qrc_path), '-o', str(py_rc_path)], check=True)
    import ClearMap.gui.creator.icons_rc  # noqa: F401

    graphics_resources_qrc_path = creator_folder / 'graphics_resources.qrc'
    graphics_resources_py_rc_path = creator_folder / 'graphics_resources_rc.py'
    if (not graphics_resources_py_rc_path.exists() or
            graphics_resources_qrc_path.stat().st_mtime > graphics_resources_py_rc_path.stat().st_mtime):
        subprocess.run(['pyrcc5', graphics_resources_qrc_path, '-o', graphics_resources_py_rc_path], check=True)
    import ClearMap.gui.creator.graphics_resources_rc  # noqa: F401  (the images of the help, ':/graphics_resources/')

    is_first_run = _is_first_run(version_tag)

    # WARNING: imported after patch_tmp to avoid issues with setting tempdir
    from ClearMap.config.update_config import update_default_config

    update_default_config()
    if not is_first_run:
        # Ensure update_default_config() is idempotent and fast
        return BootResult(first_run=False, desktop_entry_ok=True, cfg_ensured=True, tmp_dir=tmp_dir)

    # migrated = _maybe_run_migrations(mc, _is_first_run(version_tab))

    # Desktop entry (best-effort; Linux only)
    desktop_ok = _ensure_desktop_entry()

    # 5) prompt user settings for first run
    # for machine parameters
    prompt_machine_params()
    # TODO: menu for performance tuning for individual pipelines

    # 6) mark done
    if is_first_run:
        _mark_first_run_done(version_tag)

    return BootResult(
        first_run=is_first_run,
        desktop_entry_ok=desktop_ok,
        cfg_ensured=True,
        tmp_dir=tmp_dir,
    )
