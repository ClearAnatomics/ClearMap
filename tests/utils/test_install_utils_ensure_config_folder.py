"""
install_gui.sh asks install_utils for the folder of the user configuration (and gets it created), instead of repeating
where it is: ~/.clearmap, or CLEARMAP_CFG_DIR.
"""
from ClearMap.Utils import install_utils


def test_ensure_config_folder_creates_the_missing_folders_and_returns_the_folder(tmp_path, monkeypatch):
    folder = tmp_path / 'missing' / 'cfg_dir'
    monkeypatch.setattr(install_utils.clearmap_settings, 'clearmap_cfg_dir', folder)

    assert install_utils.ensure_config_folder() == folder
    assert folder.is_dir()


def test_ensure_config_folder_keeps_what_the_folder_holds(tmp_path, monkeypatch):
    (tmp_path / 'machine_params_v3_0.cfg').write_text("elastix_path = '/elastix'\n")
    monkeypatch.setattr(install_utils.clearmap_settings, 'clearmap_cfg_dir', tmp_path)

    install_utils.ensure_config_folder()

    assert (tmp_path / 'machine_params_v3_0.cfg').read_text() == "elastix_path = '/elastix'\n"
