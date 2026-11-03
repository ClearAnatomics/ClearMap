"""
install_gui.sh records the elastix path it installed in the machine parameters of the user configuration folder,
which is ~/.clearmap unless CLEARMAP_CFG_DIR says otherwise.
"""
import pytest

from ClearMap.Utils import install_utils


@pytest.fixture
def cfg_dir(tmp_path, monkeypatch):
    folder = tmp_path / 'cfg_dir'
    folder.mkdir()
    (folder / 'machine_params_v3_0.cfg').write_text("elastix_path = '/old/elastix'\n")
    monkeypatch.setattr(install_utils.clearmap_settings, 'clearmap_cfg_dir', folder)
    return folder


def test_set_elastix_path_writes_the_machine_parameters_of_the_configuration_folder(cfg_dir):
    install_utils.set_elastix_path('/new/elastix')

    assert (cfg_dir / 'machine_params_v3_0.cfg').read_text().strip() == "elastix_path = '/new/elastix'"
