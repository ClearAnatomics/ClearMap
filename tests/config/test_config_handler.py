import pytest


@pytest.fixture()
def default_cfg():
    return {'cell_detection': {'min_radius': 2}}


@pytest.mark.parametrize('min_radius', [5, 6, 3])
def test_patch_cfg(default_cfg, min_radius):
    from ClearMap.config.config_handler import patch_cfg
    cfg = default_cfg.copy()
    cfg['cell_detection']['min_radius'] = min_radius
    patch_cfg(cfg, default_cfg)
    assert cfg['cell_detection']['min_radius'] == min_radius


# ---- Global configs (machine, display) -------------------------------------------------------------------

GLOBAL_CFG_NAME = 'machine'


@pytest.fixture()
def empty_user_global_dir(tmp_path, monkeypatch):
    """A new home: no global config yet (the application creates them on its first launch)"""
    from ClearMap.config.config_handler import ConfigHandler
    monkeypatch.setattr(ConfigHandler, 'user_global_dir', tmp_path / 'config')
    monkeypatch.setattr(ConfigHandler, 'user_defaults_dir', tmp_path)
    return tmp_path / 'config'


def test_global_path_of_a_missing_config_is_where_it_would_be_created(empty_user_global_dir):
    """get_cfg(must_exist=False) relies on a path, not on None, to tell that the file is missing"""
    from ClearMap.config.config_handler import ConfigHandler

    path = ConfigHandler.get_global_path(GLOBAL_CFG_NAME, must_exist=False)

    assert path.parent.is_relative_to(empty_user_global_dir)
    assert path.stem == GLOBAL_CFG_NAME
    assert not path.exists()


def test_global_path_of_a_missing_config_raises_when_it_must_exist(empty_user_global_dir):
    from ClearMap.config.config_handler import ConfigHandler

    with pytest.raises(FileNotFoundError, match=GLOBAL_CFG_NAME):
        ConfigHandler.get_global_path(GLOBAL_CFG_NAME, must_exist=True)


def test_missing_global_config_is_none_when_it_need_not_exist(empty_user_global_dir):
    from ClearMap.config.config_handler import ConfigHandler

    assert ConfigHandler('').get_cfg(GLOBAL_CFG_NAME, must_exist=False) is None


@pytest.mark.parametrize('must_exist', [True, False])
def test_global_path_of_an_existing_config_is_its_path(empty_user_global_dir, must_exist):
    from ClearMap.config.config_handler import ConfigHandler
    where_it_should_be = ConfigHandler.get_global_path(GLOBAL_CFG_NAME, must_exist=False)
    where_it_should_be.parent.mkdir(parents=True)
    where_it_should_be.write_text('verbosity: debug\n')

    assert ConfigHandler.get_global_path(GLOBAL_CFG_NAME, must_exist=must_exist) == where_it_should_be


@pytest.mark.parametrize('must_exist', [True, False])
def test_global_path_finds_a_config_of_the_legacy_folder(empty_user_global_dir, must_exist):
    """Before, the global configs were in the user defaults folder: they must not be shadowed by a new one"""
    from ClearMap.config.config_handler import ConfigHandler
    where_it_should_be = ConfigHandler.get_global_path(GLOBAL_CFG_NAME, must_exist=False)
    legacy_path = ConfigHandler.user_defaults_dir / where_it_should_be.relative_to(empty_user_global_dir)
    legacy_path.parent.mkdir(parents=True)
    legacy_path.write_text('verbosity: debug\n')

    path = ConfigHandler.get_global_path(GLOBAL_CFG_NAME, must_exist=must_exist)

    assert path == legacy_path != where_it_should_be


def test_global_path_of_a_missing_config_follows_base_dir_and_version(empty_user_global_dir, tmp_path):
    """The path where it should be is the one of the folder and the version asked for, not the user's"""
    from ClearMap.config.config_handler import ConfigHandler
    other_dir = tmp_path / 'other'

    path = ConfigHandler.get_global_path(GLOBAL_CFG_NAME, must_exist=False, base_dir=other_dir, version='3.0')

    assert path.is_relative_to(other_dir)
    assert path == ConfigHandler.get_global_canonical_path(GLOBAL_CFG_NAME, root=other_dir, version='3.0')
    assert path != ConfigHandler.get_global_canonical_path(GLOBAL_CFG_NAME)


# ---- Copy of a default config to the user folder ---------------------------------------------------------

YAML_WITH_COMMENT = 'verbosity: debug  # Documents the parameter for the user\n'


def test_copy_if_missing_copies_the_file_as_is_with_its_comments(tmp_path):
    from ClearMap.config.config_handler import ConfigHandler
    src = tmp_path / 'package' / 'machine.yml'
    src.parent.mkdir()
    src.write_text(YAML_WITH_COMMENT)
    dst = tmp_path / 'user' / 'sub' / 'machine.yml'

    assert ConfigHandler.copy_if_missing(src, dst) == dst

    assert dst.read_text() == YAML_WITH_COMMENT
    assert not list(dst.parent.glob('*.tmp'))


def test_copy_if_missing_never_overwrites_the_edits_of_the_user(tmp_path):
    from ClearMap.config.config_handler import ConfigHandler
    src = tmp_path / 'machine.yml'
    src.write_text(YAML_WITH_COMMENT)
    dst = tmp_path / 'user_machine.yml'
    dst.write_text('verbosity: quiet\n')

    ConfigHandler.copy_if_missing(src, dst)

    assert dst.read_text() == 'verbosity: quiet\n'


def test_copy_if_missing_converts_to_the_format_of_the_destination(tmp_path):
    import json
    from ClearMap.config.config_handler import ConfigHandler
    src = tmp_path / 'machine.yml'
    src.write_text(YAML_WITH_COMMENT)
    dst = tmp_path / 'machine.json'

    ConfigHandler.copy_if_missing(src, dst)

    assert json.loads(dst.read_text()) == {'verbosity': 'debug'}


# ---- Copy of a default config to the user folder ---------------------------------------------------------

YAML_WITH_COMMENT = 'verbosity: debug  # Documents the parameter for the user\n'


def test_copy_if_missing_copies_the_file_as_is_with_its_comments(tmp_path):
    from ClearMap.config.config_handler import ConfigHandler
    src = tmp_path / 'package' / 'machine.yml'
    src.parent.mkdir()
    src.write_text(YAML_WITH_COMMENT)
    dst = tmp_path / 'user' / 'sub' / 'machine.yml'

    assert ConfigHandler.copy_if_missing(src, dst) == dst

    assert dst.read_text() == YAML_WITH_COMMENT
    assert not list(dst.parent.glob('*.tmp'))


def test_copy_if_missing_never_overwrites_the_edits_of_the_user(tmp_path):
    from ClearMap.config.config_handler import ConfigHandler
    src = tmp_path / 'machine.yml'
    src.write_text(YAML_WITH_COMMENT)
    dst = tmp_path / 'user_machine.yml'
    dst.write_text('verbosity: quiet\n')

    ConfigHandler.copy_if_missing(src, dst)

    assert dst.read_text() == 'verbosity: quiet\n'


def test_copy_if_missing_converts_to_the_format_of_the_destination(tmp_path):
    import json
    from ClearMap.config.config_handler import ConfigHandler
    src = tmp_path / 'machine.yml'
    src.write_text(YAML_WITH_COMMENT)
    dst = tmp_path / 'machine.json'

    ConfigHandler.copy_if_missing(src, dst)

    assert json.loads(dst.read_text()) == {'verbosity': 'debug'}
