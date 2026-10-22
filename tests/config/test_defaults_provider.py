from pathlib import Path
import pytest

from ClearMap.config.defaults_provider import DefaultsProvider


def test_defaults_provider_provides_minimal_sections(src_dir: Path, sections):
    defaults_provider = DefaultsProvider(user_dir=Path(src_dir))

    # basic smoke: each default config is a dict with some required top-level keys
    for section in sections:
        cfg = defaults_provider.get_default_config(section)

        # Type check
        assert isinstance(cfg, dict), f'default for {section} should be a dict'

        assert cfg != {}, f'default for {section} should not be empty'

        # Test a few keys expected to be present
        match section:
            case 'sample':
                assert 'templates' in cfg and isinstance(cfg['templates']['channel'], dict)
                assert 'path' in cfg['templates']['channel'].keys()
                orientation = cfg['templates']['channel']['orientation']
                assert isinstance(orientation, list)
            case 'stitching':
                assert 'templates' in cfg and isinstance(cfg['templates'], dict)
                layout = cfg['templates']['layout_template']
                assert layout['rigid'] and isinstance(layout['rigid'], dict)
                assert layout['wobbly'] and isinstance(layout['wobbly'], dict)
                valid_range = layout['wobbly']['stack_valid_range']
                assert valid_range and isinstance(valid_range, list) and len(valid_range) == 2 and isinstance(valid_range[0], int) and isinstance(valid_range[0], int)
                assert 'rigid' not in cfg['templates']['derived_template']
            case 'registration':
                assert 'atlas' in cfg and isinstance(cfg['atlas'], dict)
                assert 'aligned_with_atlas' in cfg['templates']
                res = cfg['templates']['aligned_with_atlas']['resampled_resolution']
                assert isinstance(res, list) and len(res) == 3 and all(isinstance(r, (int, float)) for r in res)
            case 'cell_map':
                assert 'templates' in cfg and 'channel' in cfg['templates']
                assert isinstance(cfg['templates']['channel']['detection']['shape_detection']['threshold'], (int, float))
            case 'vasculature':
                assert 'binarization' in cfg
                assert 'graph_construction' in cfg


def test_defaults_provider_user_file_roundtrip(tmp_path, src_dir):
    defaults_provider = DefaultsProvider(user_dir=Path(src_dir))
    # Writes defaults if file missing, then reads them
    out_path = tmp_path / 'sample.yml'
    defaults_provider.ensure_user_file('sample', out_path)
    assert out_path.exists(), 'ensure_user_file should create the file if missing'

    # Should equal what get_default_config returns (modulo ordering)
    expected = defaults_provider.get_default_config('sample')

    # Not necessary to test all keys, just a few representative ones
    cfg = defaults_provider.get('sample')
    minimum_keys = {'channels', 'sample_id', 'comments', 'clearmap_version'}
    assert set(cfg.keys()) >= minimum_keys
    assert cfg['templates']['channel']['orientation'] == expected['templates']['channel']['orientation']


@pytest.fixture()
def new_home(tmp_path, monkeypatch):
    """A new home: the user has no global config (machine, display) yet"""
    from ClearMap.config.config_handler import ConfigHandler
    monkeypatch.setattr(ConfigHandler, 'user_global_dir', tmp_path / 'config')
    monkeypatch.setattr(ConfigHandler, 'user_defaults_dir', tmp_path)
    return tmp_path


@pytest.mark.parametrize('global_section', ['machine', 'display'])
def test_global_section_on_a_new_home_is_copied_from_the_package_and_used(new_home, global_section):
    """The user config is a copy of the package one (like /etc vs ~/.config), created if missing"""
    from ClearMap.config.config_handler import ConfigHandler
    user_path = ConfigHandler.get_global_canonical_path(global_section)
    assert not user_path.exists()

    cfg = DefaultsProvider().get(global_section)

    package_cfg = dict(ConfigHandler.get_cfg_from_path(ConfigHandler.get_default_path(global_section,
                                                                                      from_package=True)))
    assert user_path.exists()
    assert dict(ConfigHandler.get_cfg_from_path(user_path)) == package_cfg == cfg


def test_global_section_edited_by_the_user_is_not_overwritten(new_home):
    from ClearMap.config.config_handler import ConfigHandler
    ConfigHandler.dump(ConfigHandler.get_global_canonical_path('machine'), {'verbosity': 'debug'})

    cfg = DefaultsProvider().get('machine')

    assert cfg == {'verbosity': 'debug'}
    assert dict(ConfigHandler.get_cfg_from_path(ConfigHandler.get_global_canonical_path('machine'))) == cfg
