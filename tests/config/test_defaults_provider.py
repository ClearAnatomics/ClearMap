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
                assert 'channel_template' in cfg and isinstance(cfg['channel_template'], dict)
                assert 'path' in cfg['channel_template'].keys()
                orientation = cfg['channel_template']['orientation']
                assert isinstance(orientation, list)
            case 'stitching':
                assert 'templates' in cfg and isinstance(cfg['templates'], dict)
                assert cfg['templates']['layout']['rigid'] and isinstance(cfg['templates']['layout']['rigid'], dict)
                assert cfg['templates']['layout']['wobbly'] and isinstance(cfg['templates']['layout']['wobbly'], dict)
                valid_range = cfg['templates']['layout']['wobbly']['stack_valid_range']
                assert valid_range and isinstance(valid_range, list) and len(valid_range) == 2 and isinstance(valid_range[0], int) and isinstance(valid_range[0], int)
                assert 'rigid' not in cfg['templates']['derived']
            case 'registration':
                assert 'atlas' in cfg and isinstance(cfg['atlas'], dict)
                assert 'autofluorescence' in cfg['templates']
                res = cfg['templates']['autofluorescence']['resampled_resolution']
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
    assert cfg['channel_template']['orientation'] == expected['channel_template']['orientation']
