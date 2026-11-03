from copy import deepcopy
from dataclasses import dataclass
import pytest

from ClearMap.config.config_adjusters.patch_ops import DELETE, deep_merge
from ClearMap.config.compound_keys import PairKey
from ClearMap.config.config_adjusters.adjusters_api import AdjusterScope, Phase
from ClearMap.config.config_adjusters.engine import run_adjusters
from ClearMap.config.config_adjusters.policy import _fix_channel_backreferences, _registration_state_before_patch
from ClearMap.config.config_adjusters.type_hints import AdjustmentContext
from ClearMap.config.defaults_provider import DefaultsProvider, get_defaults_provider, set_defaults_provider
from ClearMap.config.registration_channels import default_reference_channel, registration_pairs

REQUIRED_SECTIONS = frozenset({'sample', 'stitching', 'registration', 'cell_map', 'tract_map', 'colocalization',
                               'vasculature'})


# convenience function for tests
def apply_patch(base_view, patch):
    base = deepcopy(base_view)
    deep_merge(base, patch)
    return base


def _template(section: str, name: str) -> dict:
    """A template of the defaults of the package (what the adjusters expand)"""
    return get_defaults_provider().get(section)['templates'][name]


# --- Fake SampleManager ------------------------------------------------------------

@dataclass
class _FakeSampleManager:
    chs: list[str]
    stitchable: list[str] | None = None
    types: dict[str, str] | None = None
    renamed: dict[str, str] | None = None
    ref: str | None = None
    detect: list[str] | None = None
    colox_ok: bool = True
    required_sections: frozenset[str] = REQUIRED_SECTIONS

    @property
    def channels(self): return self.chs

    @property
    def stitchable_channels(self): return self.stitchable or self.chs

    @property
    def renamed_channels(self): return self.renamed or {}

    def get_channels_by_pipeline(self, pipeline: str, as_list: bool = False):
        # Simple pass-through: use all channels for all pipelines in tests
        return list(self.chs)

    def data_type(self, channel: str):
        return (self.types or {}).get(channel)

    @property
    def reference_channel_setting(self):  # As the registration config would say: ref, else the default of the types
        return self.ref if self.ref is not None else default_reference_channel({c: self.data_type(c) for c in self.chs})

    @property
    def alignment_reference_channel(self): return self.ref

    @property
    def is_colocalization_compatible(self): return self.colox_ok

    @property
    def channels_to_detect(self): return list(self.detect or self.chs)

    def colocalization_pairs(self, *, oriented: bool = False):
        src = list(self.channels_to_detect)
        out = []
        for i in range(len(src)):
            for j in range(i + 1, len(src)):
                a, b = src[i], src[j]
                out.append((a, b) if oriented else tuple(sorted((a, b))))
        return out

    def colocalization_pair_keys(self, *, oriented: bool = False):
        return [str(PairKey(a, b, oriented=oriented)) for a, b in self.colocalization_pairs(oriented=oriented)]

    def compute_required_sections(self) -> set[str]:
        return set(self.required_sections)

# --- The defaults of the package (the templates the adjusters expand) --------------

@pytest.fixture(scope='module', autouse=True)
def _package_defaults_provider(tmp_path_factory):
    """
    The real defaults of the package, not a copy: a copy goes stale when the templates change.
    The user defaults folder is an empty one, so that the user's ~/.clearmap is not read.
    """
    user_dir = tmp_path_factory.mktemp('empty_user_defaults')
    old = set_defaults_provider(DefaultsProvider(user_dir=user_dir))
    try:
        yield
    finally:
        set_defaults_provider(old)


def _adjust(view, sm, sections=None):
    """Run the PRE_VALIDATE adjusters of `sections` (all if None) for the sample manager `sm`"""
    ctx = AdjustmentContext(scope=AdjusterScope.EXPERIMENT, sample_manager=sm)
    return run_adjusters(view=view, ctx=ctx, phase=Phase.PRE_VALIDATE, active_sections=sections)


def _adjust_stitching(view, sm):
    return _adjust(view, sm, ('stitching',))

def _adjust_registration(view, sm):
    return _adjust(view, sm, ('registration',))

def _adjust_cell_map(view, sm):
    return _adjust(view, sm, ('cell_map',))

def _adjust_coloc(view, sm):
    return _adjust(view, sm, ('colocalization',))

def _adjust_vasc(view, sm):
    return _adjust(view, sm, ('vasculature',))

def _adjust_groups(view, sm):
    return _adjust(view, sm, ('group_analysis',))

def _adjust_batch(view, sm):
    return _adjust(view, sm, ('batch_processing',))

def _adjust_machine(view, sm):
    return _adjust(view, sm, ('machine',))

def _adjust_display(view, sm):
    return _adjust(view, sm, ('display',))


def test_populate_sample_channel_defaults_fills_missing():
    sm = _FakeSampleManager(['Ch488'])
    view = {'sample': {'channels': {'Ch488': {'data_type': 'nuclei'}}}}
    patch = _adjust(view, sm)
    patch = apply_patch(view, patch)
    filled = patch['sample']['channels']['Ch488']
    template = _template('sample', 'channel')
    assert filled['orientation'] == template['orientation']
    assert filled['resolution'] == template['resolution']
    # user-provided survives
    assert filled['data_type'] == 'nuclei' != template['data_type']

def test_stitching_materialize_layout_channels():
    """Two stitchable channels: each one is its own layout, with the full layout template"""
    sm = _FakeSampleManager(['Ch488', 'Ch561'], types={'Ch488': 'nuclei', 'Ch561': 'nuclei'})
    view = {'sample': {'channels': {'Ch488': {}, 'Ch561': {}}}, 'stitching': {'channels': {}}}
    patch = _adjust_stitching(view, sm)
    patch = apply_patch(view, patch)
    chs = patch['stitching']['channels']
    assert set(chs.keys()) == {'Ch488', 'Ch561'}
    layout_template = _template('stitching', 'layout_template')
    for channel in ('Ch488', 'Ch561'):
        assert chs[channel]['layout_channel'] == channel  # ${channel} expanded
        assert chs[channel]['rigid'] == layout_template['rigid']
        assert chs[channel]['wobbly'] == layout_template['wobbly']


@pytest.fixture()
def sm():
   return _FakeSampleManager(['Ch488'])

@pytest.fixture()
def sample_view_with_ghost():
   return {'stitching': {'channels': {'Ch488': {'layout_channel': 'Ch488'},
                                      'Ghost': {'layout_channel': 'Ghost'}}}}

def test_prune_stitching_channels_emits_replace_marker(sample_view_with_ghost, sm):
    """
    A: the adjuster should *emit* a patch that contains {'Ghost': REPLACE}.
    We only verify the patch content here.
    """
    patch = _adjust_stitching(sample_view_with_ghost, sm)
    patch = apply_patch(sample_view_with_ghost, patch)

    # Navigate to the specific subpatch
    channels_section = patch['stitching']['channels']
    assert channels_section.keys() == {'Ch488'}


def test_stitching_prunes_removed_channels(sample_view_with_ghost, sm):
    patch = _adjust_stitching(sample_view_with_ghost, sm)
    patch = apply_patch(sample_view_with_ghost, patch)
    chs_section = patch['stitching']['channels']
    assert ('Ghost' not in chs_section) or (chs_section['Ghost'] is DELETE), 'Ghost channel should be marked for deletion'

def test_registration_reconciles_channels_and_placeholders():
    sm = _FakeSampleManager(['AF', 'Ch561'], types={'AF': 'autofluorescence', 'Ch561': 'vessels'}, ref='AF')
    view = {'registration': {'channels': {}}}
    patch = _adjust_registration(view, sm)
    patch = apply_patch(view, patch)
    chs = patch['registration']['channels']

    # Channels were inserted
    assert set(chs.keys()) == {'AF', 'Ch561'}
    # The reference channel is aligned with the atlas
    assert chs['AF']['align_with'] == 'atlas' and chs['AF']['resample'] is True
    assert chs['AF']['params_files'] == _template('registration', 'aligned_with_atlas')['params_files']
    # A regular channel is aligned with the alignment reference
    assert chs['Ch561']['align_with'] == 'AF'
    assert chs['Ch561']['params_files'] == _template('registration', 'regular')['params_files']
    # The partner moves onto the channel: the atlas onto the reference, the reference onto the others
    assert chs['AF']['moving_channel'] == chs['Ch561']['moving_channel'] == 'other'
    assert registration_pairs(patch['registration']) == [('atlas', 'AF'), ('AF', 'Ch561')]

def test_cell_map_channel_template_applied():
    sm = _FakeSampleManager(['Ch488', 'Ch561'])
    view = {'cell_map': {'channels': {}}}
    patch = _adjust_cell_map(view, sm)
    patch = apply_patch(view, patch)
    chs = patch['cell_map']['channels']
    # Channels where inserted
    assert set(chs.keys()) == {'Ch488', 'Ch561'}
    #
    assert chs['Ch488']['detection'] == _template('cell_map', 'channel')['detection']


def test_vasculature_builds_channels_and_combined_present():
    sm = _FakeSampleManager(['V1', 'V2'], types={'V1': 'vessels', 'V2': 'veins'})
    view = {'vasculature': {'binarization': {}}}
    patch = _adjust_vasc(view, sm)
    patch = apply_patch(view, patch)
    binarization = patch['vasculature']['binarization']
    assert set(binarization.keys()) == {'combined', 'single_channels'}
    assert set(binarization['single_channels'].keys()) == {'V1', 'V2'}
    v1 = binarization['single_channels']['V1']
    assert 'binarize' in v1
    assert 'deep_fill' in v1

# def test_group_analysis_defaults_and_group_normalization():
#     sm = _FakeSampleManager(['A'])
#     view = {'group_analysis': {'groups': {'g1': ['a', 'a', '', 'b']}}}
#     # phase populates defaults then adjusts groups
#     patch = _adjust_groups(view, sm)
#     ga = patch['group_analysis']
#     assert 'paths' in ga
#     assert ga['groups']['g1'] == ['a', 'b']
#
# def test_batch_defaults_and_comparisons_validation():
#     sm = _FakeSampleManager(['A'])
#     view = {'batch_processing': {'groups': {'g1': ['x', 'x'], 'g2': ['y']},
#                                  'comparisons': [['g1', 'g2'], ['g1', 'g2'], ['g2','g2'], ['g2','g1'] ]}}
#     patch = _adjust_batch(view, sm)
#     bp = patch['batch_processing']
#     # comparisons deduped, keep order, drop invalid identical pairs
#     assert bp['comparisons'] == [['g1', 'g2'], ['g2', 'g1']]
#     # groups de-duplicated
#     assert bp['groups']['g1'] == ['x']

def test_machine_and_display_fill_missing():
    sm = _FakeSampleManager(['A'])
    view = {'machine': {}, 'display': {}}
    # One view each: the runner merges what the adjusters emit into the view it is given
    mpatch = _adjust_machine(deepcopy(view), sm)
    dpatch = _adjust_display(deepcopy(view), sm)
    machine_defaults = get_defaults_provider().get('machine')
    display_defaults = get_defaults_provider().get('display')
    assert mpatch['machine']['verbosity'] == machine_defaults['verbosity']
    assert dpatch['display'].keys() == display_defaults.keys()


def test_the_channel_renames_leave_the_values_which_are_not_channels_alone():
    # A channel cannot be named "other" (reserved): if a rename of it ever came through, the moving_channel "other"
    # of the other channels (a value of the config, relative to their align_with) would not be rewritten
    channels = {'fos': {'align_with': 'auto', 'moving_channel': 'other'},
                'auto': {'align_with': 'atlas', 'moving_channel': 'Other'}}
    patch = {'registration': {'channels': channels}}

    _fix_channel_backreferences({}, {'other': 'renamed', 'Other': 'renamed', 'auto': 'reference'}, patch)

    assert channels == {'fos': {'align_with': 'reference', 'moving_channel': 'other'},
                        'auto': {'align_with': 'atlas', 'moving_channel': 'Other'}}


def test_the_registration_of_before_is_compared_under_the_names_after_the_renames():
    view_before = {'sample': {'channels': {'auto': {'data_type': 'autofluorescence'}, 'fos': {'data_type': 'nuclei'}}},
                   'registration': {'reference_channel': 'auto',
                                    'channels': {'auto': {'align_with': 'atlas'}, 'fos': {'align_with': 'auto'}}}}

    before = _registration_state_before_patch(view_before, {'auto': 'af'})

    assert before.config['reference_channel'] == 'af'
    assert list(before.config['channels']) == ['af', 'fos']
    assert before.data_types == {'af': 'autofluorescence', 'fos': 'nuclei'}
    assert _registration_state_before_patch(None, {'auto': 'af'}) is None  # A loaded config: nothing to compare
