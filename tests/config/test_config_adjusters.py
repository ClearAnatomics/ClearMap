from copy import deepcopy
from dataclasses import dataclass
import pytest

from ClearMap.Utils.utilities import DELETE, deep_merge, _REPLACE
from ClearMap.config.config_adjusters import (
    set_defaults_provider, run_adjusters, Phase, _split_compound, _canonical_pair_key,
    _get_compound_key_candidate_ancestors, _normalize_compound_config, _prune_compound_members, _migrate_compound_entry,
)

# convenience function for tests
def apply_patch(base_view, patch):
    base = deepcopy(base_view)
    deep_merge(base, patch)
    return base

@pytest.mark.parametrize("input_, expected", [
    ("A-B", ["A", "B"]),
])
def test_split_compound(input_, expected):
    assert _split_compound(input_) == expected

@pytest.mark.parametrize("input_, exc_type", [
    ("AB", ValueError),             # no '-' present
    ("A-B-C", NotImplementedError) # multiple '-' present
])
def test_split_compound_errors(input_, exc_type):
    with pytest.raises(exc_type):
        _split_compound(input_)


@pytest.mark.parametrize("input_, expected", [
    ("B-A", "A-B"),                   # unsorted -> sorted
    (("B", "A"), "A-B"),              # tuple input
    ("A-A", "A-A"),                   # note: self-pair preserved (docstring fixed)
    ("Alpha-Beta", "Alpha-Beta"),
])
def test_canonical_pair_key(input_, expected):
    assert _canonical_pair_key(input_) == expected

# @pytest.mark.parametrize()
# def test_canonical_pair_key():
#     pass


@pytest.mark.parametrize(
    "new_key, rename_map, expected",
    [
        (
            "Ch561-Ch640",
            {"Ch488": "Ch561", "Ch561": "Ch640"},
            [
                "Ch488-Ch640",
                "Ch488-Ch561",
            ],
        ),
        (
            "Ch561-Ch640",
            {},
            []),
        (
            "Ch561-Ch640",
            {"Ch488": "Ch561"},
            [
                "Ch488-Ch640",
                "Ch488-Ch561",
            ],
        ),
    ],
)
def test_get_compound_key_candidate_ancestors(new_key, rename_map, expected):
    result = _get_compound_key_candidate_ancestors(new_key, rename_map)
    assert result == sorted(expected)


@pytest.mark.parametrize("current, expected_norm, expected_moved", [
    ({"B-A": 1, "A-A": 2, "X": 3},
     {"A-B": 1, "A-A": 2, "X": 3}, True),

    ({"A-B": 10},
     {"A-B": 10}, False),

    # Unknown shapes should pass through and be pruned elsewhere if needed
    ({"not_a_pair": {"k": 1}},
     {"not_a_pair": {"k": 1}}, False),
])
def test_normalize_compound_config(current, expected_norm, expected_moved):
    norm, moved = _normalize_compound_config(current)
    assert norm == expected_norm
    assert moved is expected_moved


@pytest.mark.parametrize("current, sample_set, expected", [
    ({"A-B": 1, "A-C": 2, "X": 3}, {"A", "B"}, {"A-B": 1}),
    ({"A-A": 9, "B-C": 8}, {"A", "B", "C"}, {"A-A": 9, "B-C": 8}),
    ({}, {"A"}, {}),
])
def test_prune_compound_members(current, sample_set, expected):
    assert _prune_compound_members(current, sample_set) == expected


def test_migrate_compound_entry_carries_first_present_candidate_verbatim():
    current_channels_cfg = {
        'Ch488-Ch561': {'x': 1},
        'Ch488-Ch640': {'y': 2},
        'Ch561-Ch640': {'z': 3},
    }
    rename_map = {'Ch488': 'Ch561', 'Ch561': 'Ch640'}
    carried, carried_flag = _migrate_compound_entry(
        current_cfg=current_channels_cfg,
        new_compound_chan='Ch561-Ch640',
        rename_map=rename_map,
    )
    # The candidate order for ('Ch561-Ch640') per fixed function:
    # ['Ch488-Ch561', 'Ch488-Ch640', 'Ch561-Ch640']
    # First present key is 'Ch488-Ch561'
    assert carried_flag is True
    assert carried == current_channels_cfg['Ch488-Ch561']  # verbatim

def test_migrate_compound_entry_no_ancestor_means_no_migration():
    current_cfg = {'A-B': 7}
    carried, carried_flag = _migrate_compound_entry(
        current_cfg=current_cfg,
        new_compound_chan='A-B',
        rename_map={'X': 'X2'},      # irrelevant mapping
    )

    assert carried is None
    assert carried_flag is False

def test_migrate_compound_entry_carries_dict_verbatim_no_mutation():
    payload = {'somefield_1': {'x': 1}, 'somefield_2': 2}
    current_cfg = {'A_old-B': payload}
    carried, flag = _migrate_compound_entry(
        current_cfg=current_cfg,
        new_compound_chan='A-B',
        rename_map={'A_old': 'A'},
    )
    assert flag is True
    assert carried is payload  # verbatim

def test_migrate_compound_entry_no_rename_map():
    current_cfg = {'A-B': {'k': 1}}
    carried, carried_flag = _migrate_compound_entry(
        current_cfg=current_cfg,
        new_compound_chan='A-B',
        rename_map=None,
    )
    assert carried_flag is False
    assert carried is None


class _FakeDefaultsProvider:
    """
    Just enough structure for adjusters to run.
    """
    def __init__(self):
        self._store = {
            "sample": {
                "channel_template": {
                    # populate_sample_channel_defaults fills these if missing
                    "orientation": [0, 0, 0],
                    "resolution": [1.0, 1.0, 1.0],
                    "data_type": "nuclei",
                }
            },
            "stitching": {
                "templates": {
                    # layout channel owns the layout
                    "layout": {
                        "layout_channel": "${channel}",
                        "rigid": True,
                        "wobbly": False,
                        "path_template": "raw/${channel}",
                    },
                    # derived channel points at a reference layout
                    "derived": {
                        "layout_channel": "${reference}",
                        "rigid": False,
                        "wobbly": True,
                        "path_template": "derived/${channel}_from_${reference}",
                    },
                }
            },
            "registration": {
                "templates": {
                    "autofluorescence": {
                        "moving": "${reference}",
                        "fixed": "${reference}",
                        "resample": True,
                    },
                    "regular": {
                        "moving": "${reference}",
                        "fixed": "${reference}",
                        "resample": False,
                    },
                }
            },
            "cell_map": {
                "templates": {
                    "channel": {
                        "detection": {
                            'colocalization_compatible': False,
                            'background_correction': {'diameter': [10, 10]}
                        },
                    }
                }
            },
            'tract_map': {
                "templates": {
                    "channel": {
                        'steps': {
                            'binarize': True,
                            'extract_coordinates': True,
                            'transform_coordinates': True,
                            'label_coordinates': True,
                            'voxelize': True,
                            'export_df': True,
                        }
                    }
                }
            },
            "colocalization": {
                "templates": {
                    "channel": {
                        "comparison": {'particle_diameter': 30},
                        'analysis': {'relative_overlap_threshold': 0.5}
                    }
                }
            },
            "vasculature": {
                "binarization": {
                    "vessels": {
                        "binarize": {"run": True, "clip_range": [350, 1000], "threshold": None},
                        "smooth": {"run": True},
                        "binary_fill": {"run": True},
                        "deep_fill": {"run": True, "size_max": 500, "overlap": 50, "resample_factor": 1},
                    },
                    "large_vessels": {
                        "binarize": {"run": True, "clip_range": [1000, 8000], "threshold": 450},
                        "smooth": {"run": True},
                        "binary_fill": {"run": True},
                        "deep_fill": {"run": True, "size_max": 1000, "overlap": 100, "resample_factor": 2},
                    },
                    "combined": { "binary_fill": True, "compress": False },
                },
                "graph_construction": {
                    "skeletonize": True, "build": True, "clean": True,
                    "reduce": True, "transform": True, "annotate": True, "use_arteries": True,
                },
                "vessel_type_postprocessing": {},
                "visualization": {},
            },
            "group_analysis": {
                "paths": {"root": "group_out"}
            },
            "batch_processing": {
                "paths": {"root": "batch_out"}
            },
            "machine": {
                "temp_folder": None,
                "n_processes": -1,
            },
            "display": {
                "colormap": "gray",
                "gamma": 1.0,
            },
        }

    def get(self, section: str):
        return deepcopy(self._store.get(section, {}))

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
    def alignment_reference_channel(self): return self.ref

    @property
    def is_colocalization_compatible(self): return self.colox_ok

    @property
    def channels_to_detect(self): return list(self.detect or self.chs)

# --- Module-scoped injection of defaults provider ---------------------------------

@pytest.fixture(scope="module", autouse=True)
def _inject_defaults_provider():
    old = set_defaults_provider(_FakeDefaultsProvider())
    try:
        yield
    finally:
        # leave the fake in place or restore if you keep a global elsewhere
        set_defaults_provider(old)

def _adjust_stitching(view, sm):
    return run_adjusters(view=view, sample_manager=sm, phase=Phase.PRE_VALIDATE, active_sections=("stitching",))

def _adjust_registration(view, sm):
    return run_adjusters(view=view, sample_manager=sm, phase=Phase.PRE_VALIDATE, active_sections=("registration",))

def _adjust_cell_map(view, sm):
    return run_adjusters(view=view, sample_manager=sm, phase=Phase.PRE_VALIDATE, active_sections=("cell_map",))

def _adjust_coloc(view, sm):
    return run_adjusters(view=view, sample_manager=sm, phase=Phase.PRE_VALIDATE, active_sections=("colocalization",))

def _adjust_vasc(view, sm):
    return run_adjusters(view=view, sample_manager=sm, phase=Phase.PRE_VALIDATE, active_sections=("vasculature",))

def _adjust_groups(view, sm):
    return run_adjusters(view=view, sample_manager=sm, phase=Phase.PRE_VALIDATE, active_sections=("group_analysis",))

def _adjust_batch(view, sm):
    return run_adjusters(view=view, sample_manager=sm, phase=Phase.PRE_VALIDATE, active_sections=("batch_processing",))

def _adjust_machine(view, sm):
    return run_adjusters(view=view, sample_manager=sm, phase=Phase.PRE_VALIDATE, active_sections=("machine",))

def _adjust_display(view, sm):
    return run_adjusters(view=view, sample_manager=sm, phase=Phase.PRE_VALIDATE, active_sections=("display",))


def test_populate_sample_channel_defaults_fills_missing():
    sm = _FakeSampleManager(["Ch488"])
    view = {"sample": {"channels": {"Ch488": {"data_type": "nuclei"}}}}
    patch = run_adjusters(view=view, sample_manager=sm, phase=Phase.PRE_VALIDATE, active_sections=None)
    patch = apply_patch(view, patch)
    filled = patch["sample"]["channels"]["Ch488"]
    assert filled["orientation"] == [0, 0, 0]
    assert filled["resolution"] == [1.0, 1.0, 1.0]
    # user-provided survives
    assert filled["data_type"] == "nuclei"

def test_stitching_materialize_layout_then_derived():
    sm = _FakeSampleManager(["Ch488", "Ch561"], types={"Ch488": "nuclei", "Ch561": "nuclei"})
    view = {"sample": {"channels": {"Ch488": {}, "Ch561": {}}}, "stitching": {"channels": {}}}
    patch = _adjust_stitching(view, sm)
    patch = apply_patch(view, patch)
    chs = patch["stitching"]["channels"]
    assert set(chs.keys()) == {"Ch488", "Ch561"}
    # one becomes layout of itself, the other derived from it
    assert chs["Ch488"]["layout_channel"] == "Ch488"
    assert chs["Ch561"]["layout_channel"] in {"Ch488", "Ch561"}
    # templates expanded
    assert "path_template" in chs["Ch561"]

@pytest.fixture()
def sm():
   return _FakeSampleManager(["Ch488"])

@pytest.fixture()
def sample_view_with_ghost():
   return {"stitching": {"channels": {"Ch488": {"layout_channel": "Ch488"},
                                      "Ghost": {"layout_channel": "Ghost"}}}}

def test_prune_stitching_channels_emits_delete_marker(sample_view_with_ghost, sm):
    """
    A: the adjuster should *emit* a patch that contains {'Ghost': DELETE}.
    We only verify the patch content here.
    """
    patch = _adjust_stitching(sample_view_with_ghost, sm)
    patch = apply_patch(sample_view_with_ghost, patch)

    # Navigate to the specific subpatch
    channels_section = patch['stitching']['channels']
    assert channels_section.keys() == {'Ch488'}
    # assert "Ghost" in channels_section, "DELETE marker should be present for Ghost"
    # assert channels_section["Ghost"] is DELETE, "Ghost must be marked with the DELETE sentinel"


def test_stitching_prunes_removed_channels(sample_view_with_ghost, sm):
    patch = _adjust_stitching(sample_view_with_ghost, sm)
    patch = apply_patch(sample_view_with_ghost, patch)
    chs_section = patch["stitching"]["channels"]
    assert ('Ghost' not in chs_section) or (chs_section['Ghost'] is DELETE), 'Ghost channel should be marked for deletion'

def test_registration_reconciles_channels_and_placeholders():
    sm = _FakeSampleManager(["AF", "Ch561"], types={"AF": "autofluorescence", "Ch561": "vessels"}, ref="AF")
    view = {"registration": {"channels": {}}}
    patch = _adjust_registration(view, sm)
    patch = apply_patch(view, patch)
    chs = patch["registration"]["channels"]

    # Channels where inserted
    assert set(chs.keys()) == {"AF", "Ch561"}
    # AF uses atlas & mask true
    assert chs["AF"]["fixed"] == "atlas" and chs["AF"]["resample"] is True
    # regular uses provided alignment reference
    assert chs["Ch561"]["fixed"] == "AF"

def test_cell_map_channel_template_applied():
    sm = _FakeSampleManager(["Ch488", "Ch561"])
    view = {"cell_map": {"channels": {}}}
    patch = _adjust_cell_map(view, sm)
    patch = apply_patch(view, patch)
    chs = patch["cell_map"]["channels"]
    # Channels where inserted
    assert set(chs.keys()) == {"Ch488", "Ch561"}
    #
    assert chs["Ch488"]["detection"]['colocalization_compatible'] is False
    assert chs["Ch488"]["detection"]['background_correction']['diameter'] == [10, 10]


# "templates": {
#     "channel": {
#         "comparison": {'particle_diameter': 30},
#         'analysis': {'relative_overlap_threshold': 0.5}
#     }

def test_colocalization_pairs_and_rename_migration():
    sm = _FakeSampleManager(['A', 'B'], detect=['A', 'B'], renamed={'old_A': 'A'})
    # existing override using old pair name
    view = {'colocalization': {'channels': {'old_A-B': {'comparison': {'particle_diameter': 25}}}}}
    patch = _adjust_coloc(view, sm)
    patch = apply_patch(view, patch)
    chs = patch['colocalization']['channels']
    # canonical key
    assert set(chs.keys()) == {'A-B'}
    assert chs['A-B']['channels'] == ['A', 'B']
    # override migrated (metric carried)
    assert chs['A-B']['comparison']['particle_diameter'] == 25  # FIXME: is 30 (default) or 25 (override) what we expect

def test_vasculature_builds_channels_and_combined_present():
    sm = _FakeSampleManager(["V1", "V2"], types={"V1": "vessels", "V2": "veins"})
    view = {"vasculature": {"binarization": {}}}
    patch = _adjust_vasc(view, sm)
    patch = apply_patch(view, patch)
    binz = patch["vasculature"]["binarization"]
    assert set(binz.keys()) == {"combined", "V1", "V2"}
    assert "binarize" in binz["V1"]
    assert "binarize" in binz["V2"]

def test_vasculature_apply_renames():
    sm = _FakeSampleManager(["V2"], renamed={"V1": "V2"})
    view = {"vasculature": {"binarization": {"V1": {"binary_fill":{"run": True}},
                                             "combined": {"binary_fill": True}}}}
    patch = _adjust_vasc(view, sm)  # reconcile will also seed section keys
    # run rename step explicitly too (same phase/pipeline)
    view1 = apply_patch(view, patch)
    patch2 = run_adjusters(view=view1, sample_manager=sm, phase=Phase.PRE_VALIDATE, active_sections=("vasculature",))
    merged = apply_patch(view1, patch2)
    chs = merged["vasculature"]["binarization"]
    assert "V1" not in chs and "V2" in chs

def test_group_analysis_defaults_and_group_normalization():
    sm = _FakeSampleManager(["A"])
    view = {"group_analysis": {"groups": {"g1": ["a", "a", "", "b"]}}}
    # phase populates defaults then adjusts groups
    patch = _adjust_groups(view, sm)
    ga = patch["group_analysis"]
    assert "paths" in ga
    assert ga["groups"]["g1"] == ["a", "b"]

def test_batch_defaults_and_comparisons_validation():
    sm = _FakeSampleManager(["A"])
    view = {"batch_processing": {"groups": {"g1": ["x", "x"], "g2": ["y"]},
                                 "comparisons": [["g1", "g2"], ["g1", "g2"], ["g2","g2"], ["g2","g1"] ]}}
    patch = _adjust_batch(view, sm)
    bp = patch["batch_processing"]
    # comparisons deduped, keep order, drop invalid identical pairs
    assert bp["comparisons"] == [["g1", "g2"], ["g2", "g1"]]
    # groups de-duplicated
    assert bp["groups"]["g1"] == ["x"]

def test_machine_and_display_fill_missing():
    sm = _FakeSampleManager(["A"])
    view = {"machine": {}, "display": {}}
    mpatch = _adjust_machine(view, sm)
    dpatch = _adjust_display(view, sm)
    assert mpatch["machine"]["n_processes"] == -1
    assert dpatch["display"]["colormap"] == "gray"
