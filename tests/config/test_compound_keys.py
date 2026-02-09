from __future__ import annotations

import pytest

from ClearMap.config.compound_keys import PairKey


@pytest.mark.parametrize("input_, expected", [
    ("A-B", ("A", "B")),
])
def test_pairkey_parse(input_, expected):
    pk = PairKey.from_string(input_, oriented=False)
    assert pk.as_tuple() == expected


@pytest.mark.parametrize("input_, exc_type", [
    ("AB", ValueError),     # no '-' present
    ("A-B-C", ValueError),  # > 1 '-' present
])
def test_pairkey_parse_errors(input_, exc_type):
    with pytest.raises(exc_type):
        PairKey.from_string(input_, oriented=False)


@pytest.mark.parametrize("input_, expected", [
    ("B-A", "A-B"),                  # unordered -> canonical
    (("B", "A"), "A-B"),             # tuple input
    ("A-A", "A-A"),                  # self-pair allowed
    ("Alpha-Beta", "Alpha-Beta"),
])
def test_pairkey_canonical_str(input_, expected):
    assert PairKey.canonical_str(input_, oriented=False) == expected


@pytest.mark.parametrize("input_, expected", [
    ("B-A", "A-B"),
    ("A-A", "A-A"),
])
def test_pairkey_canonical_object(input_, expected):
    pk = PairKey.from_string(input_, oriented=False)
    assert str(pk.canonical()) == expected



@pytest.mark.parametrize(
    "new_key, rename_map, expected",
    [
        (
            "Ch561-Ch640",
            {"Ch488": "Ch561", "Ch561": "Ch640"},
            ["Ch488-Ch640", "Ch488-Ch561"],
        ),
        ("Ch561-Ch640", {}, []),
        (
            "Ch561-Ch640",
            {"Ch488": "Ch561"},
            ["Ch488-Ch640", "Ch488-Ch561"],
        ),
    ],
)
def test_pairkey_candidate_ancestors(new_key, rename_map, expected):
    pk_new = PairKey.from_string(new_key, oriented=False)
    out = pk_new.candidate_ancestors_from_renames(rename_map)
    out = sorted(str(pk) for pk in out)
    assert out == sorted(expected)


@pytest.mark.parametrize("current, expected_norm, expected_moved", [
    ({"B-A": 1, "A-A": 2, "X": 3},
     {"A-B": 1, "A-A": 2, "X": 3}, True),

    ({"A-B": 10},
     {"A-B": 10}, False),

    # Unknown shapes pass through (pruned elsewhere)
    ({"not_a_pair": {"k": 1}},
     {"not_a_pair": {"k": 1}}, False),
])
def test_pairkey_normalize_container_keys(current, expected_norm, expected_moved):
    norm, moved = PairKey.normalize_container_keys(current, oriented=False)
    assert norm == expected_norm
    assert moved is expected_moved


@pytest.mark.parametrize("current, allowed_atoms, expected", [
    ({"A-B": 1, "A-C": 2, "X": 3}, {"A", "B"}, {"A-B": 1}),
    ({"A-A": 9, "B-C": 8}, {"A", "B", "C"}, {"A-A": 9, "B-C": 8}),
    ({}, {"A"}, {}),
])
def test_pairkey_prune_container_invalid_atoms(current, allowed_atoms, expected):
    out = PairKey.prune_container_invalid_atoms(current, allowed_atoms=set(allowed_atoms), oriented=False)
    assert out == expected


def test_pairkey_migrate_container_payload_first_present_candidate_verbatim():
    container = {
        "Ch488-Ch561": {"x": 1},
        "Ch488-Ch640": {"y": 2},
        "Ch561-Ch640": {"z": 3},
    }
    rename_map = {"Ch488": "Ch561", "Ch561": "Ch640"}

    payload, carried = PairKey.migrate_container_payload(
        container=container,
        new_key="Ch561-Ch640",
        rename_map=rename_map,
        oriented=False,
    )

    assert carried is True
    assert payload is container["Ch488-Ch561"]  # verbatim


def test_pairkey_migrate_container_payload_no_ancestor_means_no_migration():
    container = {"A-B": 7}
    payload, carried = PairKey.migrate_container_payload(
        container=container,
        new_key="A-B",
        rename_map={"X": "X2"},
        oriented=False,
    )
    assert payload is None
    assert carried is False


def test_pairkey_migrate_container_payload_carries_dict_verbatim_no_mutation():
    payload = {"somefield_1": {"x": 1}, "somefield_2": 2}
    container = {"A_old-B": payload}

    carried_payload, carried = PairKey.migrate_container_payload(
        container=container,
        new_key="A-B",
        rename_map={"A_old": "A"},
        oriented=False,
    )

    assert carried is True
    assert carried_payload is payload  # verbatim


def test_pairkey_migrate_container_payload_no_rename_map():
    container = {"A-B": {"k": 1}}
    payload, carried = PairKey.migrate_container_payload(
        container=container,
        new_key="A-B",
        rename_map=None,
        oriented=False,
    )
    assert carried is False
    assert payload is None
