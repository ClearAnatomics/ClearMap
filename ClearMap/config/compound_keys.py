# -*- coding: utf-8 -*-
"""
compound_keys
=============

Hashable, serialisable keys built from two or more atomic string parts,
with optional orientation and built-in rename / container-migration helpers.

Two concrete classes are provided:

* :class:`CompoundKey` — N ≥ 2 parts, general case.
* :class:`PairKey`     — exactly 2 parts; preferred API for pairs.
  ``CompoundKey.from_string`` / ``from_tuple`` auto-promote to ``PairKey``
  when the result has exactly 2 parts.

Serialisation
-------------
Both classes serialise to ``'part0<sep>part1<sep>…'`` (default ``'-'``).
Unoriented keys are stored in sorted (canonical) order so that
``PairKey('B', 'A') == PairKey('A', 'B')``.

Typical use
-----------
>>> from ClearMap.config.compound_keys import PairKey, CompoundKey
>>> k = PairKey('cfos', 'dapi')
>>> str(k)
'cfos-dapi'
>>> k2 = PairKey.from_string('cfos-dapi')
>>> k == k2
True
"""
from __future__ import annotations

import itertools
from typing import Iterable, Mapping, Any, Callable, Optional

DEFAULT_PAIR_SEP = '-'


class CompoundKey:
    """
    An immutable, hashable key composed of N ≥ 2 atomic string parts.

    For the common two-part case prefer :class:`PairKey`, which adds
    ``.a`` / ``.b`` accessors and pair-specific container helpers.
    ``CompoundKey.from_string`` and ``from_tuple`` return a ``PairKey``
    automatically when the result has exactly two parts.

    Orientation
    -----------
    ``oriented=False`` (default)
        Parts are sorted on construction so that key identity is
        independent of argument order.
    ``oriented=True``
        Part order is preserved; ``('A', 'B') != ('B', 'A')``.

    Parameters
    ----------
    *parts : str
        Two or more atomic string parts.
    oriented : bool
        Whether part order is significant.
    sep : str
        Separator used for string serialisation and parsing.
        Defaults to ``DEFAULT_PAIR_SEP``
    """

    SEP: str = DEFAULT_PAIR_SEP

    __slots__ = ('_parts', '_oriented', '_sep')

    _min_parts: int = 2
    _max_parts: int | None = None   # None = unlimited

    def __init__(self, *parts: str,
                 oriented: bool = False, sep: str = DEFAULT_PAIR_SEP) -> None:
        if not isinstance(sep, str) or not sep:
            raise ValueError('sep must be a non-empty string')
        parts = tuple(p.strip() for p in parts)
        n = len(parts)
        if n < self._min_parts:
            raise ValueError(f'{type(self).__name__} needs ≥{self._min_parts} parts, got {n}')
        if self._max_parts is not None and n != self._max_parts:
            raise ValueError(f'{type(self).__name__} needs exactly {self._max_parts} parts, got {n}')
        if any(not p for p in parts):
            raise ValueError('Parts must be non-empty strings')
        if any(sep in p for p in parts):
            raise ValueError(f'Parts must not contain separator {sep!r}')
        self._oriented = bool(oriented)
        self._sep = sep
        self._parts = parts if oriented else tuple(sorted(parts))

    # ── constructors ─────────────────────────────────────────────────────

    @classmethod
    def from_tuple(cls, parts: tuple[str, ...], *,
                   oriented: bool = False,
                   sep: str = DEFAULT_PAIR_SEP) -> 'CompoundKey | PairKey':
        """
        Construct from an existing tuple of parts.

        Automatically returns a :class:`PairKey` when called on
        ``CompoundKey`` and ``len(parts) == 2``.

        Parameters
        ----------
        parts : tuple[str, ...]
            Atomic parts.
        oriented : bool
            Whether part order is significant.
        sep : str
            Separator for serialisation.

        Returns
        -------
        CompoundKey | PairKey
        """
        if cls is CompoundKey and len(parts) == 2:
            return PairKey(*parts, oriented=oriented, sep=sep)
        return cls(*parts, oriented=oriented, sep=sep)

    @classmethod
    def from_string(cls, s: str, *,
                    oriented: bool = False,
                    sep: str = DEFAULT_PAIR_SEP) -> 'CompoundKey | PairKey':
        """
        Parse a serialised key string such as ``'cfos-dapi'``.

        Automatically returns a :class:`PairKey` when called on
        ``CompoundKey`` and the string contains exactly one separator.

        Parameters
        ----------
        s : str
            Serialised key, e.g. ``'a-b'`` or ``'a-b-c'``.
        oriented : bool
            Whether part order is significant.
        sep : str
            Expected separator character(s).

        Returns
        -------
        CompoundKey | PairKey

        Raises
        ------
        ValueError
            If the string yields fewer parts than ``_min_parts`` or a
            number inconsistent with ``_max_parts``.
        """
        parts = tuple(p.strip() for p in s.split(sep))
        if len(parts) < cls._min_parts:
            raise ValueError(f'Expected at least {cls._min_parts} parts, got {len(parts)}')
        if cls._max_parts is not None and len(parts) != cls._max_parts:
            raise ValueError(f'Expected exactly {cls._max_parts} parts, got {len(parts)}')
        if cls is CompoundKey and len(parts) == 2:
            return PairKey(*parts, oriented=oriented, sep=sep)
        return cls(*parts, oriented=oriented, sep=sep)

    # ── properties ───────────────────────────────────────────────────────

    @property
    def parts(self) -> tuple[str, ...]:
        """All parts as an immutable tuple."""
        return self._parts

    @property
    def n(self) -> int:
        """Number of parts."""
        return len(self._parts)

    @property
    def oriented(self) -> bool:
        """Whether part order is significant for equality and hashing."""
        return self._oriented

    @property
    def sep(self) -> str:
        """Separator used for string serialisation."""
        return self._sep

    # ── dunder ───────────────────────────────────────────────────────────

    def __str__(self) -> str:
        return self._sep.join(self._parts)

    def __repr__(self) -> str:
        parts_str = ', '.join(f'"{p}"' for p in self._parts)
        return f'{type(self).__name__}({parts_str}, oriented={self._oriented}, sep="{self._sep}")'

    def __eq__(self, other: object) -> bool:
        return isinstance(other, CompoundKey) and self._cmp_tuple() == other._cmp_tuple()

    def __hash__(self) -> int:
        return hash((self._parts, self._oriented, self._sep))

    def _cmp_tuple(self) -> tuple:
        """
        Canonical sort key used by all comparison operators

        - sep included to prevent comparing incomparable serializations
        - oriented included so oriented and non-oriented keys do not collide in sort identity
        - then parts (a, b, ...) lexicographic
        """
        return self._sep, self._oriented, self._parts

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, CompoundKey):
            return NotImplemented
        else:
            return self._cmp_tuple() < other._cmp_tuple()

    def __le__(self, other: object) -> bool:
        if not isinstance(other, CompoundKey):
            return NotImplemented
        else:
            return self._cmp_tuple() <= other._cmp_tuple()

    def __gt__(self, other: object) -> bool:
        if not isinstance(other, CompoundKey):
            return NotImplemented
        else:
            return self._cmp_tuple() > other._cmp_tuple()

    def __ge__(self, other: object) -> bool:
        if not isinstance(other, CompoundKey):
            return NotImplemented
        else:
            return self._cmp_tuple() >= other._cmp_tuple()

    # ── validation ───────────────────────────────────────────────────────

    def validate_parts(self, allowed: Iterable[str]) -> bool:
        """
        Return ``True`` if every part of this key is in *allowed*.

        Parameters
        ----------
        allowed : Iterable[str]
            The set of permitted atomic values.
        """
        s = set(allowed)
        return all(p in s for p in self._parts)

    validate_atoms = validate_parts   # backward-compat alias

    @classmethod
    def is_valid_key_str(cls, s: object, *, sep: str = DEFAULT_PAIR_SEP) -> bool:
        """
        Return ``True`` if *s* is a string that can be parsed by this class.

        Checks separator count against ``_min_parts`` / ``_max_parts`` and
        ensures no part is empty or itself contains the separator.

        Parameters
        ----------
        s : object
            Candidate string.
        sep : str
            Separator to split on.
        """
        if not isinstance(s, str):
            return False
        parts = [p.strip() for p in s.split(sep)]
        n = len(parts)
        if n < cls._min_parts:
            return False
        if cls._max_parts is not None and n != cls._max_parts:
            return False
        return all(parts) and all(sep not in p for p in parts)

    # ── conversions ──────────────────────────────────────────────────────

    def as_tuple(self) -> tuple[str, ...]:
        """Return parts as a plain tuple (same object as stored internally)."""
        return self._parts

    def canonical(self) -> 'CompoundKey':
        """
        Return the unoriented (sorted) form of this key.

        If the key is already unoriented, returns ``self`` unchanged.
        The returned object has the same concrete type (``PairKey``
        subclass is preserved via ``type(self)``).
        """
        if not self._oriented:
            return self
        return type(self)(*self._parts, oriented=False, sep=self._sep)

    def swapped(self) -> 'CompoundKey':
        """
        Return a new key with parts in reversed order.

        For a :class:`PairKey` this swaps ``a`` and ``b``.
        For longer keys parts are fully reversed.
        The ``oriented`` flag is preserved.
        """
        return type(self)(*reversed(self._parts), oriented=self._oriented, sep=self._sep)

    @classmethod
    def split_str(cls, key: str, oriented: bool) -> tuple[str, ...]:
        """
        Parse *key* and return its parts as a tuple.

        Convenience wrapper around :meth:`from_string` + :meth:`as_tuple`.

        Parameters
        ----------
        key : str
            Serialised key string.
        oriented : bool
            Whether to treat the key as oriented.
        """
        return cls.from_string(key, oriented=oriented).as_tuple()

    @classmethod
    def canonical_str(cls, key: str | tuple[str, ...], oriented: bool) -> str:
        """
        Return the canonical (sorted) string representation of *key*.

        Accepts either a serialised string or a raw tuple of parts.

        Parameters
        ----------
        key : str | tuple[str, ...]
            Key to canonicalise.
        oriented : bool
            Whether to treat the key as oriented before canonicalising.
        """
        if isinstance(key, tuple):
            obj = cls(*key, oriented=oriented)
        else:
            obj = cls.from_string(key, oriented=oriented)
        return str(obj)

    # ── rename helpers ───────────────────────────────────────────────────

    @staticmethod
    def _resolve_rename(name: str, rename_map: Mapping[str, str], max_hops: int) -> str:
        """
        Starting from *name*, keep following rename_map until the name
        no longer appears as a key, or until *max_hops* steps have been
        taken.

        Example: rename_map = {'A': 'B', 'B': 'C'}
        Starting from 'A' → 'B' → 'C' → stop (C not in map).  Returns 'C'.

        The hop limit exists so that a circular mapping such as
        {'A': 'B', 'B': 'A'} does not loop forever.

        Parameters
        ----------
        name : str
            The name to resolve.
        rename_map : Mapping[str, str]
            Old → new name mapping.
        max_hops : int
            Give up after this many steps even if the chain continues.
        """
        cur = name
        hops = 0
        while cur in rename_map and hops < max_hops:
            nxt = rename_map[cur]
            if nxt == cur:
                break
            cur = nxt
            hops += 1
        return cur

    def rename(self, rename_map: Mapping[str, str], *, max_hops: int = 32) -> 'CompoundKey | PairKey':
        """
        Return a new key with each part renamed according to *rename_map*.

        Renames are applied transitively (up to *max_hops* steps).
        Unoriented keys are re-canonicalized as usual by the constructor after renaming.

        Parameters
        ----------
        rename_map : Mapping[str, str]
            Old → new name mapping.
        max_hops : int
            Maximum transitive rename steps per part.
        """
        new_parts = tuple(self._resolve_rename(p, rename_map, max_hops) for p in self._parts)
        return type(self)(*new_parts, oriented=self._oriented, sep=self._sep)

    @classmethod
    def rename_key(cls, key: str, rename_map: Mapping[str, str], *,
                   oriented: bool, max_hops: int = 32) -> str:
        """
        Parse *key*, apply *rename_map* to its parts, and return the
        serialised result.

        Parameters
        ----------
        key : str
            Serialised key to rename.
        rename_map : Mapping[str, str]
            Old → new name mapping.
        oriented : bool
            Whether the key should be treated as oriented.
        max_hops : int
            Maximum transitive rename steps per part.
        """
        obj = cls.from_string(key, oriented=oriented, sep=cls.SEP)
        return str(obj.rename(rename_map, max_hops=max_hops))

    # ── container key operations ─────────────────────────────────────────

    @classmethod
    def _map_container_keys(cls, cur: Mapping[str, Any], *, transform: Callable[[str], str],
                            keep_unknown: bool = True, strict_collision: bool = True) -> tuple[dict[str, Any], bool]:
        """
        Apply *transform* to every compound-key string in *cur*, returning
        a new dict and a flag indicating whether any key actually changed.

        Non-string keys and strings that do not pass :meth:`is_valid_key_str`
        are either kept verbatim (``keep_unknown=True``) or dropped.

        Parameters
        ----------
        cur : Mapping[str, Any]
            Source mapping.
        transform : Callable[[str], str]
            Function applied to each valid compound-key string.
        keep_unknown : bool
            If ``True``, keys that are not valid compound-key strings are
            copied unchanged.  If ``False`` they are dropped.
        strict_collision : bool
            If ``True``, raise ``ValueError`` when two distinct source keys
            map to the same transformed key.

        Returns
        -------
        tuple[dict[str, Any], bool]
            ``(new_dict, changed)`` where *changed* is ``True`` if at least
            one key was modified.
        """
        out: dict[str, Any] = {}
        changed = False
        for k, v in (cur or {}).items():
            if not isinstance(k, str):
                if strict_collision and k in out:
                    raise ValueError(f'duplicate non-string key: {k!r}')
                out[k] = v
                continue
            if cls.is_valid_key_str(k):
                nk = transform(k)
            else:
                if not keep_unknown:
                    continue
                nk = k
            changed = changed or (nk != k)
            if strict_collision and nk in out:
                raise ValueError(f'key transform collision: {k!r} → {nk!r} already exists')
            out[nk] = v
        return out, changed

    @classmethod
    def normalize_container_keys(cls, cur: Mapping[str, Any], *,
                                  oriented: bool) -> tuple[dict[str, Any], bool]:
        """
        Re-serialise every valid compound-key string in *cur* to its
        canonical form (sorted for unoriented keys).

        Parameters
        ----------
        cur : Mapping[str, Any]
            Source mapping whose keys may include compound-key strings.
        oriented : bool
            Whether keys should be treated as oriented.

        Returns
        -------
        tuple[dict[str, Any], bool]
            ``(normalised_dict, changed)``.
        """
        def transform(k: str) -> str:
            if not cls.is_valid_key_str(k):
                return k
            return str(cls.from_string(k, oriented=oriented, sep=cls.SEP))
        return cls._map_container_keys(cur, transform=transform)

    @classmethod
    def rename_container_keys(cls, cur: Mapping[str, Any], rename_map: Mapping[str, str], *,
                               oriented: bool, max_hops: int = 32) -> tuple[dict[str, Any], bool]:
        """
        Apply *rename_map* to the parts of every compound-key string in *cur*.

        Parameters
        ----------
        cur : Mapping[str, Any]
            Source mapping.
        rename_map : Mapping[str, str]
            Old → new name mapping.
        oriented : bool
            Whether keys should be treated as oriented.
        max_hops : int
            Maximum transitive rename steps per part.

        Returns
        -------
        tuple[dict[str, Any], bool]
            ``(renamed_dict, changed)``.
        """
        def transform(k: str) -> str:
            if not cls.is_valid_key_str(k):
                return k
            cmpnd_key = cls.from_string(k, oriented=oriented, sep=cls.SEP)
            return str(cmpnd_key.rename(rename_map, max_hops=max_hops))
        return cls._map_container_keys(cur, transform=transform)

    @classmethod
    def prune_container_invalid_atoms(cls, container: Mapping[str, Any], *,
                                       allowed_atoms: set[str], oriented: bool) -> dict[str, Any]:
        """
        Remove entries whose compound-key string contains a part not in
        *allowed_atoms*.

        Parameters
        ----------
        container : Mapping[str, Any]
            Source mapping to filter.
        allowed_atoms : set[str]
            The set of permitted atomic part values.
        oriented : bool
            Whether keys should be treated as oriented.

        Returns
        -------
        dict[str, Any]
            Filtered copy of *container*.
        """
        return {str(cmpnd_key): v
                for k, v in container.items()
                if cls.is_valid_key_str(k)
                and (cmpnd_key := cls.from_string(k, oriented=oriented)).validate_parts(allowed_atoms)}

    @classmethod
    def migrate_container_payload(cls, container: Mapping[str, Any], *, new_key: str,
                                   rename_map: Mapping[str, str], oriented: bool) -> tuple[Optional[Any], bool]:
        """
        Look up a stored value for *new_key*, falling back to any plausible
        predecessor key produced by :meth:`candidate_ancestors_from_renames`.

        Use this when a channel was renamed and the config entry was written
        under the old key: the new key will not be found directly, but a
        predecessor key will be.

        Parameters
        ----------
        container : Mapping[str, Any]
            The config mapping to search.
        new_key : str
            The compound-key string under which a value is desired.
        rename_map : Mapping[str, str]
            Old → new name mapping used to derive candidate predecessors.
        oriented : bool
            Whether keys should be treated as oriented.

        Returns
        -------
        tuple[Optional[Any], bool]
            ``(value, True)`` if a predecessor key was found;
            ``(None, False)`` otherwise.
        """
        if not rename_map or not cls.is_valid_key_str(new_key):
            return None, False
        ancestors = cls.from_string(new_key, oriented=oriented).candidate_ancestors_from_renames(rename_map)
        for candidate in ancestors:
            if (ck := str(candidate)) in container:
                return container[ck], True
        return None, False

    def candidate_ancestors_from_renames(self, rename_map: Mapping[str, str]) -> list['CompoundKey']:
        """
        Return every key that this key *could have been* before a rename.

        Background
        ----------
        Keys are stored in config under their name at the time of writing.
        When channels are renamed, the stored key becomes stale.  This
        method generates all plausible old keys so callers can look them
        up and carry the stored value forward.

        Algorithm
        ---------
        Given rename_map {"old": "new", …}, invert it to find what each
        current part *used to be called*.  Then take every combination of
        (current-or-predecessor) across all positions, discarding
        combos that equal self (nothing was renamed) or contain the same
        value twice (illegal key).

        For oriented keys position order is preserved.
        For unoriented keys the constructor sorts parts into canonical form.

        Parameters
        ----------
        rename_map : Mapping[str, str]
            Maps old name → new name.  Only direct (single-hop) renames
            are considered.

        Examples
        --------
        **PairKey, oriented — two renames, one per side:**

        >>> PairKey('C', 'D', oriented=True).candidate_ancestors_from_renames({'A': 'C', 'B': 'D'})
        [PairKey(a='A', b='B', oriented=True, sep='-'),
         PairKey(a='A', b='D', oriented=True, sep='-'),
         PairKey(a='C', b='B', oriented=True, sep='-')]

        Position 0 was C, or A (which was renamed to C).
        Position 1 was D, or B (which was renamed to D).
        All combinations except self ('C','D') are returned.

        **PairKey, unoriented — same rename map:**

        >>> PairKey('C', 'D', oriented=False).candidate_ancestors_from_renames({'A': 'C', 'B': 'D'})
        [PairKey(a='A', b='B', oriented=False, sep='-'),
         PairKey(a='A', b='D', oriented=False, sep='-'),
         PairKey(a='B', b='C', oriented=False, sep='-')]

        Same combinations, but each pair is sorted by the constructor:
        ('C','B') becomes 'B-C'.

        **PairKey — rename does not touch either part:**

        >>> PairKey('C', 'D', oriented=True).candidate_ancestors_from_renames({'X': 'Y'})
        []

        **CompoundKey triplet, oriented:**

        >>> CompoundKey('C', 'D', 'E', oriented=True).candidate_ancestors_from_renames(
        ...     {'A': 'C', 'B': 'E'})
        [CompoundKey("A", "D", "B", oriented=True, sep="-"),
         CompoundKey("A", "D", "E", oriented=True, sep="-"),
         CompoundKey("C", "D", "B", oriented=True, sep="-")]

        Position 0 options: {C, A}.  Position 1 options: {D} (no rename).
        Position 2 options: {E, B}.  Self ('C','D','E') is discarded.

        **CompoundKey triplet, unoriented:**

        >>> CompoundKey('C', 'D', 'E', oriented=False).candidate_ancestors_from_renames(
        ...     {'A': 'C', 'B': 'E'})
        [CompoundKey("A", "D", "B", oriented=False, sep="-"),
         CompoundKey("A", "D", "E", oriented=False, sep="-"),
         CompoundKey("B", "C", "D", oriented=False, sep="-")]

        Same combinations; each triplet is sorted by the constructor:
        ('C','D','B') becomes ('B','C','D').
        """
        if not rename_map:
            return []

        # inverted lookup dict
        inv: dict[str, list[str]] = {}
        for old, new in rename_map.items():
            inv.setdefault(new, []).append(old)

        # For each position collect the current value plus every name that
        # was renamed *to* that value (one step back in history).
        # Example: current part 'C', rename_map had {'A': 'C', 'E': 'C'}
        #   → slot = ('C', 'A', 'E')
        options = [tuple({p} | set(inv.get(p, []))) for p in self._parts]

        candidates: set[CompoundKey] = set()
        for combo in itertools.product(*options):
            if len(set(combo)) != len(combo):   # invalid (duplicate)
                continue
            if combo == self._parts:            # invalid (self)
                continue
            try:
                candidates.add(type(self)(*combo,
                                          oriented=self._oriented,
                                          sep=self._sep))
            except ValueError:
                pass

        return sorted(candidates)


class PairKey(CompoundKey):
    """
    :class:`CompoundKey` specialised for exactly two atomic parts.

    :class:`CompoundKey` class-methods (``from_string``, ``from_tuple``)
    automatically return a ``PairKey`` when the result has two parts, so
    callers rarely need to instantiate this class directly.

    All rename / container helpers defined on :class:`CompoundKey` work
    unchanged; ``PairKey`` only adds ``.a`` / ``.b`` property accessors
    and narrows return-type annotations.

    Parameters
    ----------
    a, b : str
        The two atomic parts.
    oriented : bool
        If ``True``, ``(a, b) != (b, a)``.
        If ``False`` (default), parts are sorted so order does not matter.
    sep : str
        Separator used for string serialisation.
    """

    __slots__ = ('_a', '_b', '_oriented', '_sep')
    _min_parts: int = 2
    _max_parts: int = 2   # enforces exactly-2 in is_valid_key_str and from_string
    SEP: str = DEFAULT_PAIR_SEP

    def __init__(self, a: str, b: str, *, oriented: bool = False, sep: str = DEFAULT_PAIR_SEP) -> None:
        super().__init__(a, b, oriented=oriented, sep=sep)

    # ── pair accessors ───────────────────────────────────────────────────

    @property
    def a(self) -> str:
        """First part (lower of the two for unoriented keys)."""
        return self._parts[0]

    @property
    def b(self) -> str:
        """Second part (higher of the two for unoriented keys)."""
        return self._parts[1]

    # ── narrowed return types ─────────────────────────────────────────────

    def as_tuple(self) -> tuple[str, str]:
        """Return ``(a, b)`` as a two-element tuple."""
        return self._parts[0], self._parts[1]

    def canonical(self) -> 'PairKey':
        """Return the unoriented (sorted) form; type is preserved as ``PairKey``."""
        return super().canonical()  # type(self) in super already returns PairKey

    def swapped(self) -> 'PairKey':
        """Return a new ``PairKey`` with ``a`` and ``b`` exchanged."""
        return super().swapped()

    # ── repr ─────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        return (f'PairKey(a="{self.a}", b="{self.b}", '
                f'oriented={self._oriented}, sep="{self._sep}")')