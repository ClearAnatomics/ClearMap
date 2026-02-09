from __future__ import annotations

from typing import Iterable, Mapping, Any, Callable, Dict, Optional

DEFAULT_PAIR_SEP = '-'


class PairKey:
    """
    PairKey represents a pair of atomic keys.

    Parameters
    ----------
    a, b:
        Atomic keys.
    oriented:
        If True, preserves direction (a, b) != (b, a).
        If False, canonicalizes to a stable sorted order.
    sep:
        String separator used for serialization/parsing.
    """

    __slots__ = ('_a', '_b', '_oriented', '_sep')
    SEP = DEFAULT_PAIR_SEP

    def __init__(self, a: str, b: str, *, oriented: bool = False, sep: str = DEFAULT_PAIR_SEP) -> None:
        if not isinstance(a, str) or not isinstance(b, str):
            raise TypeError('PairKey parts must be strings')
        a = a.strip()
        b = b.strip()
        if not a or not b:
            raise ValueError('PairKey parts must be non-empty strings')
        if not isinstance(sep, str) or not sep:
            raise ValueError('sep must be a non-empty string')
        if sep in a or sep in b:
            raise ValueError(f'PairKey parts must not contain separator {sep!r}')

        self._oriented = bool(oriented)
        self._sep = sep

        if self._oriented:
            self._a, self._b = a, b
        else:
            sorted_a, sorted_b = (a, b) if a <= b else (b, a)
            self._a, self._b = sorted_a, sorted_b

    @staticmethod
    def is_valid_key_str(s: object, *, sep: str = DEFAULT_PAIR_SEP) -> bool:
        if not isinstance(s, str):
            return False
        if sep not in s or s.count(sep) != 1:
            return False
        a, b = s.split(sep, 1)
        a = a.strip()
        b = b.strip()
        if not a or not b:
            return False
        if sep in a or sep in b:
            return False
        return True

    # Construct from string
    @classmethod
    def from_string(cls, s: str, *, oriented: bool = False, sep: str = DEFAULT_PAIR_SEP) -> 'PairKey':
        if not isinstance(s, str):
            raise TypeError('PairKey.from_string expects a string')
        if sep not in s:
            raise ValueError(f'Pair key {s!r} does not contain separator {sep!r}')
        if s.count(sep) != 1:
            raise ValueError(f'Pair key {s!r} must contain exactly one separator {sep!r}')
        a, b = s.split(sep, 1)
        return cls(a, b, oriented=oriented, sep=sep)

    @property
    def a(self) -> str:
        return self._a

    @property
    def b(self) -> str:
        return self._b

    @property
    def oriented(self) -> bool:
        return self._oriented

    @property
    def sep(self) -> str:
        return self._sep

    def as_tuple(self) -> tuple[str, str]:
        return self._a, self._b

    def canonical(self) -> 'PairKey':
        """
        Return an unordered/canonical representative (sorted), regardless of current orientation.
        """
        if not self._oriented:
            return self
        return PairKey(self._a, self._b, oriented=False, sep=self._sep)

    def swapped(self) -> 'PairKey':
        """
        Return a new PairKey with atoms swapped. Preserves `oriented` flag.
        """
        return PairKey(self._b, self._a, oriented=self._oriented, sep=self._sep)

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, PairKey):
            return NotImplemented
        return self._cmp_tuple() < other._cmp_tuple()

    def __le__(self, other: object) -> bool:
        if not isinstance(other, PairKey):
            return NotImplemented
        return self._cmp_tuple() <= other._cmp_tuple()

    def __gt__(self, other: object) -> bool:
        if not isinstance(other, PairKey):
            return NotImplemented
        return self._cmp_tuple() > other._cmp_tuple()

    def __ge__(self, other: object) -> bool:
        if not isinstance(other, PairKey):
            return NotImplemented
        return self._cmp_tuple() >= other._cmp_tuple()

    def __str__(self) -> str:
        return f'{self._a}{self._sep}{self._b}'

    def __repr__(self) -> str:
        orient = 'True' if self._oriented else 'False'
        return (f'PairKey(a="{self._a}", b="{self._b}", '
                f'oriented={orient}, sep="{self._sep}")')

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, PairKey):
            return False
        return (self._a, self._b, self._oriented, self._sep) == (other._a, other._b, other._oriented, other._sep)

    def __hash__(self) -> int:
        return hash((self._a, self._b, self._oriented, self._sep))

    def _cmp_tuple(self) -> tuple:
        """
        Defines canonical ordering between PairKeys for stable sorting.

        - sep included to prevent comparing incomparable serializations
        - oriented included so oriented and non-oriented keys do not collide in sort identity
        - then (a, b) lexicographic
        """
        return self._sep, self._oriented, self._a, self._b

    def validate_atoms(self, allowed: Iterable[str]) -> bool:
        s = set(allowed) if not isinstance(allowed, set) else allowed
        return self._a in s and self._b in s

    @classmethod
    def split_str(cls, key: str, oriented: bool) -> tuple[str, str]:
        return cls.from_string(key, oriented=oriented).as_tuple()

    @classmethod
    def canonical_str(cls, pair: str | tuple[str, str], oriented: bool) -> str:
        if isinstance(pair, tuple):
            pair = cls(*pair, oriented=oriented)
        else:
            pair = cls.from_string(pair, oriented=oriented)
        return str(pair)

    @staticmethod
    def _resolve_rename(name: str, rename_map: Mapping[str, str], max_hops: int) -> str:
        cur = name
        hops = 0
        while cur in rename_map and hops < max_hops:
            nxt = rename_map[cur]
            if nxt == cur:
                break
            cur = nxt
            hops += 1
        return cur

    def rename(self, rename_map: Mapping[str, str], *, max_hops: int = 32) -> 'PairKey':
        """
        Apply rename_map (old->new) transitively to both atoms.
        If oriented=False, result is re-canonicalized as usual.
        """
        a2 = self._resolve_rename(self._a, rename_map, max_hops)
        b2 = self._resolve_rename(self._b, rename_map, max_hops)
        return PairKey(a2, b2, oriented=self._oriented, sep=self._sep)

    @classmethod
    def rename_key(cls, key: str, rename_map: Mapping[str, str], *, oriented: bool, max_hops: int = 32) -> str:
        pk = cls.from_string(key, oriented=oriented, sep=cls.SEP)
        return str(pk.rename(rename_map, max_hops=max_hops))

    @classmethod
    def _map_container_keys(cls, cur: Mapping[str, Any], *, transform: Callable[[str], str],
                            keep_unknown: bool = True, strict_collision: bool = True) -> tuple[dict[str, Any], bool]:
        """
        Apply `transform` to all string keys in cur, producing a new dict.


        Parameters
        ----------
        cur
        transform
        keep_unknown
        strict_collision

        Returns
        -------

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

            if strict_collision and nk in out:  # and out[nk] is not v:
                raise ValueError(f'key transform collision: {k!r} -> {nk!r} already exists')
            out[nk] = v

        return out, changed

    @classmethod
    def normalize_container_keys(cls, cur: Mapping[str, Any], *, oriented: bool) -> tuple[dict[str, Any], bool]:
        def _t(k: str) -> str:
            if not cls.is_valid_key_str(k):
                return k
            return str(cls.from_string(k, oriented=oriented, sep=cls.SEP))
        return cls._map_container_keys(cur, transform=_t)

    @classmethod
    def rename_container_keys(cls, cur: Mapping[str, Any], rename_map: Mapping[str, str], *,
                             oriented: bool, max_hops: int = 32) -> tuple[dict[str, Any], bool]:
        def _t(k: str) -> str:
            if not cls.is_valid_key_str(k):
                return k
            pk = cls.from_string(k, oriented=oriented, sep=cls.SEP)
            return str(pk.rename(rename_map, max_hops=max_hops))
        return cls._map_container_keys(cur, transform=_t)

    @classmethod
    def prune_container_invalid_atoms(cls, container: Mapping[str, Any], *, allowed_atoms: set[str], oriented: bool) -> dict[str, Any]:
        out: Dict[str, Any] = {}
        for k, v in container.items():
            if cls.is_valid_key_str(k):
                pk = cls.from_string(k, oriented=oriented)
                if pk.validate_atoms(allowed_atoms):
                    out[str(pk)] = v
        return out

    @classmethod
    def migrate_container_payload(cls, container: Mapping[str, Any], *, new_key: str,
                                  rename_map: Mapping[str, str], oriented: bool) -> tuple[Optional[Any], bool]:
        """
        If new_key is missing, try to find a prior key in cur_map that should carry-over payload
        after channel renames. Returns (payload, carried_flag).
        """
        if (not rename_map or
                not cls.is_valid_key_str(new_key)):
            return None, False

        pk_new = cls.from_string(new_key, oriented=oriented)

        for candidate in pk_new.candidate_ancestors_from_renames(rename_map):
            ck = str(candidate)
            if ck in container:
                return container[ck], True

        return None, False

    def candidate_ancestors_from_renames(self, rename_map: Mapping[str, str]) -> list['PairKey']:
        """
        Given this key as the *new* key, propose plausible *old* keys to search for override carry-over.

        rename_map is expected to be old->new. We invert it to find old names that mapped to our atoms.

        Notes
        -----
        - For oriented=True, candidates preserve orientation (old_a -> new_a, old_b -> new_b).
        - For oriented=False, candidates are canonicalized (sorted) automatically by the constructor.


        Usage
        -----
        >>> pk = PairKey('C', 'D', oriented=True)
        >>> rename_map = {'B': 'D', 'X': 'C'}
        >>> pk.candidate_ancestors_from_renames(rename_map)
        [PairKey(a='X', b='D', oriented=True, sep='-'),
         PairKey(a='C', b='B', oriented=True, sep='-'),
         PairKey(a='X', b='B', oriented=True, sep='-')]
        """
        if not rename_map:
            return []

        inv: dict[str, list[str]] = {}
        for old, new in rename_map.items():
            inv.setdefault(new, []).append(old)

        olds_a = inv.get(self._a, [])
        olds_b = inv.get(self._b, [])

        candidates: set[PairKey] = set()

        if self._oriented:
            # Only substitute on each side; do not mix sides.
            for oa in olds_a:
                if oa != self._b:
                    candidates.add(PairKey(oa, self._b, oriented=True, sep=self._sep))
            for ob in olds_b:
                if self._a != ob:
                    candidates.add(PairKey(self._a, ob, oriented=True, sep=self._sep))
            for oa in olds_a:
                for ob in olds_b:
                    if oa != ob:
                        candidates.add(PairKey(oa, ob, oriented=True, sep=self._sep))
        else:
            # Unoriented: allow cross-combinations as well (mirrors prior behavior)
            for oa in olds_a:
                if oa != self._b:
                    candidates.add(PairKey(oa, self._b, oriented=False, sep=self._sep))
                if oa != self._a:
                    candidates.add(PairKey(oa, self._a, oriented=False, sep=self._sep))
                for ob in olds_b:
                    if oa != ob:
                        candidates.add(PairKey(oa, ob, oriented=False, sep=self._sep))

            for ob in olds_b:
                if ob != self._a:
                    candidates.add(PairKey(self._a, ob, oriented=False, sep=self._sep))
                if ob != self._b:
                    candidates.add(PairKey(self._b, ob, oriented=False, sep=self._sep))

        candidates.discard(self)
        return sorted(candidates)
