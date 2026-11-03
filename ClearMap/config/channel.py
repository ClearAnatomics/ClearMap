"""
The identity of the channels of a sample.

A channel is known by its name in the config (the key of its entries), in the file names and on screen.
The name can change (the user renames the channel), so it cannot be what the code uses to tell channels apart:
everything which keeps one thing per channel (pages, params, workers...) would have to be re-keyed by hand on
each rename, and the one forgotten shows as a stale page or a ``KeyError`` on the old name.

Here a channel is an immutable :class:`Channel` handle whose identity is a :class:`ChannelId`, minted by the
:class:`ChannelRegistry` of the sample and never changed. The name is looked up in the registry, which is the
only place where names are validated and changed. The things kept per channel are in a :class:`ChannelMap`.

.. important::
    Ids and handles are **session-local**: they are not in the config, they are minted again when an experiment
    is opened, and they refuse to be pickled. The code below the orchestrators (the worker processes) only gets
    names and paths, as the config holds them: ``{channel.name: value}`` is the way out
    (see :meth:`ChannelMap.by_name`), and :meth:`ChannelRegistry.resolve` the way back in.
"""
import re
import uuid
from collections.abc import Iterable, Iterator, Mapping, MutableMapping
from typing import NoReturn, TypeVar, Union

from ClearMap.Utils.exceptions import (ChannelNameTakenError, ClearMapTypeError, InvalidChannelIdError,
                                       InvalidChannelNameError,
                                       UnknownChannelError)
from ClearMap.config.registration_channels import ALIGN_WITH_UNDECIDED, INTRINSICALLY_ALIGNED, MOVING_OTHER, MOVING_SELF

# WARNING: Keep in sync with ``typedefs.schema.yaml#/$defs/channelName`` (the sample channel names).
#  No '-': it joins the channels of the compound keys (see compound_keys.py)
CHANNEL_NAME_RE = re.compile(r'[A-Za-z0-9_]+')

SESSION_LOCAL_MESSAGE = ('{what} is session-local and cannot be pickled or sent to another process: '
                         'send the channel name (or {{channel.name: value}}) and resolve it with the '
                         'ChannelRegistry of the receiving side')
V = TypeVar('V')


def folded_channel_name(name: str) -> str:
    """The name as compared for uniqueness: channel names are file names, which some file systems fold"""
    return name.casefold()


# WARNING: Keep in sync with ``typedefs.schema.yaml#/$defs/channelName``
#: The values which the config fields holding a channel name accept besides the names of the channels
#: (``align_with``, ``moving_channel``, ``layout_channel``, ...). A channel of one of these names would be mistaken
#: for them, and renaming it would rewrite them. Compared whatever the case, as the names are.
RESERVED_CHANNEL_NAMES = frozenset(folded_channel_name(name) for name in (
    MOVING_SELF, MOVING_OTHER, INTRINSICALLY_ALIGNED,
    ALIGN_WITH_UNDECIDED,  # The partner of a channel is not chosen yet
    'undefined',  # Not chosen yet (e.g. the layout_channel of the stitching)
    'none',  # How None is written in the GUI combo boxes (and in YAML, whatever the case)
))


def is_reserved_channel_name(value) -> bool:
    """Whether value is one of the values of the config which are not channels (see RESERVED_CHANNEL_NAMES)"""
    return isinstance(value, str) and folded_channel_name(value) in RESERVED_CHANNEL_NAMES


def _refuse_to_be_pickled(what: str) -> NoReturn:
    raise ClearMapTypeError(SESSION_LOCAL_MESSAGE.format(what=what))


#: The text of a UUID4, as ``str(uuid.uuid4())`` writes it. A channel name of the same shape is possible but
#: unlikely (names are short), so this makes the collision of a name and an id infinitesimal.
UUID4_REGEX = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$",
    re.IGNORECASE
)


class ChannelId:
    """
    The identity of a channel: a value which only means "this channel", unique in the session, never changed by a rename.

    It wraps the string of a UUID but is deliberately **not** a ``str`` (nor comparable with one), so that a type
    checker, or the first run, tells a channel id from a channel name.
    Only the :class:`ChannelRegistry` mints them (:meth:`new`).
    """
    __slots__ = ('_value',)

    def __init__(self, value: str):
        if not isinstance(value, str):
            raise ClearMapTypeError(f'A ChannelId wraps the text of a UUID4, got {type(value).__name__}: {value!r}')
        if UUID4_REGEX.fullmatch(value) is None:
            raise InvalidChannelIdError(f'{value!r} is not the text of a UUID4 (36 characters, hex digits and '
                                        f'hyphens, version 4): a channel id is not a channel name')
        self._value = value.lower()  # One text per UUID: the same id compares equal whatever the case it was read in

    @classmethod
    def new(cls) -> 'ChannelId':
        return cls(str(uuid.uuid4()))

    @property
    def value(self) -> str:
        return self._value

    def __eq__(self, other: object):
        if isinstance(other, str):
            raise ClearMapTypeError(f'A ChannelId is not comparable with a str ({other!r}): '
                            f'did you mean to resolve a channel name with ChannelRegistry.resolve?')
        if isinstance(other, ChannelId):
            return self._value == other._value
        return NotImplemented

    def __hash__(self):
        return hash(self._value)

    def __repr__(self):
        return f'ChannelId({self._value[:8]})'

    def __reduce__(self):
        _refuse_to_be_pickled('A ChannelId')


class Channel:
    """
    An immutable handle on a channel of a sample: its identity is its :class:`ChannelId`, its name is read from the
    :class:`ChannelRegistry` which created it (so it follows the renames).

    Handles are created by the registry only. Two handles are equal if they have the same id.

    .. warning::
        ``str(channel)`` is the name, for display and logs **only**: whatever is written in a config, a file name
        or a key of another structure uses ``channel.name`` explicitly (or better, a :class:`ChannelMap`).
        Comparing a channel with a name raises, so that no site silently falls back to comparing names.
    """
    __slots__ = ('_id', '_registry')

    def __init__(self, channel_id: ChannelId, registry: 'ChannelRegistry'):
        self._id = channel_id
        self._registry = registry

    @property
    def id(self) -> ChannelId:
        return self._id

    @property
    def name(self) -> str:
        """The current name of the channel. Raises :class:`UnknownChannelError` if it was removed"""
        return self._registry.name_of(self)

    def __str__(self):
        return self.name

    def __repr__(self):
        try:
            return f'Channel({self.name!r}, {self._id!r})'
        except UnknownChannelError:
            return f'Channel(<removed>, {self._id!r})'

    def __hash__(self):
        return hash(self._id)

    def __eq__(self, other: object):
        if isinstance(other, str):
            raise ClearMapTypeError(f'A Channel is not comparable with a str ({other!r}): '
                            f'resolve the name with ChannelRegistry.resolve or compare channel.name')
        if isinstance(other, Channel):
            return self._id == other._id
        return NotImplemented

    def __copy__(self):
        return self

    def __deepcopy__(self, memo):
        return self  # An immutable handle. Copying the registry with it would make another sample of the same channel

    def __reduce__(self):
        _refuse_to_be_pickled(f'The Channel {self!r}')


class ChannelRegistry:
    """
    The channels of one sample: mints their ids, and validates and changes their names.

    Names are made of letters, digits and underscores (``CHANNEL_NAME_RE``), are not reserved
    (``RESERVED_CHANNEL_NAMES``) and are unique **case insensitively** (``Ch`` and ``ch`` would be the same file on
    macOS and Windows).

    .. warning::
        Owned by the sample manager, changed only inside a config commit (the owner of the one-off side effects
        of a commit). Everything else only reads it.
    """

    def __init__(self):
        self._name_by_id: dict[ChannelId, str] = {}  # In the order the channels were added
        self._id_by_folded_name: dict[str, ChannelId] = {}

    @staticmethod
    def validated_name(name: str) -> str:
        if not isinstance(name, str) or not CHANNEL_NAME_RE.fullmatch(name):
            raise InvalidChannelNameError(f'{name!r} is not a valid channel name: '
                                          f'only letters, digits and "_" are allowed')
        if is_reserved_channel_name(name):
            raise InvalidChannelNameError(f'{name!r} is not a valid channel name: it is reserved (whatever the case) '
                                          f'for a value of the config ({", ".join(sorted(RESERVED_CHANNEL_NAMES))})')
        return name

    def add(self, name: str) -> Channel:
        """Mint the id of a new channel. Raises if the name is invalid or already used (whatever the case)"""
        self.validated_name(name)
        taken_by = self._id_by_folded_name.get(folded_channel_name(name))
        if taken_by is not None:
            raise ChannelNameTakenError(f'A channel is already named {self._name_by_id[taken_by]!r} '
                                        f'(names are unique whatever the case): cannot add {name!r}')
        channel_id = ChannelId.new()
        self._name_by_id[channel_id] = name
        self._id_by_folded_name[folded_channel_name(name)] = channel_id
        return Channel(channel_id, self)

    def rename_many(self, new_names: Mapping[Channel, str]) -> None:
        """
        Rename channels at once. The final names are checked first, then applied: a refused call changes nothing,
        and names can be swapped (``a -> b`` and ``b -> a`` in the same call).
        """
        final_names = dict(self._name_by_id)
        for channel, new_name in new_names.items():
            self._known(channel)
            final_names[channel.id] = self.validated_name(new_name)

        channel_ids_by_folded_name: dict[str, list[ChannelId]] = {}
        for channel_id, name in final_names.items():
            channel_ids_by_folded_name.setdefault(folded_channel_name(name), []).append(channel_id)
        clashes = sorted(final_names[ids[0]] for ids in channel_ids_by_folded_name.values() if len(ids) > 1)
        if clashes:
            raise ChannelNameTakenError(f'Channels would share a name (names are unique whatever the case): '
                                        f'{", ".join(map(repr, clashes))}')

        self._name_by_id = final_names
        self._id_by_folded_name = {folded_channel_name(name): channel_id for channel_id, name in final_names.items()}

    def remove(self, channel: Channel) -> None:
        self._known(channel)
        del self._id_by_folded_name[folded_channel_name(self._name_by_id.pop(channel.id))]

    def name_of(self, channel: Channel) -> str:
        self._known(channel)
        return self._name_by_id[channel.id]

    def resolve(self, name_or_channel: Union[str, Channel]) -> Channel:
        """
        The channel of this sample called ``name`` (exactly: ``'FOS'`` is not ``'fos'``). This is where the
        names enter the code which uses channels: from a config, an event from outside, a file.
        A :class:`Channel` is returned if it is one of this registry, and refused otherwise.
        """
        if isinstance(name_or_channel, Channel):
            self._known(name_or_channel)
            return name_or_channel
        if not isinstance(name_or_channel, str):
            raise ClearMapTypeError(f'Cannot resolve a channel from {name_or_channel!r}: give its name')
        return self._channel_of_name(name_or_channel)

    def has_name(self, name: str) -> bool:
        return name in self.names

    def has_id(self, channel_id: ChannelId) -> bool:
        """Whether the channel of this id is one of this registry (an id of a replaced registry is not)"""
        if not isinstance(channel_id, ChannelId):
            raise ClearMapTypeError(f'{channel_id!r} is not a ChannelId')
        return channel_id in self._name_by_id

    @property
    def names(self) -> list[str]:
        """The names of the channels, in the order they were added"""
        return list(self._name_by_id.values())

    @property
    def channels(self) -> list[Channel]:
        return list(self)

    def __contains__(self, channel: object) -> bool:
        if isinstance(channel, str):
            raise ClearMapTypeError(f'{channel!r} is a name: use has_name, or resolve it')
        return isinstance(channel, Channel) and channel._registry is self and channel.id in self._name_by_id

    def __iter__(self) -> Iterator[Channel]:
        return (Channel(channel_id, self) for channel_id in self._name_by_id)

    def __len__(self) -> int:
        return len(self._name_by_id)

    def _channel_of_name(self, name: str) -> Channel:
        channel_id = self._id_by_folded_name.get(folded_channel_name(name))
        if channel_id is None or self._name_by_id[channel_id] != name:
            raise UnknownChannelError(f'No channel called {name!r} in this sample (channels: {self.names})')
        return Channel(channel_id, self)

    def _known(self, channel: Channel) -> None:
        if not isinstance(channel, Channel):
            raise ClearMapTypeError(f'Expected a Channel, got {channel!r}')
        if channel._registry is not self or channel.id not in self._name_by_id:
            # WARNING: not repr(channel): it reads the name, from this very registry
            raise UnknownChannelError(f'{channel.id!r} is not a channel of this sample '
                                      f'(removed, or of another sample)')


class ChannelMap(MutableMapping[Channel, V]):
    """
    What is kept for each channel of a sample (params, a page, a worker...): a mapping whose **keys are**
    :class:`Channel` **s** (a name or anything else raises ``ClearMapTypeError``), so a rename moves nothing.
    The values are the things kept: widgets and params are values, never keys.

    Session-local like the channels: it refuses to be pickled. ``{channel.name: value}`` (see :meth:`by_name`)
    is what goes to the config or to another process.
    """

    def __init__(self, items: Iterable[tuple[Channel, V]] = ()):
        self._items: dict[Channel, V] = {}
        for channel, value in items:
            self[channel] = value

    @staticmethod
    def _checked(key: object) -> Channel:
        if not isinstance(key, Channel):
            raise ClearMapTypeError(f'The keys of a ChannelMap are Channels, got {type(key).__name__}: {key!r}. '
                            f'Resolve a name with ChannelRegistry.resolve')
        return key

    def __getitem__(self, key: Channel) -> V:
        return self._items[self._checked(key)]

    def __setitem__(self, key: Channel, value: V) -> None:
        self._items[self._checked(key)] = value

    def __delitem__(self, key: Channel) -> None:
        del self._items[self._checked(key)]

    def __contains__(self, key: object) -> bool:
        return self._checked(key) in self._items

    def __iter__(self) -> Iterator[Channel]:
        return iter(self._items)

    def __len__(self) -> int:
        return len(self._items)

    def __repr__(self):
        return f'ChannelMap({self._items!r})'

    def by_name(self) -> dict[str, V]:
        """The values by the current name of their channel: for the config, or to send to another process"""
        return {channel.name: value for channel, value in self._items.items()}

    def __reduce__(self):
        _refuse_to_be_pickled('A ChannelMap')
