# -*- coding: utf-8 -*-
"""
upgrade_prompts
===============

The questions of the upgrade of an experiment (2.1 to 3.0 and later), and who answers them.

Upgrading converts the config files and renames files: it is only done with someone who confirms. The upgrade asks a
:class:`UpgradePrompter`, which has no default and never answers by itself:

- the dialogs of the GUI are in :class:`ClearMap.gui.dialogs.DialogUpgradePrompter`,
- :class:`ConsolePrompter` asks in a terminal, so an experiment can be upgraded from a shell,
- a batch job (e.g. slurm) has no one to confirm: it refuses, and the experiment must be upgraded beforehand.

This module imports no Qt.
"""
__author__ = 'Charly Rousseau <charly.rousseau@icm-institute.org>'
__license__ = 'GPLv3 - GNU General Public License v3 (see LICENSE.txt)'
__copyright__ = 'Copyright © 2022 by Charly Rousseau'
__webpage__ = 'https://idisco.info'
__download__ = 'https://github.com/ClearAnatomics/ClearMap'

import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

from ClearMap.IO.FileUtils import Rename
from ClearMap.IO.assets_constants import DATA_CONTENT_TYPES
from ClearMap.Utils.exceptions import ClearMapConfigError, ClearMapValueError
from ClearMap.config.channel import ChannelRegistry, folded_channel_name

# The channels of 2.1 were named after their use. 'raw' is the first channel which is not an autofluorescence
OLD_RAW_CHANNEL = 'raw'
OLD_ARTERIES_CHANNEL = 'arteries'
OLD_AUTOFLUORESCENCE_CHANNEL = 'autofluorescence'
OLD_RAW_CHANNEL_NEW_NAMES = {'CellMap': ('nuclei', 'nuclei'), 'TubeMap': ('vessels', 'vessels')}
CONTENT_TYPES_TEXT = ', '.join(map(str, DATA_CONTENT_TYPES))


@dataclass(frozen=True)
class ChannelAnswer:
    """The new name and the content type (one of ``DATA_CONTENT_TYPES``) of a channel of 2.1"""
    name: str
    data_type: str


def suggested_channel_answers(old_channel_names: Sequence[str]) -> dict[str, ChannelAnswer]:
    """
    The new name and type of each channel of 2.1, from the pipeline it was used for.

    A sample with an ``arteries`` channel was processed by TubeMap, so its ``raw`` channel is ``vessels``. Otherwise
    it was processed by CellMap and ``raw`` is ``nuclei``. ``arteries`` and ``autofluorescence`` keep their name,
    as does any other channel, whose type is left ``undefined``.
    """
    pipeline = 'TubeMap' if OLD_ARTERIES_CHANNEL in old_channel_names else 'CellMap'
    known = {OLD_RAW_CHANNEL: ChannelAnswer(*OLD_RAW_CHANNEL_NEW_NAMES[pipeline]),
             OLD_ARTERIES_CHANNEL: ChannelAnswer(OLD_ARTERIES_CHANNEL, 'arteries'),
             OLD_AUTOFLUORESCENCE_CHANNEL: ChannelAnswer(OLD_AUTOFLUORESCENCE_CHANNEL, 'autofluorescence')}
    return {name: known.get(name, ChannelAnswer(name, 'undefined')) for name in old_channel_names}


def check_channel_answers(answers: Mapping[str, ChannelAnswer]) -> None:
    """
    Raise if the new names cannot be the channels of a sample: invalid or reserved, the same whatever the case,
    or a content type that does not exist. Nothing has been converted when this is called.
    """
    seen: dict[str, str] = {}
    for old_name, answer in answers.items():
        ChannelRegistry.validated_name(answer.name)  # InvalidChannelNameError
        if answer.data_type not in DATA_CONTENT_TYPES:
            raise ClearMapValueError(f'Channel {old_name!r}: {answer.data_type!r} is not a content type, '
                                     f'expected one of {DATA_CONTENT_TYPES}')
        folded = folded_channel_name(answer.name)
        if folded in seen:
            raise ClearMapValueError(f'Channels {seen[folded]!r} and {old_name!r} would both be named {answer.name!r} '
                                     f'(names are unique whatever the case): give them different names')
        seen[folded] = old_name


class UpgradePrompter(ABC):
    """Whoever answers the questions of an upgrade: a dialog, a terminal"""

    @abstractmethod
    def confirm(self, message: str) -> bool:
        """Whether to go on with the upgrade described by ``message``"""

    @abstractmethod
    def choose_channels(self, suggested: Mapping[str, ChannelAnswer]) -> dict[str, ChannelAnswer] | None:
        """The answer for each channel of 2.1 (named in ``suggested``), or None to stop the upgrade"""

    @abstractmethod
    def choose_renames(self, renames: Sequence[Rename]) -> list[Rename] | None:
        """The renames of ``renames`` to apply, or None to stop the upgrade"""


class ConsolePrompter(UpgradePrompter):
    """
    Ask in a terminal. Enter takes the suggestion between brackets; the end of the input (Ctrl-D) stops the upgrade.

    Parameters
    ----------
    ask : callable
        Reads a line after showing its argument, like ``input``. Without one, ``input`` is used and a terminal is
        required: a batch job refuses (:class:`ClearMapConfigError`) instead of waiting for an answer, or migrating
        without one.
    say : callable
        Shows a line, like ``print``.
    """

    def __init__(self, *, ask: Callable[[str], str] | None = None, say: Callable[[str], None] = print):
        if ask is None:
            if not sys.stdin.isatty():
                raise ClearMapConfigError('Upgrading an experiment needs someone to confirm it, and there is no terminal '
                                          '(a batch job?). Upgrade the experiment first, from a shell: '
                                          'python -m ClearMap.config.convert_config_versions <experiment folder>')
            ask = input
        self._ask = ask
        self._say = say

    def _ask_or_stop(self, question: str) -> str | None:
        try:
            return self._ask(question).strip()
        except EOFError:
            return None

    def confirm(self, message: str) -> bool:
        self._say(message)
        return (self._ask_or_stop('Continue? [y/N] ') or '').lower() in ('y', 'yes')

    def choose_channels(self, suggested: Mapping[str, ChannelAnswer]) -> dict[str, ChannelAnswer] | None:
        chosen: dict[str, ChannelAnswer] = {}
        for old_name, suggestion in suggested.items():
            name = self._ask_or_stop(f'New name of the channel {old_name!r} [{suggestion.name}]: ')
            data_type = None if name is None else self._ask_or_stop(
                f'Content type of {old_name!r} [{suggestion.data_type}] (one of {CONTENT_TYPES_TEXT}): ')
            if data_type is None:
                return None
            chosen[old_name] = ChannelAnswer(name or suggestion.name, data_type or suggestion.data_type)
        try:
            check_channel_answers(chosen)
        except ClearMapValueError as err:
            self._say(f'{err}. Start again.')
            return self.choose_channels(suggested)
        return chosen

    def choose_renames(self, renames: Sequence[Rename]) -> list[Rename] | None:
        for number, rename in enumerate(renames, start=1):
            self._say(f'{number}. {rename.source.name} -> {rename.target.name}')
        skipped = self._ask_or_stop('Numbers of the renames to leave out (e.g. 2,5), Enter for none: ')
        if skipped is None:
            return None
        try:
            left_out = {int(number) for number in skipped.split(',') if number.strip()}
        except ValueError:
            self._say(f'{skipped!r} is not a list of numbers. Start again.')
            return self.choose_renames(renames)
        selected = [rename for number, rename in enumerate(renames, start=1) if number not in left_out]
        return selected if self.confirm(f'{len(selected)} files will be renamed.') else None
