"""
Plan the renaming of the files of the channels which are renamed.

The files of a channel are named after it (``<sample_id>_<channel>_<asset>.<ext>``, the registration folders
``<moving>_to_<fixed>``, whose pairs come from the registration config). When a channel is renamed, they are renamed too, or its results are lost to the pipeline.

Nothing is renamed here: the plan is built from the workspace (so that the names are those the assets build) and
handed over to the user to vet before :func:`ClearMap.IO.FileUtils.apply_renames` applies it.

What is not known is left alone: raw data (never renamed), anything outside the folder of the sample,
and the files which are not named after an asset of the channel.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping

from ClearMap.IO.FileUtils import Rename
from ClearMap.IO.assets_specs import ChannelId
from ClearMap.IO.workspace2 import Workspace2

STATUS_PREFIXES = ('', 'debug_')  # Files written in debug mode carry the status in front of their name


@dataclass(frozen=True)
class NotRenamed:
    """What belongs to a renamed channel but is left as it is, and why"""
    path: Path
    reason: str


@dataclass
class ChannelRenamePlan:
    """The files to rename, and the ones which belong to the renamed channels but are left alone"""
    renames: list[Rename] = field(default_factory=list)
    not_renamed: list[NotRenamed] = field(default_factory=list)

    def __bool__(self) -> bool:
        return bool(self.renames or self.not_renamed)


def plan_channel_file_renames(workspace: Workspace2, channel_renames: Mapping[str, str],
                              registration_pairs: Iterable[tuple[str, str]]) -> ChannelRenamePlan:
    """
    The files to rename when channels are renamed. Plan before the workspace itself is renamed:
    the files are still named after the old names, which are the ones the workspace knows.

    Parameters
    ----------
    workspace : Workspace2
        The workspace of the sample (its assets name the files).
    channel_renames : Mapping[str, str]
        The new name of each renamed channel, by old name.
    registration_pairs : Iterable[tuple[str, str]]
        The (moving, fixed) channels of each registration, under the old names (from the registration config,
        e.g. the atlas and the reference channel): their folders are ``<moving>_to_<fixed>``.

    Returns
    -------
    ChannelRenamePlan
        The renames, in no particular order, and the paths which are left alone.
    """
    channel_renames = {old: new for old, new in channel_renames.items() if old != new}
    plan = ChannelRenamePlan()
    if not channel_renames:
        return plan

    listings: dict[Path, list[Path]] = {}  # Each folder is listed once
    claimed: set[Path] = set()  # An asset family may include the files of another (cells, cells_raw)
    for channel in workspace.channels:
        new_channel = _renamed_channel(channel, channel_renames)
        if channel is None or new_channel is None:
            continue
        for asset in workspace[channel].values():
            _plan_asset(workspace, asset, new_channel, plan, listings, claimed)
    _plan_registration_folders(workspace, channel_renames, registration_pairs, plan)
    return plan


def _renamed_channel(channel: ChannelId | None, channel_renames: Mapping[str, str]) -> ChannelId | None:
    """The channel under its new name (the parts of a compound channel stay in place), None if it is not renamed"""
    if channel is None:
        return None
    if isinstance(channel, tuple):
        renamed = tuple(channel_renames.get(part, part) for part in channel)
        return None if renamed == channel else renamed
    return channel_renames.get(channel)


def _plan_asset(workspace: Workspace2, asset, new_channel: ChannelId, plan: ChannelRenamePlan,
                listings: dict[Path, list[Path]], claimed: set[Path]) -> None:
    if asset.type_spec.name == 'raw':
        location = _location(asset)
        if location is not None:
            plan.not_renamed.append(NotRenamed(location, 'raw data is never renamed'))
        return
    if asset.is_expression:  # Named by an expression, not after the channel (the registration: see below)
        return

    # The status is a prefix of the file name: handled with the prefixes below
    old_base = asset.base_name_for(asset.channel_spec.name, with_status=False)
    new_base = asset.base_name_for(new_channel, with_status=False)
    directory = Path(asset.directory)
    inside_sample = directory.resolve().is_relative_to(Path(workspace.directory).resolve())
    if directory not in listings:
        listings[directory] = sorted(directory.iterdir()) if directory.is_dir() else []

    for prefix in STATUS_PREFIXES:
        old_start = prefix + old_base
        for path in listings[directory]:
            if path in claimed or not _starts_with_name(path.name, old_start):
                continue
            claimed.add(path)
            if inside_sample:
                plan.renames.append(Rename(path, path.with_name(prefix + new_base + path.name[len(old_start):])))
            else:
                plan.not_renamed.append(NotRenamed(path, 'outside the folder of the sample'))


def _plan_registration_folders(workspace: Workspace2, channel_renames: Mapping[str, str],
                               registration_pairs: Iterable[tuple[str, str]], plan: ChannelRenamePlan) -> None:
    """The registration folders ``<moving>_to_<fixed>`` and the landmarks inside them (``<channel>_landmarks``)"""
    folder = Path(workspace.directory) / workspace.resource_type_to_folder.get('elastix', '')
    for moving, fixed in dict.fromkeys(registration_pairs):  # Once each, in order
        if moving not in channel_renames and fixed not in channel_renames:
            continue
        registration_folder = folder / f'{moving}_to_{fixed}'
        if not registration_folder.is_dir():
            continue
        for path in sorted(registration_folder.iterdir()):
            for name in {moving, fixed} & channel_renames.keys():
                landmarks = f'{name}_landmarks'
                if path.name == landmarks or path.name.startswith(f'{landmarks}.'):
                    plan.renames.append(Rename(path, path.with_name(channel_renames[name] + path.name[len(name):])))
        new_moving, new_fixed = channel_renames.get(moving, moving), channel_renames.get(fixed, fixed)
        plan.renames.append(Rename(registration_folder, folder / f'{new_moving}_to_{new_fixed}'))


def _starts_with_name(file_name: str, start: str) -> bool:
    """Whether the file name is start, or start followed by an extension or by more of the name"""
    return file_name.startswith(start) and file_name[len(start):len(start) + 1] in ('_', '.')


def _location(asset) -> Path | None:
    """Where the files of the asset are (the folder they share, for many), None if there are none"""
    paths = asset.all_existing_paths()
    return Path(os.path.commonpath(paths)) if paths else None
