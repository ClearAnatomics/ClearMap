"""
This script is used to update the default config files in the ClearMap package.

It will detect if a default config file is missing and create it, or merge it with the existing one.
If a parameter was altered by the user, it will not be overwritten.

This module will be extended to support conversion from one ClearMap configuration version to another.
"""
import copy
import os.path

import inspect
import shutil
from pathlib import Path

from ClearMap.config.config_loader import (ConfigLoader, get_alternatives, CONFIG_NAMES,
                                           get_cfg_reader_function, clearmap_version)


CFG_DIR = Path(inspect.getfile(inspect.currentframe())).resolve().parent
CLEARMAP_DIR = str(CFG_DIR.parent.parent)  # used by shell script


def update_default_config():
    loader = ConfigLoader('~/.clearmap')
    for cfg_name in CONFIG_NAMES:
        if cfg_name == 'alignment' and clearmap_version >= '3.0.0':
            continue
        default_cfg_path = loader.get_default_path(cfg_name, must_exist=True, from_package=True)
        cfg_paths = [loader.get_default_path(alternative, must_exist=False, from_package=False)
                     for alternative in get_alternatives(cfg_name)]
        existing_paths = [os.path.exists(p) for p in cfg_paths]
        if not any(existing_paths):  # missing then copy
            print(f'Creating missing default config for {cfg_name}')
            shutil.copy(default_cfg_path, cfg_paths[0])
        else:  # if present merge
            cfg_path = cfg_paths[existing_paths.index(True)]

            read_cfg = get_cfg_reader_function(cfg_path)
            cfg = read_cfg(cfg_path)
            default_cfg = read_cfg(default_cfg_path)
            print(f'Merging {cfg_name}')
            merge_config(default_cfg, cfg)


def merge_config(source_cfg, dest_cfg):
    dest_copy = copy.deepcopy(dest_cfg)
    dest_cfg = deep_merge_dicts(dest_cfg, source_cfg)
    dest_cfg = remove_extra_keys(dest_cfg, source_cfg)
    # FIXME: add step to swap brackets and parenthesis
    if dest_copy != dest_cfg:
        dest_cfg.write()


def remove_extra_keys(a, b):
    """remove keys from `a` if not in `b`"""
    for key in a:
        if key in b:
            if isinstance(a[key], dict) and isinstance(b[key], dict):
                a[key] = remove_extra_keys(a[key], b[key])
        else:
            del a[key]
    return a


def deep_merge_dicts(a, b, path=None):  # modified from https://stackoverflow.com/a/7205107
    if path is None:
        path = []
    for key in b:
        if key in a:
            if isinstance(a[key], dict) and isinstance(b[key], dict):
                deep_merge_dicts(a[key], b[key], path + [str(key)])
            elif a[key] != b[key]:
                print(f'Config value "{key}" has been modified by user, skipping.')
        else:
            a[key] = b[key]

    return a


if __name__ == '__main__':
    update_default_config()
