"""
This script is used to update the default config files in the ClearMap package.

It will detect if a default config file is missing and create it, or merge it with the existing one.
If a parameter was altered by the user, it will not be overwritten.

This module will be extended to support conversion from one ClearMap configuration version to another.
"""
import copy

import inspect
import shutil
from pathlib import Path

from packaging.version import Version

from ClearMap.config.config_handler import ConfigHandler, clearmap_version, ALTERNATIVES_REG, LEGACY_SECTIONS

CFG_DIR = Path(inspect.getfile(inspect.currentframe())).resolve().parent
CLEARMAP_DIR = str(CFG_DIR.parent.parent)  # used by shell script


def update_default_config():  # FIXME: add carry over of previous version settings
    for cfg_name in ALTERNATIVES_REG.canonical_config_names:
        if ALTERNATIVES_REG.is_legacy_cfg(cfg_name):
            continue
        package_default_path = ConfigHandler.get_default_path(cfg_name, must_exist=True, from_package=True)
        package_default_cfg = ConfigHandler.get_cfg_from_path(package_default_path)
        if ALTERNATIVES_REG.is_global_cfg(cfg_name):
            user_path = ConfigHandler.get_global_canonical_path(cfg_name)
        else:
            user_path = ConfigHandler.get_user_defaults_canonical_path(cfg_name)
        if not user_path.exists():  # missing -> copy
            print(f'Creating missing default config for {cfg_name}')
            ConfigHandler.dump(user_path, package_default_cfg)  # this can convert formats too
        else:  # if config for **this** version is already present -> merge
            user_default_cfg = ConfigHandler.get_cfg_from_path(user_path)

            print(f'Merging "{cfg_name}" config section')
            merge_config(package_default_cfg, user_default_cfg)


def merge_config(source_cfg, dest_cfg):
    dest_copy = copy.deepcopy(dest_cfg)
    dest_cfg = deep_merge_dicts(dest_cfg, source_cfg)
    dest_cfg = remove_extra_keys(dest_cfg, source_cfg)
    # FIXME: add step to swap brackets and parenthesis
    if dest_copy != dest_cfg:
        dest_cfg.write()


def remove_extra_keys(a, b):
    """remove keys from `a` if not in `b`"""
    for key in list(a.keys()):
        if key in b:
            v_a, v_b = a[key], b[key]
            if isinstance(v_a, dict) and isinstance(v_b, dict):
                a[key] = remove_extra_keys(v_a, v_b)
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
