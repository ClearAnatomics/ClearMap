import os.path

import inspect
import shutil

from ClearMap.config.config_loader import ConfigLoader, get_alternatives, is_tab_file, CONFIG_NAMES

CFG_DIR = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
CLEARMAP_DIR = os.path.dirname(os.path.dirname(CFG_DIR))


def get_default(cfg_name):
    prefix = 'default_' if is_tab_file(cfg_name) else ''
    cfg_file_name = f'{prefix}{cfg_name}_params.cfg'
    return os.path.join(CFG_DIR, cfg_file_name)


def update_default_config():
    loader = ConfigLoader(None)
    for cfg_name in CONFIG_NAMES:
        default_cfg_path = get_default(cfg_name)
        cfg_paths = [loader.get_default_path(alternative) for alternative in get_alternatives(cfg_name)]
        existing_paths = [os.path.exists(p) for p in cfg_paths]
        if not any(existing_paths):  # missing then copy
            print(f'Creating missing default config for {cfg_name}')
            shutil.copy(default_cfg_path, cfg_paths[0])
        else:  # if present merge
            cfg_path = cfg_paths[existing_paths.index(True)]
            ext = os.path.splitext(cfg_path)[-1]
            read_cfg = loader.loader_functions[ext]
            cfg = read_cfg(cfg_path)
            default_cfg = read_cfg(default_cfg_path)
            was_modified = False
            for k in default_cfg.keys():
                if k not in cfg.keys():
                    cfg[k] = default_cfg[k]
                    was_modified = True
                else:
                    if cfg[k] != default_cfg[k]:
                        print(f'Value "{k}" in {cfg_name}_params has been modified, skipping.')
            if was_modified:
                cfg.write()


if __name__ == '__main__':
    update_default_config()
