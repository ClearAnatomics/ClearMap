import os

import configobj

# FIXME: implement validation


# TODO: add options to read different config file types (i.e. yaml and json)
def get_configs(cfg_path, processing_params_path, machine_cfg_path='~/.clearmap/machine_params.cfg'):
    sample_config = get_cfg(cfg_path)
    machine_config = get_cfg(os.path.expanduser(machine_cfg_path))
    processing_config = get_cfg(processing_params_path)

    return machine_config, sample_config, processing_config


def get_cfg(cfg_path):
    return configobj.ConfigObj(cfg_path, encoding="UTF8", indent_type='    ', unrepr=True, file_error=True)
