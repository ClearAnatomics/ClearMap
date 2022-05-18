import os

import configobj

# FIXME: implement validation


# TODO: add options to read different config file types (i.e. yaml and json)
def get_configs(cfg_path, processing_params_path, machine_cfg_path='~/.clearmap/machine_params.cfg'):
    sample_config = get_configobj_cfg(cfg_path)
    processing_config = get_configobj_cfg(processing_params_path)
    machine_config = get_configobj_cfg(machine_cfg_path)

    return machine_config, sample_config, processing_config


def get_configobj_cfg(cfg_path):
    cfg_path = clean_path(cfg_path)
    return configobj.ConfigObj(cfg_path, encoding="UTF8", indent_type='    ', unrepr=True, file_error=True)


def get_yml_cfg(cfg_path):
    raise NotImplementedError


def get_json_cfg(cfg_path):
    raise NotImplementedError


def clean_path(path):
    return os.path.normpath(os.path.expanduser(path))


class ConfigLoader(object):
    supported_exts = ('.cfg', '.ini', '.yml', '.json')
    loader_functions = {
        '.cfg': get_configobj_cfg,
        '.ini': get_configobj_cfg,
        '.yml': get_yml_cfg,
        '.json': get_json_cfg
    }

    def __init__(self, src_dir):
        self.src_dir = src_dir
        self.default_dir = os.path.expanduser('~/.clearmap/')
        self.sample_cfg_path = ''  # OPTIMISE: could use cached property
        self.preferences_path = ''
        self.cell_map_cfg_path = ''

    def get_cfg_path(self, cfg_name, must_exist=True):
        """

        Parameters
        ----------
        cfg_name: str
        must_exist: bool

        Returns
        -------

        """
        variants = [cfg_name]  # For e.g. legacy names
        if cfg_name == 'alignments':
            variants.append('processing')
        elif cfg_name == 'vasculature':
            variants.append('tube_map')

        for _cfg_name in variants:
            if not _cfg_name.endswith('params'):
                _cfg_name += '_params'
            for ext in self.supported_exts:
                cfg_path = clean_path(os.path.join(self.src_dir, '{}{}'.format(_cfg_name, ext)))
                if os.path.exists(cfg_path):
                    return cfg_path
        if not must_exist:  # If none found but not necessary, return the first possible option
            return clean_path(os.path.join(self.src_dir, '{}{}'.format(cfg_name, self.supported_exts[0])))
        raise FileNotFoundError('Could not find file {} in {} with variants {}'
                                .format(cfg_name, self.src_dir, variants))

    def get_cfg(self, cfg_name):
        if self.is_tab_file(cfg_name):
            cfg_path = self.get_cfg_path(cfg_name)
        else:
            cfg_path = self.get_default_path(cfg_name)
        ext = os.path.splitext(cfg_path)[-1]
        return self.loader_functions[ext](cfg_path)

    def get_default_path(self, cfg_name):
        if not cfg_name.endswith('params') and 'sample' not in cfg_name:  # TODO: just ust sample_params instead
            cfg_name += '_params'
        for ext in self.supported_exts:
            prefix = 'default_' if self.is_tab_file(cfg_name) else ''
            cfg_name = '{}{}{}'.format(prefix, cfg_name, ext)
            cfg_path = clean_path(os.path.join(self.default_dir, cfg_name))
            return cfg_path
        raise FileNotFoundError('Could not find file {} in {}'.format(cfg_name, self.default_dir))

    def is_machine_file(self, cfg_name):
        return any([base in cfg_name for base in ('machine', 'preferences')])

    def is_tab_file(self, cfg_name):
        return cfg_name in ('{}_params'.format(name) for name in ('sample',
                                                                  'alignments',
                                                                  'processing',
                                                                  'cell_map',
                                                                  'cells',
                                                                  'vasculature',
                                                                  'tube_map'))
