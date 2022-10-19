import inspect
import os

import configobj

# FIXME: implement validation


INSTALL_CFG_DIR = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))  # Where this file resides (w cfgs)
CLEARMAP_CFG_DIR = os.path.expanduser('~/.clearmap/')


def get_configobj_cfg(cfg_path):
    cfg_path = clean_path(cfg_path)
    try:
    return configobj.ConfigObj(cfg_path, encoding="UTF8", indent_type='    ', unrepr=True, file_error=True)
    except configobj.ConfigObjError as err:
        print(f'Could not read config file "{cfg_path}", some errors were encountered: "{err}"')


def get_yml_cfg(cfg_path):
    """

    Parameters
    ----------
    cfg_path (str)

    Returns Should return a dict like object with a write method and filename that gives the path
    -------

    """
    raise NotImplementedError


def get_json_cfg(cfg_path):
    """

    Parameters
    ----------
    cfg_path (str)

    Returns Should return a dict like object with a write method and filename that gives the path
    -------

    """
    raise NotImplementedError


tabs_alternatives = [
    ['sample'],
    ['alignment', 'processing'],
    ['cell_map'],
    ['vasculature', 'tube_map'],
    ['batch'],
]

alternative_names = tabs_alternatives + [['machine'], ['display']]
CONFIG_NAMES = [names[0] for names in alternative_names]


def get_alternatives(cfg_name):
    alternatives = [names for names in alternative_names if cfg_name in names]
    if not alternatives:
        raise ValueError(f'Could not find any alternative for {cfg_name}')
    return alternatives[0]


def flatten_alternatives(alternatives):
    flat = []
    for names in alternatives:
        flat.extend(names)
    return flat


def is_tab_file(cfg_name):
    cfg_name = cfg_name.replace('_params', '')
    return cfg_name in flatten_alternatives(tabs_alternatives)


def clean_path(path):
    return os.path.normpath(os.path.expanduser(path))


def is_machine_file(cfg_name):
    return any([base in cfg_name for base in ('machine', 'preferences')])


class ConfigLoader(object):
    loader_functions = {
        '.cfg': get_configobj_cfg,
        '.ini': get_configobj_cfg,
        '.yml': get_yml_cfg,
        '.json': get_json_cfg
    }
    supported_exts = tuple(loader_functions.keys())
    default_dir = CLEARMAP_CFG_DIR

    def __init__(self, src_dir):
        self.src_dir = src_dir
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

        variants = get_alternatives(cfg_name)
        for alternative_name in variants:
            if not alternative_name.endswith('params'):
                alternative_name += '_params'
            for ext in self.supported_exts:
                cfg_path = clean_path(os.path.join(self.src_dir, f'{alternative_name}{ext}'))
                if os.path.exists(cfg_path):
                    return cfg_path
        if not must_exist:  # If none found but not necessary, return the first possible option
            return clean_path(os.path.join(self.src_dir, f'{cfg_name}_params{self.supported_exts[0]}'))
        raise FileNotFoundError(f'Could not find file {cfg_name} in {self.src_dir} with variants {variants}')

    def get_cfg(self, cfg_name):
        if is_tab_file(cfg_name):
            cfg_path = self.get_cfg_path(cfg_name)
        else:
            cfg_path = self.get_default_path(cfg_name)
        return self.get_cfg_from_path(cfg_path)

    @staticmethod
    def get_cfg_from_path(cfg_path):
        ext = os.path.splitext(cfg_path)[-1]
        return ConfigLoader.loader_functions[ext](cfg_path)

    @staticmethod
    def get_default_path(cfg_name, must_exist=True, install_mode=False):  # FIXME: recursive w/ alternatives
        if cfg_name.endswith('_params'):
            cfg_name = cfg_name.replace('_params', '')
        for name_variant in get_alternatives(cfg_name):
            name_variant += '_params'
            for ext in ConfigLoader.supported_exts:
                cfg_path = ConfigLoader._name_to_default_path(name_variant, ext, install_mode=install_mode)
                if os.path.exists(cfg_path):
                    break
            if os.path.exists(cfg_path):  # REFACTOR: avoid duplicate w/ above
                break
        else:
            if must_exist:
                cfg_dir = INSTALL_CFG_DIR if install_mode else ConfigLoader.default_dir
                raise FileNotFoundError(f'Could not find file {cfg_name} in {cfg_dir}')
            else:  # Return first (default) ext if none found
                return ConfigLoader._name_to_default_path(f'{cfg_name}_params', ConfigLoader.supported_exts[0],
                                                          install_mode=install_mode)
        return cfg_path

    @staticmethod
    def _name_to_default_path(cfg_name, ext, install_mode=False):
        prefix = 'default_' if is_tab_file(cfg_name) else ''
        cfg_name = f'{prefix}{cfg_name}{ext}'

        cfg_dir = INSTALL_CFG_DIR if install_mode else ConfigLoader.default_dir
        cfg_path = clean_path(os.path.join(cfg_dir, cfg_name))
        return cfg_path


def get_configs(cfg_path, processing_params_path, machine_cfg_path=None):
    if machine_cfg_path is None:
        machine_cfg_path = ConfigLoader.get_default_path('machine')
    sample_config = ConfigLoader.get_cfg_from_path(cfg_path)
    processing_config = ConfigLoader.get_cfg_from_path(processing_params_path)
    machine_config = ConfigLoader.get_cfg_from_path(machine_cfg_path)

    return machine_config, sample_config, processing_config
