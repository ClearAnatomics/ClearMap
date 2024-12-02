"""
A simple configuration module for the atlases. This is parsed by the ClearMap GUI to
display the available atlases.
"""

ATLAS_NAMES_MAP = {
    'ABA 2017 - adult mouse - 25µm': {'resolution': 25, 'base_name': 'ABA_25um_2017'},
    'ABA - adult mouse - 25µm': {'resolution': 25, 'base_name': 'ABA_25um'},
    # 'LAMBADA - P1 - 25µm': {'resolution': 25, 'base_name': 'LAMBADA_P1_25um'},
    # 'LAMBADA - P3 - 25µm': {'resolution': 25, 'base_name': 'LAMBADA_P3_25um'},
    # 'LAMBADA - P5 - 25µm': {'resolution': 25, 'base_name': 'LAMBADA_P5_25um'},
    # 'LAMBADA - P7 - 25µm': {'resolution': 25, 'base_name': 'LAMBADA_P7_25um'},
    # 'LAMBADA - P9 - 25µm': {'resolution': 25, 'base_name': 'LAMBADA_P9_25um'},
    # 'LAMBADA - P10 - 25µm': {'resolution': 25, 'base_name': 'LAMBADA_P10_25um'},
    # 'LAMBADA - P12 - 25µm': {'resolution': 25, 'base_name': 'LAMBADA_P12_25um'},
    # 'LAMBADA - P14 - 25µm': {'resolution': 25, 'base_name': 'LAMBADA_P14_25um'},
    # 'LAMBADA - P21 - 25µm': {'resolution': 25, 'base_name': 'LAMBADA_P21_25um'},
}


STRUCTURE_TREE_NAMES_MAP = {
    'ABA json 2022': 'ABA_annotation_last.json',
    'ABA json clearmap': 'ABA_annotation.json',
}
