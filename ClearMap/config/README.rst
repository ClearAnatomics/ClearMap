config
======

This package contains both the modules to handle the configuration and the default configuration files
------------------------------------------------------------------------------------------------------

Composition:

config_loader
    The module that parses the configuration in different formats

update_Config
    The module to update the default configuration in ~/.clearmap when changing versions

machine_params
    The preferences that are machine specific (number of cores, start path...)

display_params
    Configuration of the default for visual attributes (e.g. fonts). Unlikely to be edited by the user.

default_sample_params
    The file to be stored in .clearmap and serve as the base for parameters of the current sample

default_alignment_params
    The file to be stored in .clearmap and serve as the base for parameters of the stitching
    and the atlas registration

default_cell_map_params
    The file to be stored in .clearmap and serve as the base for parameters of cell detection

default_vasculature_params
    The file to be stored in .clearmap and serve as the base for parameters of vasculature analysis

default_batch_params
    The file to be stored in .clearmap and serve as template for batch processing