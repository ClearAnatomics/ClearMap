Introduction
============
This package pertains to the graphical interface (henceforth referred to as GUI). It is built using
PyQt5 and currently creates a frontend to the CellMap script. Support for TubeMap is planned in the future.

This module uses new versions of the *scripts* that are still located in ClearMap/Scripts
(sample_preparation.py and cell_map.py) that were rewritten using object-oriented programming.

The behaviour of the code is controlled through configuration files that can be found under ClearMap/config
Please see installation_ for instruction on placing these configuration files in the right location.
Should the user prefer to do so, the aforementioned scripts can be run using these configuration files without
the need for the GUI. In the future, this functionality will for the basis of a batch mode
allowing to start the analysis of several samples at once without further user intervention.

Installation
============
To install the GUI you need to follow these steps:

1. Download the ClearMap source code

  .. code-block:: bash

    $ cd /path/to/where/you/should/download
    $ git clone https://github.com/ChristophKirst/ClearMap2.git
    $ cd ClearMap2

2. Select the **dev** branch

  .. code-block:: bash

    $ git checkout dev

3. Make the install script executable

  .. code-block:: bash

    $ chmod u+x install_gui.sh <environment_file.yml>  # If the environment file is not supplied it will use python 3.7

4. Start the install and answer the prompts

  .. code-block:: bash

    $ ./install_gui.sh


Using the interface
===================

The interface as mentioned above uses configuration files. Some files are
system-wide, i.e. they pertain to all experiments, like the 
**~/.clearmap/machine_params.cfg** file which selects the start folder, the
number of CPU cores for each step ...
Others are experiment specific. 

For each experiment folder, 3 files should be present at the root of the 
experiment directory: 

  **sample_params.cfg**
    Controls the variables pertaining to the sample. Essentially, these are the
    acquisition metadata
   
  **processing_params.cfg**
    Controls the variables pertaining to the stitching and the alignment
 
  **cell_map_params.cfg**
    Controls the behaviour of the cell detection (size of the filter kernels,
    of the expected cells ...)
    
  

When selecting the folder as the first step after starting the interface below,
if these 3 files are not present, the program will offer to install them for you
from the defaults located in ~/.clearmap, thereby offering you a way to customise
the default settings for all experiments.

To use the interface, simply run:

  .. code-block:: bash

    $ conda activate ClearMapUi
    $ clearmap-gui

Structure
=========

run_gui
    The main module used to start the Graphical User Interface

tabs
    The module containing the classes for the tabs representing different steps in the analysis
    (usually matched by a processor in ClearMap/processors)

params
    The parameters for the GUI of the different tabs. These are linked to the pertaining config files.

widgets
    Custom widgets for the graphical interface.

dialogs
    Custom dialogs (e.g. prompts, warnings ...) for the GUI

widget_monkey_patch_callbacks
    These are functions that will be bound as methods to the graphical widgets based on type and name
    to enhance their abilities.

gui_logging
    A special widget used for normal and error logging. This is crucial to track progress.

pyuic_utils
    A customised version of the PyQt5 module of the same name to patch the classes built from
    the **.ui** files.

style
    As the name indicates (defines e.g. colors)

Bugs
====

Please report bugs on the `github issue tracker`_ using the **GUI** label.

.. _github issue tracker: https://github.com/ChristophKirst/ClearMap2/issues
