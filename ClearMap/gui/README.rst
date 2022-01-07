Introduction
============
This module pertains to the graphical interface (henceforth referred to as GUI). It is built using
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

2. Select the **gui** branch

  .. code-block:: bash

    $ git checkout gui

3. Create a conda environment using the *ClearMapUi.yml* environment file.

  .. code-block:: bash

   $ conda env create -f ClearMapUi.yml

  This will create an environment called *ClearMapUi*. Should you wish to change 
  this name, please check the conda documentation and amend the name in the 
  instructions below accordingly.
  To limit dependencies, *spyder* (which is particularly dependency hungry) 
  is not part of the environment.
  Please use *pip* after activating the environment should you wish to install it

  .. code-block:: bash

    $ conda activate ClearMapUi
    $ pip install 'spyder<5'

4. Copy the configuration files to the proper location:

  .. code-block:: bash

   $ mkdir ~/.clearmap
   $ cp ClearMap/config/*.cfg ~/.clearmap/


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

    $ env PYTHONPATH=`pwd`:$PYTHONPATH python ClearMap/gui/run_gui.py

In the future, this will be simplified to simply

  .. code-block:: bash

    $ clearmap_gui

Bugs
====

Please report bugs on the `github issue tracker`_ using the **GUI** label.

.. _github issue tracker: https://github.com/ChristophKirst/ClearMap2/issues
