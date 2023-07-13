processors
==========

This package contains the classes to replace the scripts. These depend on the config files and serve as the backend to the graphical interface but can also be used independently.


Composition
===========

sample_preparation
    The classes that control the stitching and the atlas registration.

cell_map
    The classes that control the cell counting.

tube_map
    The classes that control the vasculature (tubular structures) analysis.

batch_process
    The module for the batch processing and summary analysis of the samples.
    This can serve as a template for users who prefer scripting.