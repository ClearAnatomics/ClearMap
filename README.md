# ClearMap2

[![DOI](https://zenodo.org/badge/59701678.svg)](https://zenodo.org/badge/latestdoi/59701678)
[![GitHub stars](https://img.shields.io/github/stars/ChristophKirst/ClearMap2.svg?style=social&label=Star)](https://github.com/ChristophKirst/ClearMap2) 
[![GitHub stars](https://img.shields.io/github/stars/ChristophKirst/ClearMap2.svg?style=social&label=Star)](https://github.com/ChristophKirst/ClearMap2)
[![Follow on Twitter](https://img.shields.io/twitter/follow/clearmap_idisco?style=social&logo=twitter)](https://twitter.com/intent/follow?screen_name=clearmap_idisco)


[![Generic badge](https://img.shields.io/badge/contributions-welcome-brightgreen.svg)](docs/contribute.md)
[![License](https://img.shields.io/github/license/ChristophKirst/ClearMap?color=green&style=plastic)](https://github.com/ChristophKirst/ClearMap2/LICENSE.txt)
![Size](https://img.shields.io/github/repo-size/ChristophKirst/ClearMap2?style=plastic)
[![Language](https://img.shields.io/github/languages/top/ChristophKirst/ClearMap?style=plastic)](https://github.com/ChristophKirst/ClearMap2)
[![](https://github.com/ChristophKirst/RepoTracker/workflows/RepoTracker/badge.svg)](https://github.com/ChristophKirst/RepoTracker/actions)

<p align="center">
<img src="https://christophkirst.github.io/Test/html/test.gif" height="160">
<img src="https://christophkirst.github.io/Test/html/CellMap_small_fast.gif" height="160">
<img src="https://christophkirst.github.io/Test/html/tubeMap_raw_movie_small.gif" height="160">
</p>


*ClearMap* is a toolbox for the analysis and registration of volumetric
data from cleared tissues.

*ClearMap's* tool box includes 

* [Wobbly-Stitcher](https://christophkirst.github.io/ClearMap2Documentation/html/wobblystitcher.html), 

* [TubeMap](https://christophkirst.github.io/ClearMap2Documentation/html/tubemap.html),

  [![TubeMap](https://christophkirst.github.io/ClearMap2Documentation/images/TubeMap_graph_movie_small.gif)](https://christophkirst.github.io/ClearMap2Documentation/html/TubeMap.html)

* [CellMap](https://christophkirst.github.io/ClearMap2Documentation/html/cellmap.html)

  [![CellMap](https://christophkirst.github.io/ClearMap2Documentation/images/CellMap_small_fast.gif)](https://christophkirst.github.io/ClearMap2Documentation/html/CellMap.html)


*ClearMap* has been designed to analyze O(TB) 3d datasets obtained 
via light sheet microscopy from iDISCO+ cleared tissue samples 
immunolabeled for proteins. 
 
*ClearMap* has been written for mapping immediate early genes [Renier2016]_
as well as vasculature networks of whole mouse brains [Kirst2020]_

.. image:: Static/cell_abstract_2020.jpg
   :target: https://doi.org/10.1016/j.cell.2016.05.007 
   :width: 300  
.. image:: Static/cell_abstract_2016.jpg
   :target: https://doi.org/10.1016/j.cell.2020.01.028
   :width: 300
  

*ClearMap* tools may also be useful for data obtained with other types of 
microscopes, types of markers, clearing techniques, as well as other species,
organs, or samples.

*ClearMap* is written in Python 3 and is designed to take advantage of
parallel processing capabilities of modern workstations. We hope the open 
structure of the code will enable many new modules to be added to *ClearMap*
to broaden the range of applications to different types of biological objects 
or structures.




Installation
------------

Install ClearMap by cloning it form `github <http://www.github.com/>`_::

    $ git clone https://github.com/ChristophKirst/ClearMap.git

See `GIT.md <https://github.com/ChristophKirst/ClearMap/blob/master/GIT.md>`_ for basic help with git.

If you want to register data to reference images via elastix or
classify pixels via ilastik configure the /ClearMap/Settings.py file.

You will most likely also need to install several python packages e.g. via 
pip or apt-get.

See the `documentation <https://rawgit.com/ChristophKirst/ClearMap/master/docs/_build/html/index.html>`_ for more details.

Additional files for mouse brain registration can be found on the `iDISCO website <https://idisco.info/>`_.


Quickstart
----------

   * see the template scripts in the `./ClearMap/Scripts <https://github.com/ChristophKirst/ClearMap/tree/master/ClearMap/Scripts>`_ folder 
   * see the `ClearMap documentation <https://rawgit.com/ChristophKirst/ClearMap/master/docs/_build/html/index.html>`_ 


Copyright
---------
__author__    = 'Christoph Kirst <christoph.kirst.ck@gmail.com>'
__license__   = 'GPLv3 - GNU General Pulic License v3 (see LICENSE.txt)'
__copyright__ = 'Copyright © 2020 by Christoph Kirst'
__webpage__   = 'http://idisco.info'
__download__  = 'http://www.github.com/ChristophKirst/ClearMap2'

License
-------
    GPLv3, see LICENSE.txt for details.



