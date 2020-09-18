ClearMap
========

[![DOI](https://zenodo.org/badge/256322811.svg)](https://zenodo.org/badge/latestdoi/256322811)
[![GitHub stars](https://img.shields.io/github/stars/ChristophKirst/ClearMap2.svg?style=social&label=Star)](https://github.com/ChristophKirst/ClearMap2) 
[![GitHub forks](https://img.shields.io/github/forks/ChristophKirst/ClearMap2.svg?style=social&label=Fork)](https://github.com/ChristophKirst/ClearMap2)
[![Follow on Twitter](https://img.shields.io/twitter/follow/clearmap_idisco?style=social&logo=twitter)](https://twitter.com/intent/follow?screen_name=clearmap_idisco)


[![Generic badge](https://img.shields.io/badge/contributions-welcome-brightgreen.svg)](docs/contribute.md)
[![License](https://img.shields.io/github/license/ChristophKirst/ClearMap?color=green&style=plastic)](https://github.com/ChristophKirst/ClearMap2/LICENSE.txt)
![Size](https://img.shields.io/github/repo-size/ChristophKirst/ClearMap2?style=plastic)
[![Language](https://img.shields.io/github/languages/top/ChristophKirst/ClearMap?style=plastic)](https://github.com/ChristophKirst/ClearMap2)
[![](https://github.com/ChristophKirst/RepoTracker/workflows/RepoTracker/badge.svg)](https://github.com/ChristophKirst/RepoTracker/actions)


<p align="center">
<img src="https://christophkirst.github.io/ClearMap2Documentation/images/TubeMap_graph_movie.gif" height="150">
<img src="https://christophkirst.github.io/ClearMap2Documentation/images/CellMap_raw_movie.gif" height="150">
<img src="https://christophkirst.github.io/ClearMap2Documentation/images/TubeMap_raw_movie.gif" height="150">
</p>


*ClearMap* is a toolbox for the analysis and registration of volumetric
data from cleared tissues.

*ClearMap's* tool box includes 

## [Wobbly-Stitcher](https://christophkirst.github.io/ClearMap2Documentation/html/wobblystitcher.html) <a href="https://christophkirst.github.io/ClearMap2Documentation/html/TubeMap.html#Stitching"> <img src="https://christophkirst.github.io/ClearMap2Documentation/images/WobblyStitcher.jpg" alt="WobblyStitcher" width="150" align="right" vspace = "5"/></a>

Stitch TB data sets non-rigidly.
  
## [TubeMap](https://christophkirst.github.io/ClearMap2Documentation/html/tubemap.html) <a href="https://christophkirst.github.io/ClearMap2Documentation/html/TubeMap.html"> <img src="https://christophkirst.github.io/ClearMap2Documentation/images/TubeMap_graph_raw_movie.gif" alt="WobblyStitcher" width="150" align="right" vspace = "5"/></a>

Extract vasculature and other tubular networks from TB data.

## [CellMap](https://christophkirst.github.io/ClearMap2Documentation/html/cellmap.html) <a href="https://christophkirst.github.io/ClearMap2Documentation/html/CellMap.html"> <img src="https://christophkirst.github.io/ClearMap2Documentation/images/CellMap_raw_bw.gif" alt="WobblyStitcher" width="150" align="right" vspace = "5"/></a>

Extract neuronal activity marker and cell shapes.


## Background and Applications

*ClearMap* has been designed to analyze O(TB) 3d datasets obtained 
via light sheet microscopy from iDISCO+ cleared tissue samples 
immunolabeled for proteins. 

*ClearMap* has been written for mapping immediate early genes 
[Renier et al. Cell 2016](https://doi.org/10.1016/j.cell.2016.05.007)
as well as vasculature networks of whole mouse brains 
[Kirst et al. Cell 2020](https://doi.org/10.1016/j.cell.2020.01.028)

<p align="center">
<a href="https://doi.org/10.1016/j.cell.2016.05.007">
<img src="https://christophkirst.github.io/ClearMap2Documentation/images/Cell_abstract_2016.jpg" 
 alt="Cell2016" width="300" hspace="40"/></a>
<a href="https://doi.org/10.1016/j.cell.2020.01.028">
<img src="https://christophkirst.github.io/ClearMap2Documentation/images/Cell_abstract_2020.jpg" 
 alt="Cell2020" width="300" hspace="40"/></a>
</p>

*ClearMap* tools may also be useful for data obtained with other types of 
microscopes, types of markers, clearing techniques, as well as other species,
organs, or samples.

*ClearMap* is written in Python 3 and is designed to take advantage of
parallel processing capabilities of modern workstations. We hope the open 
structure of the code will enable many new modules to be added to *ClearMap*
to broaden the range of applications to different types of biological objects 
or structures.


[Installation](https://christophkirst.github.io/ClearMap2Documentation/html/installation.html)
==============================================================================================

Please refer to our [documentation](https://christophkirst.github.io/ClearMap2Documentation)
on how to [install](https://christophkirst.github.io/ClearMap2Documentation/html/installation.html) ClearMap.

[Tutorials](https://christophkirst.github.io/ClearMap2Documentation/html/usage.html)
====================================================================================

Tutorials for [TubeMap](https://christophkirst.github.io/ClearMap2Documentation/html/tubemap.html) 
and [CellMap](https://christophkirst.github.io/ClearMap2Documentation/html/tubemap.html) are
avaialbe as [jupyer](https://jupyter.org/) notebooks:
  
  * [TubeMap](https://christophkirst.github.io/ClearMap2Documentation/html/TubeMap.html)
  
  * [CellMap](https://christophkirst.github.io/ClearMap2Documentation/html/TubeMap.html)


[Documentation](https://christophkirst.github.io/ClearMap2Documentation)
========================================================================

ClearMap comes with a full [documentation](https://christophkirst.github.io/ClearMap2Documentation).

For experimental protocols also refer to [idisco.info](http:://idisco.info)


[News and Media](https://christophkirst.github.io/ClearMap2Documentation/html/media.html)
=========================================================================================

ClearMap has been featured in differnet articles, interviews and a TEDx talk:

<p align="center">
<a href="https://www.ted.com/talks/christoph_kirst_a_transparent_journey_into_the_brain_and_its_flexible_function">
<img src="https://christophkirst.github.io/ClearMap2Documentation/images/TEDx_video.jpeg" alt="Cell2016" height="200" hspace="40"/></a>
<a href="https://www.youtube.com/watch?v=-LEfL55-EUU">
<img src="https://img.youtube.com/vi/-LEfL55-EUU/0.jpg" alt="Cell2020" height="200" hspace="40"/></a>
</p>

<p align="center">
<a href="https://www.nature.com/articles/s41684-020-0556-7">
<img src="https://media.springernature.com/full/nature-cms/uploads/product/nature/header-86f1267ea01eccd46b530284be10585e.svg" 
  alt="Cell2016" height="80" hspace="40/></a>
<a href="https://www.sciencedirect.com/science/article/pii/S0092867416307371">
<img src="https://publons.com/media/thumbs/publishers/logos/613fa6f5-fea2-4e4a-a934-ba44a1e85f21.png.200x200_q95_detail_letterbox_upscale.png" alt="Cell2020" height="80" hspace="40"/></a>
</p>

See also our [media gallery](https://christophkirst.github.io/ClearMap2Documentation/html/media.html).


References
==========

See who cites us: 

  * [ClearMap 1.0](https://scholar.google.com/scholar?cites=14871582180549937567&as_sdt=2005&sciodt=0,5&hl=en)
  
  * [ClearMap 2.0](https://scholar.google.com/scholar?cites=15218093461598622032&as_sdt=2005&sciodt=0,5&hl=en)
  
and cite us if you use the sofware in any form:

    @article{kirst2020mapping,
       title={Mapping the fine-scale organization and plasticity of the brain vasculature},
       author={Kirst, Christoph and Skriabine, Sophie and Vieites-Prado, Alba and Topilko, Thomas and Bertin, Paul and Gerschenfeld, Gaspard and Verny, Florine and Topilko, Piotr and Michalski, Nicolas and Tessier-Lavigne, Marc and others},
       journal={Cell},
       volume={180},
       number={4},
       pages={780--795},
       year={2020},
       publisher={Elsevier},
       url={https://doi.org/10.1016/j.cell.2016.05.007}}
    
    @article{renier2016mapping,
       title={Mapping of brain activity by automated volume analysis of immediate early genes},
       author={Renier, Nicolas and Adams, Eliza L and Kirst, Christoph and Wu, Zhuhao and Azevedo, Ricardo and Kohl, Johannes and Autry, Anita E and Kadiri, Lolahon and Venkataraju, Kannan Umadevi and Zhou, Yu and others},
       journal={Cell},
       volume={165},
       number={7},
       pages={1789--1802},
       year={2016},
       publisher={Elsevier},
       url={https://doi.org/10.1016/j.cell.2020.01.028}}


Authors
=======

ClearMap was originally designed and developed by 
[Christoph Kirst](https://profiles.ucsf.edu/christoph.kirst). 

Scripts and specific applications were developed by [Nicolas Renier](https://www.renier-lab.com/nicolasrenier)
and [Christoph Kirst](https://profiles.ucsf.edu/christoph.kirst).

The deep vessel filling network was designed and created by Sophie Skriabine 
and integrated to ClearMap by [Christoph Kirst](https://profiles.ucsf.edu/christoph.kirst).

The documentation was written by [Christoph Kirst](https://profiles.ucsf.edu/christoph.kirst).
and [Nicolas Renier](https://www.renier-lab.com/nicolasrenier).

Contributions are very welcome.


License
=======

This project is licensed under the [GNU General Public License v3.0](LICENSE). 

For other options contact the author Christoph Kirst 
(christoph.kirst.ck@gmail.com).

Copyright © 2020 by Christoph Kirst

Versions
========

VERSION 2.0
-----------
Rewrite of ClearMap 1.0 to handle larger data sets (TB).
This version implements [TubeMap](https://christophkirst.github.io/ClearMap2Documentation/html/tubemap.html)

VERSION 1.0
-----------
First version of ClearMap. Implements [CellMap](https://christophkirst.github.io/ClearMap2Documentation/html/cellmap.html) 
See https://github.com/ChristophKirst/ClearMap

