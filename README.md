ClearMap 3.1
============

[![DOI](https://zenodo.org/badge/256322811.svg)](https://zenodo.org/badge/latestdoi/256322811)
[![GitHub stars](https://img.shields.io/github/stars/ClearAnatomics/ClearMap.svg?style=social&label=Star)](https://github.com/ClearAnatomics/ClearMap)
[![GitHub forks](https://img.shields.io/github/forks/ClearAnatomics/ClearMap.svg?style=social&label=Fork)](https://github.com/ClearAnatomics/ClearMap)
[![Follow on Twitter](https://img.shields.io/twitter/follow/clearmap_idisco?style=social&logo=twitter)](https://twitter.com/intent/follow?screen_name=clearmap_idisco)

[![Generic badge](https://img.shields.io/badge/contributions-welcome-brightgreen.svg)](docs/contribute.md)
[![License](https://img.shields.io/github/license/ClearAnatomics/ClearMap?color=green&style=plastic)](https://github.com/ClearAnatomics/ClearMap/blob/master/LICENSE.txt)
![Size](https://img.shields.io/github/repo-size/ClearAnatomics/ClearMap?style=plastic)
[![Language](https://img.shields.io/github/languages/top/ClearAnatomics/ClearMap?style=plastic)](https://github.com/ClearAnatomics/ClearMap)
[![](https://github.com/ClearAnatomics/RepoTracker/workflows/RepoTracker/badge.svg)](https://github.com/ClearAnatomics/RepoTracker/actions)


<p align="center">
<img src="https://clearanatomics.github.io/ClearMapDocumentation/html/_static/TubeMap_graph_movie_small.gif" height="150">
<img src="https://clearanatomics.github.io/ClearMapDocumentation/html/_static/CellMap_small_fast.gif" height="150">
<img src="https://clearanatomics.github.io/ClearMapDocumentation/html/_static/TubeMap_raw_movie_small.gif" height="150">
</p>

*ClearMap* is a toolbox for the analysis and registration of volumetric
data from cleared tissues.

> **New in 3.1**: Redesigned pipeline orchestrators, YAML-based configuration,
> channel-centred workspace, colocalization analysis, and improved GUI.
> Install by running `./install_gui.sh` and launch with `clearmap-ui`.


## 3.1 — What's new

  * [x] Redesigned pipeline orchestrators replacing the old `processors` package
  * [x] YAML-based configuration with JSON-schema validation and automatic
    derivation of dependent fields
  * [x] Unlimited channel number and various data types with new workspace with typed asset management
  * [x] Colocalization analysis between fluorescence channels
  * [x] Improved atlas registration with landmark support
  * [x] TractMap pipeline for myelinated-tract analysis
  * [x] Batch processing and group analysis improvements
  * [x] Important performance optimisations in vasculature pipeline
  * [x] Improved documentation with full API reference


## Pipelines

### [Wobbly-Stitcher](https://clearanatomics.github.io/ClearMapDocumentation/html/advanced/wobblystitcher.html) <a href="https://clearanatomics.github.io/ClearMapDocumentation/html/advanced/wobblystitcher.html"><img src="https://clearanatomics.github.io/ClearMapDocumentation/html/_static/WobblyStitcher_non-rigid.jpg" alt="WobblyStitcher" width="150" align="right" vspace="5"/></a>

Stitch terabyte-scale datasets non-rigidly using the WobblyStitcher algorithm.

### [TubeMap](https://clearanatomics.github.io/ClearMapDocumentation/html/tubemap.html) <a href="https://clearanatomics.github.io/ClearMapDocumentation/html/tubemap.html"><img src="https://clearanatomics.github.io/ClearMapDocumentation/html/_static/TubeMap_graph_movie_small.gif" alt="TubeMap" width="150" align="right" vspace="5"/></a>

Extract vasculature and other tubular networks from terabyte-scale data,
build annotated graphs, and analyse vessel morphology.

### [CellMap](https://clearanatomics.github.io/ClearMapDocumentation/html/cellmap.html) <a href="https://clearanatomics.github.io/ClearMapDocumentation/html/cellmap.html"><img src="https://clearanatomics.github.io/ClearMapDocumentation/html/_static/CellMap_small_fast.gif" alt="CellMap" width="150" align="right" vspace="5"/></a>

Detect and map neuronal activity markers and cell shapes across the whole brain.


## Background and Applications

*ClearMap* has been designed to analyse O(TB) 3-D datasets obtained via
light-sheet microscopy from iDISCO+ cleared tissue samples immunolabelled
for proteins.

*ClearMap* has been written for mapping immediate early genes
[Renier et al. Cell 2016](https://doi.org/10.1016/j.cell.2016.05.007)
as well as vasculature networks of whole mouse brains
[Kirst et al. Cell 2020](https://doi.org/10.1016/j.cell.2020.01.028).

<p align="center">
<a href="https://doi.org/10.1016/j.cell.2016.05.007">
<img src="https://clearanatomics.github.io/ClearMapDocumentation/html/_static/cell_abstract_2016.jpg"
 alt="Cell2016" width="300" hspace="40"/></a>
<a href="https://doi.org/10.1016/j.cell.2020.01.028">
<img src="https://clearanatomics.github.io/ClearMapDocumentation/html/_static/cell_abstract_2020.jpg"
 alt="Cell2020" width="300" hspace="40"/></a>
</p>

*ClearMap* tools may also be useful for data obtained with other types of
microscopes, markers, clearing techniques, as well as other species,
organs, or samples.

*ClearMap* is written in Python 3 and is designed to take advantage of
the parallel processing capabilities of modern workstations. We hope the open 
structure of the code will enable many new modules to be added to *ClearMap*
to broaden the range of applications to different types of biological objects 
or structures.


## Installation

Please refer to the
[installation guide](https://clearanatomics.github.io/ClearMapDocumentation/html/installation.html)
in the documentation.

**Quick start:**

    git clone https://github.com/ClearAnatomics/ClearMap.git
    cd ClearMap
    chmod +x install_gui.sh
    ./install_gui.sh
    conda activate ClearMap3.1
    clearmap-ui

Minimum recommended hardware: 256 GB RAM, 12-core CPU, NVMe SSD storage.
An nVidia GPU with >= 24 GB VRAM is required for the vasculature
deep-filling step.


## Documentation

Full documentation is available at
[clearanatomics.github.io/ClearMap](https://clearanatomics.github.io/ClearMapDocumentation/html/).

For experimental protocols also refer to [idisco.info](https://idisco.info).


## News and Media

<p align="center">
<a href="https://www.ted.com/talks/christoph_kirst_a_transparent_journey_into_the_brain_and_its_flexible_function">
<img src="https://clearanatomics.github.io/ClearMapDocumentation/html/_static/TEDx.png" alt="TEDx" height="200" hspace="40"/></a>
<a href="https://www.youtube.com/watch?v=-LEfL55-EUU">
<img src="https://img.youtube.com/vi/-LEfL55-EUU/0.jpg" alt="Cell2020" height="200" hspace="40"/></a>
</p>

ClearMap has been featured in various articles and media:

<p align="center">
<a href="https://www.nature.com/articles/s41684-020-0556-7">
<img src="https://media.springernature.com/full/nature-cms/uploads/product/nature/header-86f1267ea01eccd46b530284be10585e.svg"
  alt="Nature News & Views" height="80" hspace="40"/></a>
<a href="https://www.sciencedirect.com/science/article/pii/S0092867416307371">
<img src="https://publons.com/media/thumbs/publishers/logos/613fa6f5-fea2-4e4a-a934-ba44a1e85f21.png.200x200_q95_detail_letterbox_upscale.png"
  alt="Cell" height="80" hspace="40"/></a>
</p>

See our [media gallery](https://clearanatomics.github.io/ClearMapDocumentation/html/media.html).


## References

See who cites us:

  * [ClearMap 1.0](https://scholar.google.com/scholar?cites=14871582180549937567&as_sdt=2005&sciodt=0,5&hl=en)
  
  * [ClearMap 2.0](https://scholar.google.com/scholar?cites=15218093461598622032&as_sdt=2005&sciodt=0,5&hl=en)

Please cite us if you use the software:

    @article{kirst2020mapping,
       title={Mapping the fine-scale organization and plasticity of the brain vasculature},
       author={Kirst, Christoph and Skriabine, Sophie and Vieites-Prado, Alba and
               Topilko, Thomas and Bertin, Paul and Gerschenfeld, Gaspard and
               Verny, Florine and Topilko, Piotr and Michalski, Nicolas and
               Tessier-Lavigne, Marc and others},
       journal={Cell},
       volume={180},
       number={4},
       pages={780--795},
       year={2020},
       publisher={Elsevier},
       url={https://doi.org/10.1016/j.cell.2020.01.028}}

    @article{renier2016mapping,
       title={Mapping of brain activity by automated volume analysis of immediate early genes},
       author={Renier, Nicolas and Adams, Eliza L and Kirst, Christoph and Wu, Zhuhao and
               Azevedo, Ricardo and Kohl, Johannes and Autry, Anita E and Kadiri, Lolahon and
               Venkataraju, Kannan Umadevi and Zhou, Yu and others},
       journal={Cell},
       volume={165},
       number={7},
       pages={1789--1802},
       year={2016},
       publisher={Elsevier},
       url={https://doi.org/10.1016/j.cell.2016.05.007}}


## Authors

ClearMap was originally designed and developed by
[Christoph Kirst](https://profiles.ucsf.edu/christoph.kirst).

Scripts and specific applications were developed by
[Nicolas Renier](https://www.renier-lab.com/nicolasrenier)
and [Christoph Kirst](https://profiles.ucsf.edu/christoph.kirst).

Version 2.1 GUI and pipeline redesign by
[Charly Rousseau](https://github.com/crousseau) and
[Etienne Doumazane](https://github.com/doumazane),
with group analysis scripts contributed by
[Sophie Skriabine](https://github.com/skriabineSop).

The deep vessel filling network was designed by
[Sophie Skriabine](https://github.com/skriabineSop) and integrated by
[Christoph Kirst](https://profiles.ucsf.edu/christoph.kirst).

Colocalization analysis by [Gaël Cousin](https://github.com/gaelccousin)
and [Charly Rousseau](https://github.com/crousseau).

The documentation was written by [Christoph Kirst](https://profiles.ucsf.edu/christoph.kirst).
and [Nicolas Renier](https://www.renier-lab.com/nicolasrenier) and the ClearMap team.

Contributions are very welcome.


## License

This project is licensed under the
[GNU General Public License v3.0](LICENSE.txt).

For other licensing options contact
[Christoph Kirst](mailto:christoph.kirst.ck@gmail.com).

Copyright © 2020-2024 by Christoph Kirst and the ClearMap team.


## Version history

### VERSION 3.1
  * Pipeline orchestrators replacing the old `processors` package
  * YAML-based configuration with schema validation
  * Channel-centred workspace and typed asset management
  * Colocalization, TractMap, and batch processing improvements
  * Full API documentation

### VERSION 2.1
  * Graphical user interface
  * Configuration-file-based parameters replacing scripts
  * Updated Allen atlas, hemisphere support, landmark registration
  * Batch processing and group analysis

### VERSION 2.0
  * Full rewrite of ClearMap 1.0 for terabyte-scale datasets
  * Implements TubeMap vasculature analysis

### VERSION 1.0
  * First release. Implements [CellMap](https://christophkirst.github.io/ClearMap2Documentation/html/cellmap.html)
  * See https://github.com/ChristophKirst/ClearMap
