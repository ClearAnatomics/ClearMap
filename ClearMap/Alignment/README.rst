The :mod:`ClearMap.Alignment` module includes methods for

* 3d resampling:

  - :mod:`~ClearMap.Alignment.Resampling`

* 3d alignment to reference atlases (e.g.
  `Allen Brain Institute Atlases <https://portal.brain-map.org/>`_) via
  interfacing to
  `elastix <https://github.com/SuperElastix/elastix>`_:

  - :mod:`~ClearMap.Alignment.Elastix`
  - :mod:`~ClearMap.Alignment.Annotation`

* 3d rigid and wobbly stitching via :doc:`/advanced/wobblystitcher`:

  - :mod:`~ClearMap.Alignment.Stitching.StitchingRigid`
  - :mod:`~ClearMap.Alignment.Stitching.StitchingWobbly`