=========
Changelog
=========

3.1.2
-----

Bug fixes only; the configuration files and the results of 3.1.1 are unchanged.

- GUI: opening an experiment (or a group of experiments) written by ClearMap 3.1.x no longer reports "Old version
  detected" and offers to upgrade it. The check compared the ``clearmap_version`` of the configs with the version of the
  package, but the configs only change with the major.minor of ClearMap, so 3.1.1 flagged its own configs. A config
  written by a newer ClearMap than the one which reads it is refused with an error which names both versions.

3.1.1
-----

Bug fixes only; the configuration files and the results of 3.1.0 are unchanged.

- Stitching: a tile pattern which matches no file stops with an error which names the pattern (#193).
- Reading planes from a list of files: the planes of a list land at positions 0..n-1 of the result, and a numbering
  with gaps (planes 0, 20, 40, ...) is read by position (#73).
- ``Workspace.info()`` no longer fails on tile patterns with string tags and prints plain integers for the tile range
  (#58).
- CellMap: measuring the intensity after a disabled step is refused. It used to give the intensity of the source (#60).
- Vasculature: ``skeletonize(..., radii=True)`` no longer fails on an unexpected ``out_dtype`` argument, and
  ``graph_from_skeleton`` refuses an array which is not 3D and says what it received (#75, #76).

3.1.0
-----

New features
~~~~~~~~~~~~

**New pipelines and analyses**

- **TractMap**: analyse myelinated tracts. Binarization with percentile clipping, coordinate extraction, transformation to
  the atlas (in parallel), labeling, voxelization and export, with a tab and plots of the binary file.
- **Colocalization**: compare two channels, from a tab or with the example script. Large samples are processed
  through memory maps.
- **Group analysis**: choose the density maps by pipeline, channel and density suffix, split densities by filtering
  criteria, and get p-value maps.

**Vasculature (TubeMap)**

- **Reorder the binarization steps** in the GUI.
- Faster graph extraction (parallel) and graph cleaning, with lengths kept and consistent units for the radii.
- Chain several graph filters, including for voxelization.
- A dedicated **veins channel**.

**CellMap**

- The advanced ``h_max`` parameter and the option to remove the crust are available.
- Weight the voxelization with a column of the cell table.
- Preview the cell detection on a subset, and stop it.

**Stitching and registration**

- Stock elastix 5.3 and later works (binary point files on every platform).
- Registration with landmarks. Choose which channel moves, relative to the channel (radio buttons).
- Stitching: ``auto`` overlaps are read from the OME metadata of the tiles, a single row or column of tiles can be
  previewed, tiles stored as one file per Z plane are stacked, and a "Convert to ClearMap" button prepares a raw source.

**Samples and channels**

- Any number of channels, with data types. Add, remove and rename them at will.
- Settings for the performance (processes, block sizes) are set per sample and per pipeline.

**Clusters and scripts**

- ``python -m ClearMap.run`` runs a pipeline without the GUI, with a Slurm job script, and ClearMap runs over SSH.
- The number of processes follows the CPUs of the job, never more: the defaults written for a workstation run
  unedited in a job. Progress is logged when there is no GUI.
- ``register()`` and ``stitch()`` run the whole registration and the whole stitching from a script.

Problems that are gone
~~~~~~~~~~~~~~~~~~~~~~

- Several experiment load crashes.
- Progress bar overshoot ("step 5/4").
- Stitching: the layout was not saved when the wobbly stitching was skipped,
  and tiles that are not OME failed the ``auto`` overlap.
- Renaming or removing a channel left stale pages in the tabs, and a rename could overwrite a file silently.
- After an Abort, the next runs stopped after their first step (CellMap, TractMap, stitching and the registration button).
- Adding a vasculature graph filter, or voxelizing with several graph filters, crashed. So did editing a group name.
- Registration: the landmark weights were not converted for elastix, and channels aligned in a loop ran forever.
- About 340 bug fixes in all.

What makes your life easier
~~~~~~~~~~~~~~~~~~~~~~~~~~~

- **Drop an experiment or group folder on the window** to open it: experiment or cohort mode is detected for you.
- **Errors explain themselves**: errors are now displayed as dialogs with a title, a hint and
buttons that make sense (retry, skip, abort, reset the workspace), with the technical details one click away.
- **Renaming a channel renames its files**, after you have checked the list (untick or edit each rename; Cancel is
  the default). The renames are all done or all undone.
- Channels processed together (TubeMap, colocalization) are checked to match pixel for pixel before they are combined.
- Using a registration computed the other way warns you.
- The configuration is validated and errors are caught early. Older configuration files are converted to the
  3.1 format.
- Faster start. Reoriented atlases are shared between experiments.
- A screenshot tool in the viewer, busy dialogs while heavy plots are prepared, warnings in yellow and errors in red.
- For clusters: ``CLEARMAP_TMP`` or ``TMPDIR`` sets the temporary folder, ``CLEARMAP_TARGET_ARCHITECTURE`` the CPU the
  compiled code is built for, ``CLEARMAP_CFG_DIR`` the folder of your configuration.

Known limits
~~~~~~~~~~~~

- Windows: elastix works with the stock releases; the graph library (``graph_tool``) and the installer are not ported yet.
  Planned for v3.4.
- NumPy 2 is not supported yet; use NumPy 1.x. Planned for v3.2.
