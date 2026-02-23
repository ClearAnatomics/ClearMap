# TODO module-level documentation: This module is an example of a custom pipeline
"""
Example usage:
```
from ClearMap.Scripts.custom_cells import detect_cells, custom_detect_cells_block, default_cell_detection_parameter
from skimage.data import cells3d

test_data = cells3d()[:,1]
sink = detect_cells(test_data, block_detection_function=custom_detect_cells_block, cell_detection_parameter=cell_detection_parameter)
```
"""

import gc

import numpy as np

from skimage.filters import threshold_otsu
from skimage.morphology import (
    remove_small_objects,
    remove_small_holes,
    binary_closing,
    ball,
)
from skimage.measure import label, regionprops

from ClearMap.ImageProcessing.Experts.Cells import (
    default_cell_detection_parameter,
    detect_cells,
)  # We need `detect_cells` imported from the same module as `custom_detect_cells_block`
import ClearMap.Utils.Timer as tmr
from ClearMap.ImageProcessing.Experts.utils import run_step


default_cell_detection_parameter["custom_step"] = {
    "save": False
}  # The `cell_detection_parameter` dictionnary needs a key with the name of the new custom step


def custom_detection(array, **kwargs):
    """Detect cells in a 3D array.
    Outputs a tuple of arrays representing the coordinates (x, y, z) of the cells **in the array**,
    and properties of each detected cell."""

    ## detect intensities above otsu threshold
    mask = array > threshold_otsu(array)

    ## postprocess mask
    mask = remove_small_objects(mask, min_size=1000)
    mask = binary_closing(mask, ball(1))
    mask = remove_small_holes(mask, area_threshold=10_000)

    ## extract individual connected components
    labeled_mask = label(mask.astype(int), connectivity=1)

    ## compute properties of the connected components
    region_properties = regionprops(labeled_mask, array)
    coordinates = np.array([region.centroid for region in region_properties])
    intensities = np.array([region.mean_intensity for region in region_properties])
    maxima = np.array([region.max_intensity for region in region_properties])
    shapes = np.array([region.area for region in region_properties])

    return coordinates, intensities, maxima, shapes


def custom_detect_cells_block(source, parameter=default_cell_detection_parameter, n_threads=None):
    """Detect cells in a block of a 3D image.
    Outputs a tuple of arrays, representing the coordinates (x, y, z) of the cells **in the image**,
    and properties of each detected cell.
    That function can be passed to CellMap's `detect_cells` function."""

    # initialize parameter and slicing
    if parameter.get("verbose"):
        prefix = "Block %s: " % (source.info(),)
        total_time = tmr.Timer(prefix)
    else:
        prefix = ""

    base_slicing = source.valid.base_slicing
    valid_slicing = source.valid.slicing
    valid_lower = source.valid.lower
    valid_upper = source.valid.upper
    lower = source.lower

    # Measurements that will be performed per cell
    steps_to_measure = {}  # FIXME: rename

    ## intensity
    parameter_intensity = parameter.get("intensity_detection")
    if parameter_intensity:
        parameter_intensity = parameter_intensity.copy()
        measure = parameter_intensity.pop("measure", [])
        measure = measure if measure else []
        ## validation
        valid_measurement_keys = list(default_cell_detection_parameter.keys()) + [
            "source"
        ]
        for m in measure:
            if m not in valid_measurement_keys:
                raise KeyError(f"Unknown measurement: {m}")
            steps_to_measure[m] = None
    ## in case source is measured, the image used is source
    if "source" in steps_to_measure:
        steps_to_measure["source"] = np.array(source.array)
    ## other cases seem to not be supported

    step_params = {
        "parameter": parameter,
        "steps_to_measure": steps_to_measure,
        "prefix": prefix,
        "base_slicing": base_slicing,
        "valid_slicing": valid_slicing,
    }

    # WARNING: if param_illumination: previous_step = source, not np.array(source.array)

    results = run_step(
        "custom_step", np.array(source.array), custom_detection, **step_params
    )

    # correct coordinate offsets of blocks
    results = (results[0] + lower,) + results[1:]

    # remove cells outside the valid region of the block
    valid_mask = np.all(
        (results[0] >= valid_lower) & (results[0] < valid_upper), axis=1
    )
    results = tuple(result[valid_mask] for result in results)

    # ensure all results array have 2 dimensions, so that they are ready to be vstacked
    results = tuple(
        result[:, None] if result.ndim == 1 else result for result in results
    )

    if parameter.get("verbose"):
        total_time.print_elapsed_time("Cell detection")

    gc.collect()
    return results
