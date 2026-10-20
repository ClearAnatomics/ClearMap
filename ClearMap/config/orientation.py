"""
The orientation of a channel: the permutation and signs of the axes of the raw images that bring them in the
orientation of the atlas (0 for an axis not defined yet).
"""
import warnings

from ClearMap.Utils.exceptions import ParamsOrientationError


DEFAULT_ORIENTATION = (0, 0, 0)


def validate_orientation(orientation, channel, raise_error=True):
    """
    Check that the orientation does not have redundant axes

    Parameters
    ----------
    orientation: tuple(int)
        The orientation to check
    """
    defined_axes = [abs(e) for e in orientation if e != 0]
    if len(defined_axes) != len(set(defined_axes)):
        if raise_error:
            raise ParamsOrientationError(f'Number of different defined axes in {orientation} is'
                                         f'{len(set(defined_axes))}. Defined axes cannot be duplicated. '
                                         f'Please amend duplicate axes.', channel=channel)
        else:
            warnings.warn(f'Invalid orientation {orientation} for {channel},'
                          f' using default {DEFAULT_ORIENTATION}')
            return DEFAULT_ORIENTATION
    return orientation
