"""
An array with an extra axis (a channel of size 1, a tuple saved as one array) reached the compiled code of the
skeleton and the graph, which failed with "operands could not be broadcast together with shapes (3,) (4,)" (#75, #76).
The check names the argument, the shape and what to do.
"""
import re

import numpy as np
import pytest

from ClearMap.Utils import array_checks
from ClearMap.Utils.exceptions import ClearMapValueError


def test_an_array_with_the_required_number_of_dimensions_is_returned_as_is():
    skeleton = np.zeros((4, 5, 6), dtype=bool)

    assert array_checks.check_ndim(skeleton, 3) is skeleton


@pytest.mark.parametrize('shape', [(4, 5, 6, 1), (4, 5), (1, 4, 5, 6)], ids=str)
def test_an_array_with_another_number_of_dimensions_is_refused_with_its_name_and_shape(shape):
    expected = f'The skeleton array has {len(shape)} dimensions (shape {shape}), 3 are required'
    with pytest.raises(ClearMapValueError, match=re.escape(expected)):
        array_checks.check_ndim(np.zeros(shape, dtype=bool), 3, name='The skeleton array')
