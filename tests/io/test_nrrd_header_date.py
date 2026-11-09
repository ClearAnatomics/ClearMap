"""
The header of a written nrrd file says when it was written, in UTC, without a deprecated call.

It used ``datetime.utcnow()``, deprecated since Python 3.12 (a DeprecationWarning at every nrrd file written).
"""
import datetime
import re
import warnings

import numpy as np

import ClearMap.IO.source.backends.nrrd_backend as nrrd_backend

HEADER_DATE_PATTERN = rb'# on (\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)\(GMT\)\.'
HEADER_DATE_FORMAT = '%Y-%m-%d %H:%M:%S'
MAX_CLOCK_DIFFERENCE = datetime.timedelta(minutes=1)


def test_the_nrrd_header_gives_the_utc_time_of_writing_without_a_deprecation_warning(tmp_path):
    location = tmp_path / 'volume.nrrd'

    with warnings.catch_warnings():
        warnings.simplefilter('error', DeprecationWarning)
        nrrd_backend._write(str(location), np.zeros((2, 2, 2), dtype=np.uint8))

    written = re.search(HEADER_DATE_PATTERN, location.read_bytes()).group(1).decode('ascii')
    written_at = datetime.datetime.strptime(written, HEADER_DATE_FORMAT).replace(tzinfo=datetime.timezone.utc)
    assert abs(datetime.datetime.now(datetime.timezone.utc) - written_at) < MAX_CLOCK_DIFFERENCE
