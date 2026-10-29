"""
ExitCode gives the exit status of the command line programs: the numbers a shell and a job scheduler read are the
conventional ones (0 for success, a non-zero status for a failure, 2 for a wrong command line as argparse exits).
"""
from ClearMap.Utils.utilities import ExitCode


def test_the_exit_codes_are_the_conventional_numbers():
    assert (ExitCode.SUCCESS, ExitCode.FAILURE, ExitCode.USAGE_ERROR) == (0, 1, 2)

