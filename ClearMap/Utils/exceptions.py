"""
Custom exceptions for ``ClearMap``
"""


class ClearMapException(Exception):
    """
    Base exception for all exceptions in ClearMap
    """
    pass


class MissingRequirementException(ClearMapException):
    """
    Exception raised when a processing step is missing a required condition (i.e. an other step has not been run yet)
    """
    pass


class ConfigNotFoundError(ClearMapException):
    """
    Exception raised when a configuration file is not found
    """
    pass


class PlotGraphError(ClearMapException):
    """
    Exception raised when attempting to plot a graph that is e.g. too large
    """
    pass


class ClearMapVRamException(ClearMapException):
    """
    Exception raised when there is insufficient VRAM available for a processing step
    """
    pass


class SmiError(ClearMapException):
    """
    Exception raised when there is an error with the ``nvidia-smi`` command
    """
    pass


class GroupStatsError(ClearMapException):
    """
    Exception raised when there is an error with the group statistics
    """
    pass
