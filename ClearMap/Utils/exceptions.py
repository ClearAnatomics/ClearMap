"""
Custom exceptions for ``ClearMap``
"""


class ClearMapException(Exception):
    """
    Base exception for all exceptions in ClearMap
    """
    pass


class ClearMapValueError(ClearMapException, ValueError):
    """
    Base exception for all exceptions related to value errors
    """
    pass


class ClearMapIoException(ClearMapException, IOError):
    """
    Base exception for all exceptions related to input/output operations
    """
    pass


class SourceModuleNotFoundError(ClearMapIoException):
    """
    Exception raised when a source module is not found
    """
    pass


class IncompatibleSource(ClearMapIoException):
    def __init__(self, source, variable_name, current_vars):
        self.source = source
        self.variable_name = variable_name
        self.current_value = current_vars.get(variable_name)
        self.source_value = getattr(source, variable_name, None)
        super().__init__(self._generate_message())

    def _generate_message(self):
        return (f'Incompatible {self.variable_name} {self.current_value} != '
                f'{self.source_value} for the source {self.source}!')


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


class ClearMapWorkspaceError(ClearMapException):
    """
    Exception raised when there is an error with the workspace
    """
    pass


class ClearMapAssetError(ClearMapWorkspaceError):
    """
    Exception raised when there is an error with the asset
    """
    pass


class AssetNotFoundError(ClearMapAssetError):
    """
    Exception raised when an asset is not found
    """
    pass


class ParamsOrientationError(ClearMapValueError):
    def __init__(self, message, channel):
        # self.message = message
        self.channel = channel
        super().__init__(message)
