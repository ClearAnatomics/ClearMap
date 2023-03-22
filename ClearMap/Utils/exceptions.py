class ClearMapException(Exception):
    pass


class MissingRequirementException(ClearMapException):
    pass


class ConfigNotFoundError(ClearMapException):
    pass


class PlotGraphError(ClearMapException):
    pass


class ClearMapVRamException(ClearMapException):
    pass


class SmiError(ClearMapException):
    pass
