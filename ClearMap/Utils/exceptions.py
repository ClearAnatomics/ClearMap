class ClearMapException(Exception):
    pass


class MissingRequirementException(ClearMapException):
    pass


class ConfigNotFoundError(ClearMapException):
    pass


class PlotGraphError(ClearMapException):
    pass
