"""
The assets (files of the workspace) a processing step needs before it runs: FilePath describes one,
``requires_assets`` refuses to run the step if one is missing.
"""
import os

from ClearMap.Utils.exceptions import MissingRequirementException
from ClearMap.Utils.tag_expression import Expression


# FIXME: rename to smth similar to AssetSpec (without conflicting with the existing AssetSpec)
class FilePath:
    def __init__(self, base, prefix=None, postfix=None, extension=None, asset_sub_type=None):
        self.base = base
        self.prefix = prefix
        self.postfix = asset_sub_type or postfix
        self.extension = extension


def requires_assets(asset_specs):
    """
    Decorator to check if the required files exist before running the function.
    For the case of channel, it can be extracted (in this order) from
    - the kwargs
    - the FilePath object
    - the instance to which the wrapped function belongs

    Parameters
    ----------
    asset_specs: List[FilePath]
        List of FilePath objects
    """
    def decorator(func):
        def wraps(*args, **kwargs):
            instance = args[0]
            workspace = instance.workspace
            for f_p in asset_specs:
                channel = kwargs.get('channel')  # Get directly from the kwargs
                if channel is None:
                    if hasattr(f_p, 'channel'):  # Get from FilePath object
                        channel = f_p.channel
                    else:  # Get the global value from the instance
                        channel = getattr(instance, 'channel', None)
                f_path = workspace.filename(f_p.base, channel=channel, sample_id=f_p.prefix, asset_sub_type=f_p.postfix,
                                            extension=f_p.extension)
                msg = f'{type(instance).__name__}.{func.__name__} missing path: "{f_path}"'
                if Expression(f_path).tags:
                    file_list = workspace.file_list(f_p.base, sample_id=f_p.prefix, asset_sub_type=f_p.postfix,
                                                    extension=f_p.extension)
                    if not file_list:
                        raise MissingRequirementException(msg + ' Pattern but no file')
                    for f in file_list:
                        if not os.path.exists(f):
                            raise MissingRequirementException(msg + f' Missing tile {f}')
                elif not os.path.exists(f_path):
                    raise MissingRequirementException(msg)
            return func(*args, **kwargs)
        return wraps
    return decorator
