"""
exceptions
==========

Custom exceptions for ``ClearMap``.

Every exception carries presentation metadata so the GUI can display
user-friendly dialogs with actionable recovery options.

Hierarchy (indentation = inheritance)
-------------------------------------
ClearMapException
├── ClearMapValueError  (also ValueError)
│   └── ParamsOrientationError
├── ClearMapRuntimeError  (also RuntimeError)
├── ClearMapIoException  (also IOError)
│   ├── SourceModuleNotFoundError
│   ├── IncompatibleSource
│   ├── MetadataError
│   │   └── NotAnOmeFile
│   └── NrrdError
├── ClearMapWorkspaceError
│   ├── ClearMapAssetError
│   │   └── AssetNotFoundError  (also FileNotFoundError, via MissingRequirementException)
│   ├── MissingChannelError
│   └── MissingAssetError  (also KeyError, via MissingRequirementException)
├── MissingRequirementException
├── ClearMapConfigError
│   ├── ConfigNotFoundError
│   ├── ConfigParsingError
│   ├── AggregatedValidationError
│   └── PatchConflictError
├── PlotGraphError
├── ClearMapVRamException
├── SmiError
└── GroupStatsError
"""
from __future__ import annotations

import inspect
from enum import Enum
from typing import List, Iterable, Tuple
from dataclasses import dataclass

_SKIP_PATTERNS = ('wrapper', '_decorator', 'decorator', '__')


def get_caller_name(skip_patterns: tuple[str, ...] = _SKIP_PATTERNS) -> str:
    """
    Walk up the call stack and return the first meaningful caller name.
    Skips frames whose ``co_name`` contains any of *skip_patterns*
    (typically wrappers, decorators and dunder methods).
    Returns ``'<unknown>'`` when no suitable frame is found.
    """
    frame = inspect.currentframe()
    try:
        f = frame.f_back.f_back if frame and frame.f_back else None
        while f is not None:
            name = f.f_code.co_name
            if not any(p in name for p in skip_patterns):
                cls = f.f_locals.get('self')
                if cls is not None:
                    return f'{type(cls).__name__}.{name}'
                return name
            f = f.f_back
        return '<unknown>'
    finally:
        del frame   # avoid reference cycles


class Severity(Enum):
    """How badly the error affects the current session.

    The GUI dialog adapts its appearance and available actions accordingly:

    RECOVERABLE
        Retry or skip possible; no data is at risk.
    DEGRADED
        Processing can continue with reduced functionality.
    FATAL
        The current operation must stop, but the workspace is intact.
    CRITICAL
        The workspace / experiment may be corrupt; suggest reset.
    """
    RECOVERABLE = 'recoverable'
    DEGRADED    = 'degraded'
    FATAL       = 'fatal'
    CRITICAL    = 'critical'


@dataclass(frozen=True)
class RecoveryOption:
    """One button the user can click in the exception dialog."""
    label: str                       # button text
    action: str                      # identifier returned to the caller
    tooltip: str = ''
    is_default: bool = False         # gets keyboard focus
    is_destructive: bool = False     # styled in red


# Pre-built singletons — import these in raise-sites or exception classes.
DISMISS = RecoveryOption(
    'OK', 'dismiss', is_default=True,
)
RETRY = RecoveryOption(
    'Retry', 'retry',
    tooltip='Attempt the same operation again',
)
SKIP = RecoveryOption(
    'Skip', 'skip',
    tooltip='Skip this step and continue',
)
ABORT = RecoveryOption(
    'Abort', 'abort',
    is_destructive=True,
)
RESET_WORKSPACE = RecoveryOption(
    'Reset Workspace & Reload', 'reset_workspace',
    tooltip='Delete workspace.yml, then reload the experiment from config files',
    is_destructive=True,
)
CLOSE_APP = RecoveryOption(
    'Close Application', 'close',
    is_destructive=True,
)


# Sensible defaults per severity.
DEFAULT_RECOVERY_OPTIONS: dict[Severity, tuple[RecoveryOption, ...]] = {
    Severity.RECOVERABLE: (DISMISS,),
    Severity.DEGRADED:    (DISMISS,),
    Severity.FATAL:       (DISMISS,),
    Severity.CRITICAL:    (RESET_WORKSPACE, CLOSE_APP),
}


# ---------------------------------------------------------------------------
# Base exception
# ---------------------------------------------------------------------------

class ClearMapException(Exception):
    """
    Base for every ClearMap-specific exception.

    Class attributes
    ----------------
    severity : Severity
        How the GUI should present this error.
    user_title : str
        Short, user-facing heading shown in the dialog title bar.
    user_hint : str
        Actionable advice shown below the error message.

    Properties
    ----------
    user_message : str
        Derived from ``str(self)``; subclasses may enrich it with
        structured fields (asset name, channel, etc.).
    recovery_options : tuple[RecoveryOption, ...]
        Buttons offered to the user.  Defaults come from
        ``DEFAULT_RECOVERY_OPTIONS[self.severity]`` but can be
        overridden per-class or per-instance.
    """
    severity: Severity = Severity.FATAL
    user_title: str = 'ClearMap Error'
    user_hint: str = ''

    # Set on *instances* to override class-level recovery_options.
    _instance_options: tuple[RecoveryOption, ...] | None = None

    @property
    def user_message(self) -> str:
        """What goes into the dialog body.  Override for richer messages."""
        return str(self)

    @property
    def recovery_options(self) -> tuple[RecoveryOption, ...]:
        if self._instance_options is not None:
            return self._instance_options
        return DEFAULT_RECOVERY_OPTIONS.get(self.severity, (DISMISS,))

    @recovery_options.setter
    def recovery_options(self, options: tuple[RecoveryOption, ...]):
        self._instance_options = options


# ---------------------------------------------------------------------------
# Value / runtime
# ---------------------------------------------------------------------------

class ClearMapValueError(ClearMapException, ValueError):
    """A parameter, argument or computed quantity is outside its valid domain.

    Carries the offending ``value`` and optionally what was ``expected``
    so the dialog can show both to the user.
    """
    severity = Severity.FATAL
    user_title = 'Invalid Value'
    user_hint = 'Check your parameters and try again.'

    def __init__(self, message: str, value=None, expected=None):
        self.value = value
        self.expected = expected
        super().__init__(message)

    @property
    def user_message(self) -> str:
        parts = [str(self)]
        if self.expected is not None:
            parts.append(f'Expected: {self.expected}')
        if self.value is not None:
            parts.append(f'Got: {self.value}')
        return '\n'.join(parts)


class ClearMapRuntimeError(ClearMapException, RuntimeError):
    """An operation failed at runtime (external tool crashed, unexpected state, etc.).

    If triggered by a shell command, ``command`` holds the command line
    for diagnostics.
    """
    severity = Severity.FATAL
    user_title = 'Runtime Error'
    user_hint = 'Check the log files for more details.'

    def __init__(self, message: str, command: str = ''):
        self.command = command
        super().__init__(message)

    @property
    def user_message(self) -> str:
        parts = [str(self)]
        if self.command:
            parts.append(f'Command: {self.command}')
        return '\n'.join(parts)


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

class ClearMapIoException(ClearMapException, IOError):
    """Base exception for failures within the ClearMap.IO dispatcher when reading, writing,
    or initializing ``Source`` objects.
    """
    user_title = 'I/O Error'


class ClearMapFileNotFoundError(ClearMapIoException, FileNotFoundError):
    """Raised when a required file is not found on disk."""
    user_title = 'File Not Found'
    severity = Severity.FATAL


class ClearMapPermissionError(ClearMapIoException, PermissionError):
    """Raised when there are insufficient permissions to read/write a file."""
    user_title = 'Permission Denied'
    user_hint = 'Check file permissions or ensure the file is not opened in another program.'
    severity = Severity.FATAL


class SourceModuleNotFoundError(ClearMapIoException):
    """No reader module is registered for the given file extension.

    Raised by the I/O dispatcher when a file's extension (e.g. ``.xyz``) does not map
    to any registered backend module (like TIF, NRRD, NPY, etc.).
    """
    user_title = 'Unsupported File Format'
    user_hint = 'The file extension is not recognised. Check the file path.'
    severity = Severity.FATAL

    def __init__(self, filename: str = '', ext: str = ''):
        if not filename:
            raise ClearMapValueError(f'filename must be provided to {self.__class__.__name__}', value=filename)
        self.filename = filename
        if not ext:
            ext = filename.rsplit('.', 1)[-1] if '.' in filename else ''
        self.ext = ext
        msg = f'Cannot determine module for file "{filename}" with extension "{ext}"!'
        super().__init__(msg)


class IncompatibleSource(ClearMapIoException):
    """Two data sources that should be combinable differ in a critical property.

    Typically raised when trying to stack or overlay images whose shapes,
    dtypes or coordinate systems do not match.
    """
    user_title = 'Incompatible Data Source'
    severity = Severity.FATAL

    def __init__(self, source, variable_name, current_vars):
        self.source = source
        self.variable_name = variable_name
        self.current_value = current_vars.get(variable_name)
        self.source_value = getattr(source, variable_name, None)
        super().__init__(f'Incompatible {self.variable_name} {self.current_value} != '
                         f'{self.source_value} for the source {self.source}!')


class MetadataError(ClearMapIoException):
    """Image metadata (resolution, channel count, etc.) could not be extracted.

    Usually means the file header is truncated, the format is unsupported,
    or the file is simply not an image.
    """
    user_title = 'Metadata Error'
    user_hint = 'The image metadata could not be read. Is the file valid?'
    severity = Severity.FATAL

    def __init__(self, message: str, img_path: str = ''):
        self.img_path = img_path
        super().__init__(message)

    @property
    def user_message(self) -> str:
        parts = [str(self)]
        if self.img_path:
            parts.append(f'File: {self.img_path}')
        return '\n'.join(parts)


# WARNING: class NotAnOmeFile(tifffile.TiffFileError):   Not available in tifffile==0.15.1
class NotAnOmeFile(MetadataError):
    """A file expected to be OME-TIFF lacks the required OME-XML header.

    The file may be a plain TIFF, a BigTIFF or a completely different format.
    Recoverable because the caller can often fall back to non-OME parsing.
    """
    severity = Severity.RECOVERABLE
    user_title = 'Not an OME-TIFF File'
    user_hint = 'Use a valid OME-TIFF file, or select a different format.'

    def __init__(self, message: str, img_path: str = ''):
        super().__init__(message, img_path=img_path)


class NrrdError(ClearMapIoException):
    """An NRRD file could not be parsed — missing header fields, unknown
    encoding, version mismatch or truncated data.
    """
    user_title = 'NRRD File Error'
    user_hint = 'The NRRD file appears to be corrupted or unsupported.'
    severity = Severity.FATAL

    def __init__(self, message: str, field: str = '', encoding: str = ''):
        self.field = field
        self.encoding = encoding
        super().__init__(message)


# ---------------------------------------------------------------------------
# Workspace / assets
# ---------------------------------------------------------------------------

class ClearMapWorkspaceError(ClearMapException):
    """Raised when the ``Workspace2`` instance detects an invalid state, such as a
    malformed folder layout (``resource_type_to_folder``), unregistered pipelines,
    or failed directory migrations.
    """
    user_title = 'Workspace Error'
    user_hint = ('If this persists, try "Reset Workspace" from the '
                 'Workspace menu to rebuild from configuration files.')

    @property
    def recovery_options(self) -> tuple[RecoveryOption, ...]:
        if self._instance_options is not None:
            return self._instance_options
        return (DISMISS, RESET_WORKSPACE)


class ClearMapAssetError(ClearMapWorkspaceError):
    """Raised when an ``Asset`` object is internally inconsistent. Typically caused by
    invalid ``TypeSpec`` or ``ChannelSpec`` definitions, or unresolved tags in a tiled
    file ``Expression`` (e.g., forbidden ``<S>`` tags before parametrization).
    """
    user_title = 'Asset Error'


class MissingChannelError(ClearMapWorkspaceError):
    """Raised when a pipeline or workspace query targets a channel ID that has not
    been added to the workspace's ``AssetCollection``.
    """
    severity = Severity.RECOVERABLE
    user_title = 'Missing Channel'
    user_hint = 'Add the channel in the Sample Info tab, then retry.'


# ---------------------------------------------------------------------------
# Missing prerequisites (the most common user-facing exception)
# ---------------------------------------------------------------------------

class MissingRequirementException(ClearMapException):
    """
    Exception raised when a processing step is missing a required condition (i.e. another step has not been run yet)
    """
    severity = Severity.RECOVERABLE
    user_title = 'Missing Prerequisite'
    user_hint = 'Run the required processing step first.'

    def __init__(self, message: str = '', missing_step_name: str = '', required_by: str = '',
                 context_message: str = '', missing_items: Iterable[str] = (), found_items: Iterable[str] = ()):
        self.missing_step_name = missing_step_name  # e.g. 'registration'
        self.required_by = required_by or get_caller_name() # e.g. 'graph annotation' (typically the caller)
        self.context_message = context_message
        self.missing_items = list(missing_items)
        self.found_items = list(found_items)
        super().__init__(message)

    @property
    def user_message(self) -> str:
        parts = []
        if str(self):
            parts.append(str(self))
        if self.missing_step_name:
            parts.append(f'Missing step: {self.missing_step_name}')

        if self.found_items or self.missing_items:
            parts.append('\nDependencies:')
            parts.extend([f'  ✓ {item}' for item in self.found_items])
            parts.extend([f'  ✗ {item}' for item in self.missing_items])

        if self.context_message:
            parts.append(f'\n{self.context_message}')

        return '\n'.join(parts).strip()

    @property
    def user_hint(self) -> str:
        if self.missing_step_name:
            return f'Please run "{self.missing_step_name}" first, then retry.'
        return 'Run the required processing step to generate the missing files, then retry.'


class MissingAssetError(MissingRequirementException, ClearMapWorkspaceError, KeyError):
    """Raised during workspace lookups when the requested asset type (e.g. 'stitched', 'cells')
    is not configured in the ``AssetCollection`` for the specified channel.

    This represents a logical missing definition in the workspace layout, not a missing physical file.
    """
    severity = Severity.RECOVERABLE
    user_title = 'Missing Asset Definition'

    def __init__(self, message: str, asset_type: str = '', channel: str = '', asset_sub_type: str = '',
                 missing_step_name: str = '', required_by: str = '', workspace_path: str = ''):
        self.asset_type = asset_type
        self.channel = channel
        self.asset_sub_type = asset_sub_type
        self.workspace_path = workspace_path
        MissingRequirementException.__init__(self, message, missing_step_name=missing_step_name,
                                             required_by=required_by)

    @property
    def user_message(self) -> str:
        parts = [str(self)]
        if self.asset_type:
            parts.append(f'Asset type: {self.asset_type}')
        if self.channel:
            parts.append(f'Channel: {self.channel}')
        return '\n'.join(parts)

    @property
    def user_hint(self) -> str:
        step = self.missing_step_name or 'the pipeline step that configures this asset'
        return f'Ensure "{step}" is enabled for this channel.'


class AssetNotFoundError(MissingRequirementException, ClearMapAssetError, FileNotFoundError):
    """The file backing a declared workspace asset (or its compressed variant) does not exist on disk.

    Unlike :class:`MissingAssetError` (lookup failure), this means the
    workspace *does* reference the asset but the expected path is absent —
    typically because results were moved, deleted or never written.
    """
    severity = Severity.RECOVERABLE
    user_title = 'Asset File Not Found'

    def __init__(self, message: str, asset_path: str = '', asset_type: str = '', asset_sub_type: str = '',
                 missing_step_name: str = '', required_by: str = '', workspace_path: str = ''):
        self.asset_path = asset_path
        self.asset_type = asset_type
        self.asset_sub_type = asset_sub_type
        self.workspace_path = workspace_path
        MissingRequirementException.__init__(self, message, missing_step_name=missing_step_name, required_by=required_by)

    @property
    def user_message(self) -> str:
        parts = [str(self)]
        if self.asset_path:
            parts.append(f'Expected at: {self.asset_path}')
        return '\n'.join(parts)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

class ClearMapConfigError(ClearMapException):
    """A configuration file is invalid, missing, or internally inconsistent."""
    user_title = 'Configuration Error'
    user_hint = 'Check the configuration files for errors.'


class ConfigNotFoundError(ClearMapConfigError):
    """The expected ``.cfg`` / ``.yml`` file does not exist at the resolved path.

    This usually means the experiment folder was never initialised or the
    file was manually deleted.
    """
    user_title = 'Configuration File Not Found'


class ConfigParsingError(ClearMapConfigError):
    """A configuration file exists but cannot be deserialised (malformed YAML,
    unknown anchors, encoding issues, etc.).
    """
    user_title = 'Configuration Parsing Error'
    user_hint = 'The configuration file contains invalid syntax or values.'


class AggregatedValidationError(ClearMapConfigError):
    """Multiple validation errors collected during a single schema or
    semantic check pass.  ``messages`` holds one human-readable string
    per individual violation.
    """
    user_title = 'Validation Errors'
    user_hint = 'Fix the errors listed below, then save again.'
    severity = Severity.FATAL

    def __init__(self, messages: Iterable[str]):
        super().__init__('\n'.join(messages))
        self.messages = list(messages)

    # FIXME: replace sig above but adapt call sites
    # def __init__(self, errors: list[str], config_path: str = ''):
    #     self.errors = errors
    #     super().__init__(f'{len(errors)} validation error(s):\n' + '\n'.join(errors), config_path=config_path)

    @property
    def user_message(self) -> str:
        numbered = [f'  {i}. {m}' for i, m in enumerate(self.messages, 1)]
        return f'{len(self.messages)} validation error(s):\n' + '\n'.join(numbered)


class PatchConflictError(ClearMapConfigError):
    """Two concurrent or sequential config patches target the same key with
    incompatible intentions (e.g. one deletes a subtree while another
    updates a value inside it).

    Marked CRITICAL because the in-memory config may already be partially
    applied and the on-disk state could be out of sync.
    """
    severity = Severity.CRITICAL
    user_title = 'Configuration Conflict'
    user_hint = ('The workspace may be in an inconsistent state.\n'
                 'It is recommended to reset the workspace and reload.')

    def __init__(self, path: Tuple[str, ...], src_value: object, dst_value: object, reason: str):
        dotted = '.'.join(path) if path else '<root>'
        super().__init__(f'Patch conflict at "{dotted}": {reason}, {dst_value=}, {src_value=}')
        self.path = path
        self.dst_value = dst_value
        self.src_value = src_value
        self.reason = reason


# ---------------------------------------------------------------------------
# Processing / pipeline
# ---------------------------------------------------------------------------

class PlotGraphError(ClearMapException):
    """The requested graph visualisation cannot be rendered, usually because
    the vertex/edge count exceeds safe memory limits for the 3-D viewer.
    """
    severity = Severity.RECOVERABLE
    user_title = 'Cannot Display Graph'
    user_hint = 'Reduce the data subset or apply stricter filters.'


class ClearMapVRamException(ClearMapException):
    """Not enough free GPU memory to launch a CUDA/OpenCL kernel.

    The user can free VRAM by closing other applications or reducing
    block sizes, then retry.
    """
    severity = Severity.RECOVERABLE
    user_title = 'Insufficient GPU Memory'
    user_hint = 'Close other GPU applications, reduce the batch/block size, or switch to CPU processing if available.'

    @property
    def recovery_options(self) -> tuple[RecoveryOption, ...]:
        if self._instance_options is not None:
            return self._instance_options
        return (RETRY, DISMISS)


class SmiError(ClearMapException):
    """``nvidia-smi`` returned an error or could not be executed.

    GPU monitoring is non-essential, so the session can continue in
    degraded mode without hardware stats.
    """
    severity = Severity.DEGRADED
    user_title = 'GPU Query Failed'
    user_hint = ('nvidia-smi could not be reached. '
                 'GPU monitoring will be disabled for this session.')


class GroupStatsError(ClearMapException):
    """Group-level statistical analysis failed — typically because density
    maps across samples have mismatched shapes, duplicate sample IDs,
    or empty groups.
    """
    user_title = 'Group Statistics Error'
    user_hint = 'Ensure all samples in each group have matching density maps with identical dimensions.'


class ParamsOrientationError(ClearMapValueError):
    """The user-supplied sample orientation axes are invalid.

    A valid orientation is a permutation of ``(±1, ±2, ±3)`` with
    no repeated absolute values.  This error is recoverable because
    the user simply needs to correct the spinboxes.
    """
    severity = Severity.RECOVERABLE
    user_title = 'Invalid Orientation'
    user_hint = 'Each axis must be one of ±1, ±2, ±3 with no duplicates.'

    def __init__(self, message: str, channel: str = ''):
        self.channel = channel
        super().__init__(message)

    @property
    def user_message(self) -> str:
        parts = [str(self)]
        if self.channel:
            parts.append(f'Channel: {self.channel}')
        return '\n'.join(parts)