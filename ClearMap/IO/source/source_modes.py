from ClearMap.Utils.Formatting import ensure
from ClearMap.Utils.exceptions import ClearMapValueError

VALID_MODES         = ('r', 'c', 'r+', 'w+')
EXISTING_FILE_MODES = ('r', 'c', 'r+')
CREATING_MODES      = ('w+',)            # truncate-or-create; one-shot, never stored
READ_ONLY_MODES     = ('r',)
WRITABLE_MODES      = ('c', 'r+', 'w+')  # 'c' accepts writes but does not save
PERSISTABLE_MODES   = ('r+', 'w+')       # writes actually reach disk.  Not 'C' because 'C' is copy-on-write in memory
DEFAULT_READ_MODE   = 'r'
DEFAULT_EDIT_MODE   = 'r+'


def validate_mode(mode, *, allow_none=False, context=''):
    """Normalise and check a mode string; raises rather than letting np.memmap decide."""
    if mode is None:
        if allow_none:
            return None
        raise ClearMapValueError(f'{context or "mode"}: a mode is required.', value=mode, expected=VALID_MODES)
    mode = ensure(mode, str)
    if mode not in VALID_MODES:
        raise ClearMapValueError(f'{context or "mode"}: invalid mode {mode!r}.', value=mode, expected=VALID_MODES)
    return mode


def mode_after_create(mode):
    """The mode a source must carry *after* a create call has consumed its creation intent.

    ``'w+'`` is a one-shot instruction: retaining it means every later reopen,
    ``as_real()`` or relocation re-truncates the file. Everything else passes through
    unchanged, including ``None``.
    """
    return DEFAULT_EDIT_MODE if mode in CREATING_MODES else mode
