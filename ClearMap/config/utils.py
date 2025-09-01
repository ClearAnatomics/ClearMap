import re

import configobj

from ClearMap.Utils.exceptions import ConfigParsingError


def configobj_to_dict(obj):
    """Recursively convert ConfigObj/Section → plain dict/list/scalars."""
    if isinstance(obj, configobj.Section):
        # iterate in insertion order (ConfigObj preserves it)
        return {k: configobj_to_dict(v) for k, v in obj.items()}
    elif isinstance(obj, dict):
        return {k: configobj_to_dict(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [configobj_to_dict(x) for x in obj]
    else:
        return obj


def _handle_configobj_failed_parse(err, cfg_path):
    # Extract line number if available
    msg = str(err)
    line_no = None
    m = re.search(r'line\s+(\d+)', msg)
    if m:
        try:
            line_no = int(m.group(1))
        except ValueError:
            line_no = None

    # Best-effort: show offending line for context
    context = ''
    if line_no is not None and cfg_path.exists():
        try:
            with cfg_path.open('r', encoding='utf-8', errors='replace') as fh:
                lines = fh.readlines()
            if 1 <= line_no <= len(lines):
                context = f'\n→ line {line_no}: {lines[line_no - 1].rstrip()}'
        except Exception:
            pass

    raise ConfigParsingError(
        f'Could not parse config file "{cfg_path}". {msg}{context}'
    ) from err
