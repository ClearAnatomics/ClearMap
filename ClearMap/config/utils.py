"""
A set of low level utilities for configuration file parsing and handling.

.. warning::

    No heavy imports here. This is used by the early_boot which sets temp_dir.
    If using modules outside the standard library, import them locally inside functions.

"""

import re

from ClearMap.Utils.exceptions import ConfigParsingError


def configobj_to_dict(obj):
    import configobj  # avoids hard dependency if not used
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


def parse_flat_yaml(text: str) -> dict:
    out = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):  # skip comments / empty lines
            continue
        if ":" not in line:  # skip malformed lines
            continue

        k, v = line.split(":", 1)
        k = k.strip()
        v = v.strip()
        if v.lower() == "null":  # None
            out[k] = None
        elif v.lower() in ("true", "false"):  # bool
            out[k] = (v.lower() == "true")
        elif ((v.startswith('"') and v.endswith('"')) or
              (v.startswith("'") and v.endswith("'"))):  # string
            out[k] = v[1:-1]
        else: # try int, then float, else raw string
            try:
                out[k] = int(v)
            except ValueError:
                try:
                    out[k] = float(v)
                except ValueError:
                    if '[' in v or '{' in v:  # likely a list or dict
                        raise ValueError(f"Unsupported complex value in flat YAML: {v}")
                    else:
                        out[k] = v  # Assume string
    return out


def write_flat_yaml(cfg_path, data: dict):
    lines = []
    for k, v in data.items():
        if v is None:
            lines.append(f"{k}: null")
        elif isinstance(v, bool):
            lines.append(f"{k}: {'true' if v else 'false'}")
        elif isinstance(v, (int, float)):
            lines.append(f"{k}: {v}")
        elif isinstance(v, str):
            lines.append(f'{k}: "{v}"')
        else:
            raise ValueError(f"Unsupported value type in flat YAML: {type(v)}")
    content = "\n".join(lines) + "\n"
    with cfg_path.open('w', encoding='utf-8') as fh:
        fh.write(content)