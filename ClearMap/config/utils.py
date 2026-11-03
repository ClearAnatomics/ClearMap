"""
A set of low level utilities for configuration file parsing and handling.

.. warning::

    No heavy imports here. This is used by the early_boot which sets temp_dir.
    If using modules outside the standard library, import them locally inside functions.

"""

import os
import re
import shutil
from pathlib import Path

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
    """
    Read a flat YAML mapping (one ``key: value`` per line, no list or mapping) without a YAML library, as the
    early boot must.
    The comments, whole line or after a value, are left out.
    """
    out = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):  # skip comments / empty lines
            continue
        if ":" not in line:  # skip malformed lines
            continue

        k, v = line.split(":", 1)
        out[k.strip()] = _value_of_flat_yaml_text(_split_value_and_comment(v)[0])
    return out


def _split_value_and_comment(text: str) -> tuple[str, str]:
    """
    Split what follows the colon of a flat YAML line in the value and its comment (with its leading spaces).
    A comment starts with a # after a space, outside a quoted value.
    """
    stripped = text.lstrip()
    quote = stripped[:1]
    if quote in ('"', "'"):
        end = 1
        while True:
            end = stripped.find(quote, end)
            if end == -1:
                raise ValueError(f'Unterminated quoted value in flat YAML: {text}')
            if quote == "'" and stripped[end + 1:end + 2] == "'":  # '' is a quote in a single-quoted value
                end += 2
                continue
            if quote == '"' and stripped[end - 1] == '\\':
                end += 1
                continue
            break
        return stripped[:end + 1], stripped[end + 1:]
    for i, char in enumerate(stripped):
        if char == '#' and (i == 0 or stripped[i - 1] in ' \t'):
            value = stripped[:i].rstrip()
            return value, stripped[len(value):]
    return stripped.rstrip(), ''


def _value_of_flat_yaml_text(v: str):
    if v.lower() in ("null", "~", ""):  # None
        return None
    if v.lower() in ("true", "false"):  # bool
        return v.lower() == "true"
    if v.startswith("'"):  # string, '' is a quote
        return v[1:-1].replace("''", "'")
    if v.startswith('"'):  # string, only the \" and \\ escapes (all a flat config needs)
        return v[1:-1].replace('\\"', '"').replace('\\\\', '\\')
    try:  # try int, then float, else raw string
        return int(v)
    except ValueError:
        try:
            return float(v)
        except ValueError:
            if '[' in v or '{' in v:  # likely a list or dict
                raise ValueError(f"Unsupported complex value in flat YAML: {v}")
            return v  # Assume string


def _flat_yaml_text_of(value) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return 'true' if value else 'false'
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, str):
        return "'" + value.replace("'", "''") + "'"  # Single quotes: no escapes, a Windows path stays as is
    raise ValueError(f"Unsupported value type in flat YAML: {type(value)}")


def set_flat_yaml_value(cfg_path, key: str, value) -> None:
    """
    Set the value of key in a flat YAML file, every other line (and the comment of that one) as it was.
    The key is added at the end if missing. The file is replaced atomically (a temporary file next to it).
    """
    cfg_path = Path(cfg_path)
    lines = cfg_path.read_text(encoding='utf-8').splitlines(keepends=True)
    new_line = f'{key}: {_flat_yaml_text_of(value)}'
    for i, line in enumerate(lines):
        line_key, colon, rest = line.partition(':')
        if colon and line_key.strip() == key and not line.lstrip().startswith('#'):
            comment = _split_value_and_comment(rest.rstrip('\r\n'))[1]
            lines[i] = new_line + comment + line[len(line.rstrip('\r\n')):]
            break
    else:
        if lines and not lines[-1].endswith('\n'):
            lines[-1] += '\n'
        lines.append(new_line + '\n')
    tmp_path = cfg_path.with_name(f'.tmp-{os.getpid()}-{cfg_path.name}')
    try:
        tmp_path.write_text(''.join(lines), encoding='utf-8')
        shutil.copymode(cfg_path, tmp_path)
        os.replace(tmp_path, cfg_path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise