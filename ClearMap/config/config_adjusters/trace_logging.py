from __future__ import annotations

from typing import Any, Optional, Callable


def _summarize_node(node: Any) -> str:
    if node is None:
        return "None"
    t = type(node).__name__
    if isinstance(node, dict):
        keys = list(node.keys())
        return f"{t}(keys={len(keys)} sample={keys[:8]})"
    return t


def _get_path(d: Any, dotted: str) -> Any:
    cur = d
    for part in dotted.split('.'):
        if cur is None:
            return None
        if isinstance(cur, dict):
            cur = cur.get(part)
        else:
            # if cur is a sentinel wrapper (e.g. _REPLACE), try to unwrap common fields
            # adjust to your actual _REPLACE implementation if needed
            payload = getattr(cur, "value", None) or getattr(cur, "payload", None)
            if isinstance(payload, dict):
                cur = payload.get(part)
            else:
                return None
    return cur


class Tracer:
    """Small helper to keep tracing calls compact and non-invasive."""
    __slots__ = ('enabled', 'sink')

    def __init__(self, enabled: bool, sink: Optional[Callable[[str], None]] = print) -> None:
        self.enabled = bool(enabled)
        self.sink = sink or (lambda _: None)

    def __call__(self, msg: str) -> None:
        if self.enabled:
            self.sink(msg)


def summarize_paths(obj, trace_paths: tuple[str, ...] | tuple[()]) -> str:
    nodes = {p: _summarize_node(_get_path(obj, p)) for p in trace_paths}
    formatted_nodes = " | ".join(f"{p}={nodes[p]}" for p in trace_paths)
    return formatted_nodes
