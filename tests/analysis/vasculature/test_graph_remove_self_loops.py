"""
``Graph.remove_self_loops()`` calls graph-tool's function wherever the installed graph-tool has it.

The function is documented in ``graph_tool.generation`` in the current graph-tool; ClearMap called it as
``graph_tool.stats.remove_self_loops``, and ClearMap3.yml pinned ``graph-tool<=2.45`` (up to ClearMap 3.1) because
later releases could not remove the self loops. The graph-tool modules are replaced by fakes here, so that both
locations are tested whatever graph-tool is installed: the graph itself is tested by
``test_graph_corrections.py::test_remove_auto_loops``.
"""
from types import SimpleNamespace

from ClearMap.Analysis.graphs import graph_gt

BASE_GRAPH = object()


def _record_calls_in(calls, location):
    def remove_self_loops(graph):
        calls.append((location, graph))
    return remove_self_loops


def _graph():
    return SimpleNamespace(base=BASE_GRAPH)


def test_the_self_loops_are_removed_with_the_function_of_graph_tool_generation(monkeypatch):
    calls = []
    monkeypatch.setattr(graph_gt, 'gtg', SimpleNamespace(remove_self_loops=_record_calls_in(calls, 'generation')))
    monkeypatch.setattr(graph_gt, 'gt', SimpleNamespace(stats=SimpleNamespace()))

    graph_gt.Graph.remove_self_loops(_graph())

    assert calls == [('generation', BASE_GRAPH)]


def test_the_self_loops_are_removed_with_the_function_of_graph_tool_stats_when_generation_has_none(monkeypatch):
    calls = []
    monkeypatch.setattr(graph_gt, 'gtg', SimpleNamespace())
    monkeypatch.setattr(graph_gt, 'gt',
                        SimpleNamespace(stats=SimpleNamespace(remove_self_loops=_record_calls_in(calls, 'stats'))))

    graph_gt.Graph.remove_self_loops(_graph())

    assert calls == [('stats', BASE_GRAPH)]
