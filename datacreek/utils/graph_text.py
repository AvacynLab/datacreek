from __future__ import annotations

"""Utilities for converting graph neighborhoods to text."""

from typing import Iterable, List
import networkx as nx


def neighborhood_to_sentence(graph: nx.Graph, path: Iterable) -> str:
    """Return a simple textual description for a path in ``graph``.

    Each node should have a ``"text"`` attribute describing its content.
    Edges may carry a ``"relation"`` attribute used when joining nodes.
    Nodes without text fallback to their ID.
    """

    nodes = list(path)
    if not nodes:
        return ""

    parts: List[str] = []
    for u, v in zip(nodes[:-1], nodes[1:]):
        u_text = str(graph.nodes[u].get("text", u))
        rel = graph[u][v].get("relation", "->") if graph.has_edge(u, v) else "->"
        parts.append(u_text)
        parts.append(rel)
    last_text = str(graph.nodes[nodes[-1]].get("text", nodes[-1]))
    parts.append(last_text)
    sent = " ".join(parts)
    if not sent.endswith("."):
        sent += "."
    return sent


def subgraph_to_text(graph: nx.Graph, nodes: Iterable) -> str:
    """Return a short textual summary of a subgraph.

    Parameters
    ----------
    graph:
        Graph containing the subgraph to describe.
    nodes:
        Iterable of node identifiers making up the subgraph.

    Returns
    -------
    str
        Human readable summary listing edges with their relations.
    """

    sub = graph.subgraph(nodes)
    lines: List[str] = []
    for u, v, data in sub.edges(data=True):
        u_text = str(graph.nodes[u].get("text", u))
        v_text = str(graph.nodes[v].get("text", v))
        rel = data.get("relation", "->")
        lines.append(f"{u_text} {rel} {v_text}")
    summary = ". ".join(lines)
    if summary and not summary.endswith("."):
        summary += "."
    return summary


def graph_to_text(graph: nx.Graph) -> str:
    """Return a summary listing every edge relation in ``graph``."""

    lines: List[str] = []
    for u, v, data in graph.edges(data=True):
        u_text = str(graph.nodes[u].get("text", u))
        v_text = str(graph.nodes[v].get("text", v))
        rel = data.get("relation", "->")
        lines.append(f"{u_text} {rel} {v_text}")
    summary = ". ".join(lines)
    if summary and not summary.endswith("."):
        summary += "."
    return summary
