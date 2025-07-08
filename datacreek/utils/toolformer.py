from __future__ import annotations

"""Lightweight helpers to insert API call placeholders."""

import re
from typing import Iterable, Tuple


def insert_tool_calls(text: str, tools: Iterable[Tuple[str, str]]) -> str:
    """Return ``text`` with simple tool call placeholders inserted.

    Parameters
    ----------
    text:
        Input text where tool calls should be added.
    tools:
        Iterable of ``(name, pattern)`` pairs. For each pair, the first
        occurrence matching ``pattern`` triggers insertion of
        ``"[TOOL:name(args)]"`` at the match location, where ``args`` is the
        matched substring.
    """

    out = text
    for name, pattern in tools:
        try:
            regex = re.compile(pattern, re.IGNORECASE)
        except re.error:
            # skip invalid patterns
            continue

        def repl(match: re.Match) -> str:
            return f"[TOOL:{name}({match.group(0)})]"

        out = regex.sub(repl, out, count=1)
    return out
