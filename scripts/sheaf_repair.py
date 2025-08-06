#!/usr/bin/env python3
"""Solve smart-patch ILP and write Cypher patch suggestions.

This utility wraps :func:`datacreek.smart_patch.solve_patch_ilp` to produce a
Cypher patch set for edges whose repair maximises the expected sheaf/hypergraph
consistency gain under a risk budget ``B``.  Given candidate edges with their
benefit :math:`c_e` and risk :math:`w_e`, the optimisation problem is

.. math::
    \max_x \sum_e c_e x_e \quad\text{s.t.}\quad \sum_e w_e x_e \le B,\;x_e \in\{0,1\}

Selected edges are stored under a unique patch set identifier and the
corresponding Cypher statements are written to ``repair_suggestions/<id>.cypher``.
The resulting file can later be reviewed through the ``/ui/edge_review`` API.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Sequence, Tuple

from datacreek.smart_patch import PATCHSET_REGISTRY, PatchCandidate, solve_patch_ilp

CandidateWithCypher = Tuple[PatchCandidate, str]


def load_candidates(path: Path) -> list[CandidateWithCypher]:
    """Load patch candidates from ``path``.

    The JSON file must contain a list of objects with ``edge_id``, ``benefit``,
    ``risk`` and ``cypher`` fields.
    """

    data = json.loads(path.read_text())
    return [
        (PatchCandidate(d["edge_id"], d["benefit"], d["risk"]), d["cypher"])
        for d in data
    ]


def generate_patch_file(
    candidates: Sequence[CandidateWithCypher],
    budget: float,
    out_dir: Path | None = None,
) -> Path:
    """Solve the ILP and persist the selected Cypher statements.

    Parameters
    ----------
    candidates:
        Sequence of ``(PatchCandidate, cypher)`` pairs.
    budget:
        Risk budget :math:`B` used in the optimisation constraint.
    out_dir:
        Directory where the patch file is written. Defaults to
        ``repair_suggestions`` relative to the current working directory.

    Returns
    -------
    Path
        Path to the generated ``.cypher`` file. The stem of the path is the
        patch set identifier registered in :data:`PATCHSET_REGISTRY`.
    """

    patchset_id, selected = solve_patch_ilp([pc for pc, _ in candidates], budget)
    out_dir = out_dir or Path("repair_suggestions")
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{patchset_id}.cypher"
    with path.open("w", encoding="utf8") as fh:
        for pc, cypher in candidates:
            if pc.edge_id in selected:
                fh.write(cypher.strip() + "\n")
    return path


def main(argv: Iterable[str] | None = None) -> None:
    """CLI entry point.

    Examples
    --------
    ``scripts/sheaf_repair.py candidates.json --budget 3``
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="JSON file with patch candidates")
    parser.add_argument("--budget", type=float, required=True, help="risk budget B")
    args = parser.parse_args(list(argv) if argv is not None else None)

    candidates = load_candidates(args.input)
    path = generate_patch_file(candidates, budget=args.budget)
    print(path)


if __name__ == "__main__":  # pragma: no cover - CLI
    main()
