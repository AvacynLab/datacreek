from pathlib import Path

from datacreek.smart_patch import PATCHSET_REGISTRY, PatchCandidate
from scripts.sheaf_repair import generate_patch_file


def test_generate_patch_file_creates_cypher_with_optimal_edges(tmp_path: Path) -> None:
    """ILP solver writes a Cypher file containing the best patch set."""

    candidates = [
        (PatchCandidate("e1", benefit=5, risk=4), "// patch e1"),
        (PatchCandidate("e2", benefit=4, risk=3), "// patch e2"),
        (PatchCandidate("e3", benefit=3, risk=2), "// patch e3"),
    ]
    path = generate_patch_file(candidates, budget=5, out_dir=tmp_path)

    patchset_id = path.stem
    assert patchset_id in PATCHSET_REGISTRY
    selected = PATCHSET_REGISTRY[patchset_id]
    assert set(selected) == {"e2", "e3"}

    lines = path.read_text().splitlines()
    assert lines == ["// patch e2", "// patch e3"]
