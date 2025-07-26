from dataclasses import dataclass

import datacreek.core.cleanup as cleanup


class FakeKG:
    def __init__(self):
        self.calls = []

    def deduplicate_chunks(self, sim):
        self.calls.append(("dedup", sim))
        return 2

    def clean_chunk_texts(self):
        self.calls.append(("clean",))
        return 3

    def normalize_date_fields(self):
        self.calls.append(("norm",))

    def resolve_entities(self, threshold, aliases=None):
        self.calls.append(("resolve", threshold, aliases))

    def mark_conflicting_facts(self):
        self.calls.append(("conflicts",))

    def validate_coherence(self):
        self.calls.append(("validate",))


def test_cleanup_with_dataset_builder():
    class FakeBuilder:
        def __init__(self, kg):
            self.kg = kg
            self.kwargs = None

        def cleanup_graph(self, **kwargs):
            self.kwargs = kwargs
            return 5, 6

    kg = FakeKG()
    builder = FakeBuilder(kg)
    stats = cleanup.cleanup_knowledge_graph(
        kg,
        dataset_builder=builder,
        resolve_threshold=0.9,
        resolve_aliases={"foo": ["bar"]},
        dedup_similarity=0.7,
        normalize_dates=False,
        mark_conflicts=True,
        validate=True,
    )

    assert stats.removed == 5 and stats.cleaned == 6
    assert builder.kwargs == {
        "resolve_threshold": 0.9,
        "resolve_aliases": {"foo": ["bar"]},
        "dedup_similarity": 0.7,
        "normalize_dates": False,
        "mark_conflicts": True,
        "validate": True,
    }
    # no KG methods should be called when builder provided
    assert kg.calls == []


def test_cleanup_without_builder():
    kg = FakeKG()
    stats = cleanup.cleanup_knowledge_graph(
        kg,
        resolve_threshold=0.95,
        resolve_aliases={"a": ["b"]},
        dedup_similarity=0.8,
        normalize_dates=True,
        mark_conflicts=True,
        validate=True,
    )

    assert stats.removed == 2
    assert stats.cleaned == 3
    assert (
        kg.calls
        == [
            ("dedup", 0.8),
            ("clean",),
            ("norm",),
            ("resolve", 0.95, {"a": ["b"]}),
            ("conflicts",),
            ("validate",),
        ]
    )
