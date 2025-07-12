from datacreek.core import fractal as fractal_mod
from datacreek.core.knowledge_graph import KnowledgeGraph

bootstrap_db = fractal_mod.bootstrap_db


def test_bootstrap_db_records_properties():
    kg = KnowledgeGraph()
    kg.graph.add_edge("a", "b")
    dims = bootstrap_db(kg, n=3, ratio=0.8)
    assert len(dims) == 3
    assert "fractal_dim" in kg.graph.graph
    assert "fractal_sigma" in kg.graph.graph
