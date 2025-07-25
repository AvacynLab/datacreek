import types
from datacreek.core.fractal import bootstrap_db, bootstrap_sigma_db
from datacreek.core.knowledge_graph import KnowledgeGraph

class DummySession:
    def __init__(self, log):
        self.log = log
    def run(self, query, **params):
        self.log.append(query)
        # simulate gds.sample response
        return types.SimpleNamespace(single=lambda: {"nodeIds": ["a", "b"]})
    def __enter__(self):
        return self
    def __exit__(self, exc_type, exc, tb):
        pass

class DummyDriver:
    def __init__(self):
        self.log = []
    def session(self):
        return DummySession(self.log)


def test_bootstrap_db_networkx(monkeypatch):
    kg = KnowledgeGraph()
    kg.graph.add_edge("a", "b")
    monkeypatch.setattr('datacreek.utils.config.load_config', lambda: {"fractal": {"bootstrap_seed": 2}})
    metrics = []
    monkeypatch.setattr('datacreek.core.fractal.update_metric', lambda n, v: metrics.append((n, v)))
    dims = bootstrap_db(kg, n=2, ratio=1.0)
    assert len(dims) == 2
    assert "fractal_seed" in kg.graph.graph
    assert "fractal_sigma" in kg.graph.graph
    assert metrics and metrics[-1][0] == "sigma_db"


def test_bootstrap_db_driver(monkeypatch):
    kg = KnowledgeGraph()
    kg.graph.add_edge("a", "b")
    monkeypatch.setattr(kg, "to_neo4j", lambda *a, **k: None)
    monkeypatch.setattr('datacreek.core.fractal.colour_box_dimension', lambda g, r: (1.0, None))
    driver = DummyDriver()
    bootstrap_db(kg, n=1, ratio=0.5, driver=driver, dataset="tmp")
    assert any("gds.beta.graph.sample" in q for q in driver.log)
    assert kg.graph.graph["fractal_dim"] == 1.0


def test_bootstrap_sigma_db(monkeypatch):
    kg = KnowledgeGraph()
    kg.graph.add_edges_from([(0,1),(1,2)])
    metrics = []
    monkeypatch.setattr('datacreek.core.fractal.update_metric', lambda n, v: metrics.append((n, v)))
    sigma = bootstrap_sigma_db(kg, [1])
    assert sigma == kg.graph.graph["fractal_sigma"]
    assert metrics and metrics[-1][0] == "sigma_db"


def test_bootstrap_sigma_driver(monkeypatch):
    kg = KnowledgeGraph()
    kg.graph.add_edge("a", "b")
    monkeypatch.setattr(kg, "to_neo4j", lambda *a, **k: None)
    driver = DummyDriver()
    monkeypatch.setattr('datacreek.core.fractal.colour_box_dimension', lambda g, r: (1.0, None))
    sigma = bootstrap_sigma_db(kg, [1], driver=driver, dataset="tmp")
    assert sigma >= 0.0
    assert any("gds.beta.graph.sample" in q for q in driver.log)
