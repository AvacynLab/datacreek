import types
import numpy as np
import pytest

from datacreek.generators.kg_generator import KGGenerator


class DummyClient:
    def __init__(self):
        self.calls = []
        self.config = {
            "generation": {"temperature": 0.0, "max_tokens": 5},
            "prompts": {
                "kg_question": "Q: {facts}",
                "kg_answer": "A: {question} {facts}",
            },
        }

    def chat_completion(self, messages, temperature=None, max_tokens=None):
        self.calls.append(messages[0]["content"])
        return "resp"


class SimpleGraph:
    class NodeAccessor:
        def __init__(self, graph):
            self.g = graph

        def __getitem__(self, node):
            return self.g._nodes[node]

        def __call__(self, data=False):
            if data:
                return [(n, self.g._nodes[n]) for n in self.g._nodes]
            return list(self.g._nodes.keys())

    def __init__(self):
        self._nodes = {}
        self._edges = {}
        self.nodes = SimpleGraph.NodeAccessor(self)

    def add_node(self, node, **data):
        self._nodes[node] = data
        self._edges.setdefault(node, set())

    def add_edge(self, a, b):
        self._edges.setdefault(a, set()).add(b)
        self._edges.setdefault(b, set()).add(a)

    def degree(self, node):
        return len(self._edges.get(node, []))


def _make_kg(num_facts=2, embeddings=None, with_text=False):
    kg = types.SimpleNamespace()
    kg.graph = SimpleGraph()
    kg.index = types.SimpleNamespace(
        transform=lambda texts: np.array(embeddings if embeddings is not None else []))
    chunk_map = {}
    kg.fact_confidence = lambda s, p, o: 0.5
    for i in range(num_facts):
        kg.graph.add_node(
            f"f{i}", type="fact", subject=f"s{i}", predicate="is", object=f"o{i}"
        )
        for j in range(i + 1):
            cid = f"c{i}_{j}"
            data = {"text": f"ctx{i}{j}"} if with_text else {}
            kg.graph.add_node(cid, **data)
            kg.graph.add_edge(f"f{i}", cid)
            chunk_map.setdefault(f"f{i}", []).append(cid)

    kg.get_chunks_for_fact = lambda fid: chunk_map.get(fid, [])
    return kg


def test_select_facts_no_embeddings():
    kg = _make_kg(3)
    gen = KGGenerator(DummyClient())
    selected = gen._select_facts(kg, 2)
    assert selected == ["f2", "f1"]


def test_select_facts_kmeans(monkeypatch):
    kg = _make_kg(3, embeddings=[[0], [1], [2]])

    class DummyKMeans:
        called = False

        def __init__(self, *a, **k):
            pass

        def fit_predict(self, X):
            DummyKMeans.called = True
            return [0, 1, 0]

    monkeypatch.setattr("datacreek.generators.kg_generator.KMeans", DummyKMeans)
    gen = KGGenerator(DummyClient())
    selected = gen._select_facts(kg, 2)
    assert DummyKMeans.called
    assert len(selected) == 2


def test_select_facts_import_error(monkeypatch):
    # more facts than requested ensures the clustering path is taken
    kg = _make_kg(3, embeddings=[[1], [2], [3]])
    monkeypatch.setattr("datacreek.generators.kg_generator.KMeans", None)
    gen = KGGenerator(DummyClient())
    with pytest.raises(ImportError):
        gen._select_facts(kg, 2)


def test_process_graph_basic(monkeypatch):
    embeddings = [[0], [1]]
    kg = _make_kg(2, embeddings=embeddings)
    monkeypatch.setattr(
        "datacreek.generators.kg_generator.KMeans",
        lambda *a, **k: types.SimpleNamespace(fit_predict=lambda X: [0, 1]),
    )
    gen = KGGenerator(DummyClient())
    result = gen.process_graph(kg, num_pairs=2)
    assert len(result["qa_pairs"]) == 2
    assert gen.client.calls and "Q:" in gen.client.calls[0]


def test_process_graph_multi_answer(monkeypatch):
    kg = _make_kg(1, embeddings=[], with_text=True)
    gen = KGGenerator(DummyClient())
    res = gen.process_graph(kg, num_pairs=2, multi_answer=True)
    assert len(res["qa_pairs"]) == 2
    assert "ctx0" in gen.client.calls[0]


def test_process_graph_empty():
    gen = KGGenerator(DummyClient())
    kg = _make_kg(0)
    res = gen.process_graph(kg, num_pairs=1)
    assert res == {"qa_pairs": []}
