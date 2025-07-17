import datacreek.core.knowledge_graph as kgmod
from datacreek.core.knowledge_graph import KnowledgeGraph


def test_langid_prevents_cross_language_merge(monkeypatch):
    monkeypatch.setattr('datacreek.utils.text.detect_language',
                        lambda t: 'en' if 'Hello' in t else 'fr')
    kg = KnowledgeGraph()
    kg.add_entity('e1', 'Hello world')
    kg.add_entity('e2', 'Bonjour le monde')
    merged = kg.resolve_entities(threshold=0.0)
    assert merged == 0


def test_langid_allows_same_language(monkeypatch):
    monkeypatch.setattr('datacreek.utils.text.detect_language', lambda t: 'en')
    kg = KnowledgeGraph()
    kg.add_entity('e1', 'Hello world')
    kg.add_entity('e2', 'Hi world')
    merged = kg.resolve_entities(threshold=0.0)
    assert merged == 1
