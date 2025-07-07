import asyncio
import json
from pathlib import Path

import fakeredis
import networkx as nx
import pytest
import requests

from datacreek.analysis import bottleneck_distance
from datacreek.core import dataset
from datacreek.core.dataset import DatasetBuilder
from datacreek.core.ingest import IngestOptions
from datacreek.models.stage import DatasetStage
from datacreek.pipelines import DatasetType, PipelineStep


def test_dataset_has_its_own_graph():
    ds1 = DatasetBuilder(DatasetType.QA, name="ds1")
    ds2 = DatasetBuilder(DatasetType.QA, name="ds2")
    ds1.add_document("doc1", source="a")
    ds1.add_chunk("doc1", "c1", "hello")

    assert ds1.search("hello") == ["c1"]
    # second dataset should be empty
    assert ds2.search("hello") == []


def test_dataset_name_validation():
    with pytest.raises(ValueError):
        DatasetBuilder(DatasetType.QA, name="bad name")
    long_name = "a" * 65
    with pytest.raises(ValueError):
        DatasetBuilder(DatasetType.QA, name=long_name)


def test_dataset_search_wrappers():
    ds = DatasetBuilder(DatasetType.QA)
    ds.add_document("d", source="s")
    ds.add_chunk("d", "c", "text")
    assert ds.search_chunks("text") == ["c"]
    assert ds.search_documents("d") == ["d"]
    assert ds.get_chunks_for_document("d") == ["c"]
    ds.graph.index.build()
    assert ds.search_embeddings("text", k=1) == ["c"]
    assert ds.search_hybrid("text", k=1) == ["c"]

    ds.add_entity("e", "text")
    ds.graph.index.build()
    assert ds.search_hybrid("text", k=1, node_type="entity") == ["e"]

    fid = ds.graph.add_fact("A", "related", "B")
    ds.graph.index.build()
    assert ds.search_facts("related") == [fid]
    assert ds.search_hybrid("related", k=1, node_type="fact") == [fid]


def test_dataset_clone():
    ds = DatasetBuilder(DatasetType.QA, name="orig")
    ds.add_document("d", source="s")
    ds.add_chunk("d", "c1", "text")
    clone = ds.clone(name="copy")
    clone.add_chunk("d", "c2", "more")

    # original should not have the new chunk
    assert ds.search("more") == []
    assert clone.search("more") == ["c2"]


def test_dataset_persistence_redis():
    ds = DatasetBuilder(DatasetType.TEXT, name="test")
    ds.add_document("d", source="source.txt")
    ds.add_chunk("d", "c1", "hello world")

    client = fakeredis.FakeStrictRedis()
    ds.to_redis(client, "ds:test")
    loaded = DatasetBuilder.from_redis(client, "ds:test")

    assert loaded.name == "test"
    assert loaded.search_chunks("hello") == ["c1"]
    # id should round-trip through persistence
    assert loaded.id == ds.id


def test_auto_persist(monkeypatch):
    client = fakeredis.FakeStrictRedis()
    ds = DatasetBuilder(DatasetType.QA, name="auto")
    ds.redis_client = client
    ds.add_document("d", source="s")
    ds.add_chunk("d", "c1", "hi")

    stored = client.get("dataset:auto")
    assert stored is not None
    loaded = DatasetBuilder.from_redis(client, "dataset:auto")
    assert loaded.search("hi") == ["c1"]


def test_extract_facts_auto_persist(monkeypatch):
    client = fakeredis.FakeStrictRedis()
    ds = DatasetBuilder(DatasetType.TEXT, name="facts")
    ds.redis_client = client
    ds.add_document("d", source="s")
    ds.add_chunk("d", "c1", "Paris is the capital of France.")
    ds.extract_facts()

    stored = client.get("dataset:facts")
    assert stored is not None
    loaded = DatasetBuilder.from_redis(client, "dataset:facts")
    fact_nodes = [n for n, d in loaded.graph.graph.nodes(data=True) if d.get("type") == "fact"]
    assert fact_nodes


def test_auto_persist_neo4j(monkeypatch):
    ds = DatasetBuilder(DatasetType.QA, name="neo")
    called = {}
    ds.neo4j_driver = object()
    monkeypatch.setattr(
        ds.graph,
        "to_neo4j",
        lambda driver, clear=True, dataset=None: called.setdefault("driver", driver),
    )
    ds.add_document("d", source="s")
    assert called["driver"] is ds.neo4j_driver


def test_save_and_load_neo4j_methods(monkeypatch):
    ds = DatasetBuilder(DatasetType.TEXT, name="neo2")
    fake_driver = object()
    called = {}

    monkeypatch.setattr(
        ds.graph.__class__,
        "to_neo4j",
        lambda self, driver, clear=True, dataset=None: called.setdefault("save", driver),
    )
    monkeypatch.setattr(
        ds.graph.__class__,
        "from_neo4j",
        staticmethod(
            lambda driver, dataset=None: called.setdefault("load", driver) or ds.graph.__class__()
        ),
    )

    ds.save_neo4j(fake_driver)
    assert called.get("save") is fake_driver
    assert any(e.operation == "save_neo4j" for e in ds.events)

    ds.load_neo4j(fake_driver)
    assert called.get("load") is fake_driver
    assert any(e.operation == "load_neo4j" for e in ds.events)


def test_events_persisted_roundtrip():
    client = fakeredis.FakeStrictRedis()
    ds = DatasetBuilder(DatasetType.TEXT, name="evt")
    ds.redis_client = client
    ds.add_document("d", source="s")
    ds.add_chunk("d", "c1", "hi")

    loaded = DatasetBuilder.from_redis(client, "dataset:evt")
    assert any(e.operation == "add_chunk" for e in loaded.events)
    assert client.llen("dataset:evt:events") == len(loaded.events)


def test_events_logged_globally():
    client = fakeredis.FakeStrictRedis()
    ds = DatasetBuilder(DatasetType.TEXT, name="glob")
    ds.redis_client = client
    ds.add_document("d", source="s")
    ds.add_chunk("d", "c1", "hi")

    events = [json.loads(e) for e in client.lrange("dataset:events", 0, -1)]
    assert any(e["dataset"] == "glob" and e["operation"] == "add_chunk" for e in events)


def test_ingest_file_method(tmp_path):
    f = tmp_path / "doc.txt"
    f.write_text("hello world")
    client = fakeredis.FakeStrictRedis()
    ds = DatasetBuilder(DatasetType.TEXT, name="demo")
    ds.redis_client = client

    ds.ingest_file(str(f), options=IngestOptions())

    assert ds.stage == DatasetStage.INGESTED
    assert ds.events[-1].operation == "ingest_document"
    assert client.get("dataset:demo") is not None
    assert list(ds.ingested_docs) == ["doc"]
    info = ds.ingested_docs["doc"]
    assert info["path"] == str(f)
    loaded = DatasetBuilder.from_redis(client, "dataset:demo")
    assert loaded.ingested_docs == ds.ingested_docs


def test_ingest_file_async(tmp_path):
    f = tmp_path / "doc.txt"
    f.write_text("hello async")
    client = fakeredis.FakeStrictRedis()
    ds = DatasetBuilder(DatasetType.TEXT, name="demo")
    ds.redis_client = client

    asyncio.run(ds.ingest_file_async(str(f), options=IngestOptions()))

    assert ds.stage == DatasetStage.INGESTED
    assert list(ds.ingested_docs) == ["doc"]
    loaded = DatasetBuilder.from_redis(client, "dataset:demo")
    assert loaded.ingested_docs == ds.ingested_docs


def test_persist_after_decorator(monkeypatch):
    client = fakeredis.FakeStrictRedis()
    ds = DatasetBuilder(DatasetType.TEXT, name="decor")
    ds.redis_client = client

    def dummy_ingest(path, dataset, **kwargs):
        dataset.add_document("d1", source=path)
        return "d1"

    monkeypatch.setattr("datacreek.core.ingest.ingest_into_dataset", dummy_ingest)
    monkeypatch.setattr(ds, "_record_event", lambda *a, **k: None)

    ds.ingest_file("file.txt")

    assert client.get("dataset:decor") is not None


def test_remove_chunk_updates_order():
    ds = DatasetBuilder(DatasetType.TEXT)
    ds.add_document("d", source="s")
    ds.add_chunk("d", "c1", "t1")
    ds.add_chunk("d", "c2", "t2")
    ds.add_chunk("d", "c3", "t3")

    ds.remove_chunk("c2")
    assert ds.get_chunks_for_document("d") == ["c1", "c3"]
    assert ("c1", "c3") in ds.graph.graph.edges
    assert ds.graph.graph.edges["d", "c1"]["sequence"] == 0
    assert ds.graph.graph.edges["d", "c3"]["sequence"] == 1


def test_remove_document_cascades():
    ds = DatasetBuilder(DatasetType.TEXT)
    ds.add_document("d", source="s")
    ds.add_chunk("d", "c1", "t1")
    ds.add_chunk("d", "c2", "t2")

    ds.remove_document("d")
    assert ds.search("t1") == []
    assert ds.search_documents("d") == []


def test_dataset_id_in_serialization():
    ds = DatasetBuilder(DatasetType.QA, name="demo")
    data = ds.to_dict()
    assert "id" in data
    loaded = DatasetBuilder.from_dict(data)
    assert loaded.id == ds.id


def test_link_similar_chunks_wrapper():
    ds = DatasetBuilder(DatasetType.TEXT)
    ds.add_document("d", source="s")
    ds.add_chunk("d", "c1", "hello world")
    ds.add_chunk("d", "c2", "hello planet")
    ds.add_chunk("d", "c3", "other text")

    ds.graph.index.build()
    ds.link_similar_chunks(k=1)

    assert ("c1", "c2") in ds.graph.graph.edges
    edge = ds.graph.graph.edges["c1", "c2"]
    assert edge["relation"] == "similar_to"
    assert 0 <= edge["similarity"] <= 1


def test_link_similar_sections_wrapper():
    ds = DatasetBuilder(DatasetType.TEXT)
    ds.add_document("d", source="s")
    ds.add_section("d", "s1", title="Intro")
    ds.add_section("d", "s2", title="Introduction")
    ds.add_section("d", "s3", title="Other")

    ds.graph.index.build()
    ds.link_similar_sections(k=1)

    assert ("s1", "s2") in ds.graph.graph.edges
    edge = ds.graph.graph.edges["s1", "s2"]
    assert edge["relation"] == "similar_to"
    assert 0 <= edge["similarity"] <= 1


def test_get_similar_chunks_wrapper():
    ds = DatasetBuilder(DatasetType.TEXT)
    ds.add_document("d", source="s")
    ds.add_chunk("d", "c1", "hello world")
    ds.add_chunk("d", "c2", "hello planet")
    ds.add_chunk("d", "c3", "other text")

    ds.graph.index.build()
    sims = ds.get_similar_chunks("c1", k=2)
    assert "c1" not in sims
    assert "c2" in sims


def test_get_similar_chunks_wrapper_unknown():
    ds = DatasetBuilder(DatasetType.TEXT)
    ds.add_document("d", source="s")
    ds.graph.index.build()
    assert ds.get_similar_chunks("missing") == []


def test_get_similar_chunks_data_wrapper():
    ds = DatasetBuilder(DatasetType.TEXT)
    ds.add_document("d", source="s")
    ds.add_chunk("d", "c1", "hello world")
    ds.add_chunk("d", "c2", "hello planet")
    ds.add_chunk("d", "c3", "other text")
    ds.graph.index.build()

    data = ds.get_similar_chunks_data("c1", k=2)
    ids = [d["id"] for d in data]
    assert "c1" not in ids
    assert "c2" in ids


def test_get_chunk_neighbors_wrapper():
    ds = DatasetBuilder(DatasetType.TEXT)
    ds.add_document("d", source="s")
    ds.add_chunk("d", "c1", "hello world")
    ds.add_chunk("d", "c2", "hello planet")
    ds.add_chunk("d", "c3", "other text")
    ds.graph.index.build()

    neighbors = ds.get_chunk_neighbors(k=1)
    assert "c1" in neighbors
    assert "c2" in neighbors["c1"]


def test_get_chunk_neighbors_data_wrapper():
    ds = DatasetBuilder(DatasetType.TEXT)
    ds.add_document("d", source="s")
    ds.add_chunk("d", "c1", "hello world")
    ds.add_chunk("d", "c2", "hello planet")
    ds.add_chunk("d", "c3", "other text")
    ds.graph.index.build()

    data = ds.get_chunk_neighbors_data(k=1)
    assert set(data.keys()) == {"c1", "c2", "c3"}
    assert data["c1"][0]["id"] in {"c2", "c3"}


def test_get_chunk_context_wrapper():
    ds = DatasetBuilder(DatasetType.TEXT)
    ds.add_document("d", source="s")
    ds.add_chunk("d", "c1", "t1")
    ds.add_chunk("d", "c2", "t2")
    ds.add_chunk("d", "c3", "t3")
    ctx = ds.get_chunk_context("c2", before=1, after=1)
    assert ctx == ["c1", "c2", "c3"]


def test_get_chunk_context_wrapper_unknown():
    ds = DatasetBuilder(DatasetType.TEXT)
    assert ds.get_chunk_context("missing") == []


def test_get_similar_sections_wrapper():
    ds = DatasetBuilder(DatasetType.TEXT)
    ds.add_document("d", source="s")
    ds.add_section("d", "s1", title="Intro")
    ds.add_section("d", "s2", title="Introduction")
    ds.add_section("d", "s3", title="Other")

    ds.graph.index.build()
    sims = ds.get_similar_sections("s1", k=2)
    assert "s1" not in sims
    assert "s2" in sims


def test_get_similar_sections_wrapper_unknown():
    ds = DatasetBuilder(DatasetType.TEXT)
    assert ds.get_similar_sections("missing") == []


def test_link_similar_documents_wrapper():
    ds = DatasetBuilder(DatasetType.TEXT)
    ds.add_document("d1", source="s", text="hello world")
    ds.add_document("d2", source="s", text="hello planet")
    ds.add_document("d3", source="s", text="other")

    ds.graph.index.build()
    ds.link_similar_documents(k=1)

    assert ("d1", "d2") in ds.graph.graph.edges
    edge = ds.graph.graph.edges["d1", "d2"]
    assert edge["relation"] == "similar_to"
    assert 0 <= edge["similarity"] <= 1


def test_get_similar_documents_wrapper():
    ds = DatasetBuilder(DatasetType.TEXT)
    ds.add_document("d1", source="s", text="hello world")
    ds.add_document("d2", source="s", text="hello planet")
    ds.add_document("d3", source="s", text="other")

    ds.graph.index.build()
    sims = ds.get_similar_documents("d1", k=2)
    assert "d1" not in sims
    assert "d2" in sims


def test_get_similar_documents_wrapper_unknown():
    ds = DatasetBuilder(DatasetType.TEXT)
    assert ds.get_similar_documents("missing") == []


def test_page_for_chunk_wrapper():
    ds = DatasetBuilder(DatasetType.TEXT)
    ds.add_document("d", source="s")
    ds.add_chunk("d", "c1", "text", page=2)

    assert ds.get_page_for_chunk("c1") == 2


def test_page_for_section_wrapper():
    ds = DatasetBuilder(DatasetType.TEXT)
    ds.add_document("doc", source="s")
    ds.add_section("doc", "sec1", page=4)
    ds.add_chunk("doc", "c1", "t", section_id="sec1", page=4)

    assert ds.get_page_for_section("sec1") == 4


def test_search_with_links_wrapper():
    ds = DatasetBuilder(DatasetType.TEXT)
    ds.add_document("d", source="s")
    ds.add_chunk("d", "c1", "hello world")
    ds.add_chunk("d", "c2", "hello planet")
    ds.add_chunk("d", "c3", "unrelated text")

    ds.graph.index.build()
    ds.link_similar_chunks(k=1)

    results = ds.search_with_links("hello", k=1, hops=1)
    assert "c1" in results
    assert "c2" in results


def test_search_with_links_data_wrapper():
    ds = DatasetBuilder(DatasetType.TEXT)
    ds.add_document("d", source="s")
    ds.add_chunk("d", "c1", "hello world")
    ds.add_chunk("d", "c2", "hello planet")
    ds.add_chunk("d", "c3", "unrelated text")

    ds.graph.index.build()
    ds.link_similar_chunks(k=1)

    results = ds.search_with_links_data("hello", k=1, hops=1)
    ids = [r["id"] for r in results]
    assert "c1" in ids
    assert "c2" in ids
    for r in results:
        if r["id"] == "c1":
            assert r["text"] == "hello world"
            assert r["document"] == "d"
            assert r["depth"] == 0
            assert r["path"] == ["c1"]
        else:
            assert r["depth"] == 1
            assert r["path"][0] == "c1"
            assert r["path"][1] == r["id"]


def test_section_helpers_wrapper():
    ds = DatasetBuilder(DatasetType.TEXT)
    ds.add_document("doc", source="s")
    ds.add_section("doc", "s1", title="Intro")
    ds.add_section("doc", "s2", title="Body")
    ds.add_chunk("doc", "c1", "t1", section_id="s1")
    ds.add_chunk("doc", "c2", "t2", section_id="s2")

    assert ds.get_sections_for_document("doc") == ["s1", "s2"]
    assert ds.get_chunks_for_section("s1") == ["c1"]
    assert ds.get_section_for_chunk("c2") == "s2"
    assert ds.get_next_chunk("c1") == "c2"
    assert ds.get_previous_chunk("c2") == "c1"
    assert ds.get_next_section("s1") == "s2"
    assert ds.get_previous_section("s2") == "s1"

    assert ds.search_entities("intro") == []
    assert ds.search_sections("intro") == ["s1"]


def test_section_helpers_wrapper_fallback():
    ds = DatasetBuilder(DatasetType.TEXT)
    ds.add_document("doc", source="s")
    ds.add_section("doc", "s1")
    ds.add_section("doc", "s2")
    ds.add_section("doc", "s3")

    for u, v, d in list(ds.graph.graph.edges(data=True)):
        if d.get("relation") == "next_section":
            ds.graph.graph.remove_edge(u, v)

    assert ds.get_next_section("s1") == "s2"
    assert ds.get_previous_section("s3") == "s2"


def test_document_lookup_wrapper():
    ds = DatasetBuilder(DatasetType.TEXT)
    ds.add_document("doc", source="s")
    ds.add_section("doc", "sec1")
    ds.add_chunk("doc", "c1", "text", section_id="sec1")

    assert ds.get_document_for_section("sec1") == "doc"
    assert ds.get_document_for_chunk("c1") == "doc"

    ds.graph.graph.remove_edge("doc", "c1")
    assert ds.get_document_for_chunk("c1") == "doc"


def test_link_entity_source_and_trust():
    ds = DatasetBuilder(DatasetType.TEXT)
    ds.add_document("d", source="src")
    ds.add_chunk("d", "c1", "hello", source="src")
    ds.add_entity("e1", "entity", source="src")
    ds.link_entity("c1", "e1", provenance="src")
    ds.graph.index.build()
    ds.score_trust()

    edge = ds.graph.graph.edges["c1", "e1"]
    assert edge["provenance"] == "src"
    assert "trust" in edge


def test_update_embeddings_wrapper():
    ds = DatasetBuilder(DatasetType.TEXT)
    ds.add_document("d", source="s")
    ds.add_chunk("d", "c1", "hello world")
    ds.graph.index.build()
    ds.update_embeddings()
    emb = ds.graph.graph.nodes["c1"].get("embedding")
    assert isinstance(emb, list)
    assert len(emb) > 0


def test_dedup_and_resolve_wrappers(monkeypatch):
    ds = DatasetBuilder(DatasetType.TEXT)
    ds.add_document("d", source="s")
    ds.add_chunk("d", "c1", "hello")
    ds.add_chunk("d", "c2", "hello")
    ds.add_entity("e1", "Beethoven")
    ds.add_entity("e2", "Ludwig van Beethoven")
    removed = ds.deduplicate_chunks()
    assert removed == 1
    merged = ds.resolve_entities(threshold=0.5)
    assert merged >= 1


def test_resolve_entities_wrapper_with_aliases():
    ds = DatasetBuilder(DatasetType.TEXT)
    ds.add_entity("e1", "IBM")
    ds.add_entity("e2", "International Business Machines")

    merged = ds.resolve_entities(
        threshold=1.0, aliases={"IBM": ["international business machines"]}
    )

    assert merged == 1
    assert "e2" not in ds.graph.graph.nodes


def test_co_mentions_wrapper():
    ds = DatasetBuilder(DatasetType.TEXT)
    ds.add_document("d1", source="s")
    ds.add_document("d2", source="s")
    ds.add_chunk("d1", "c1", "Paris is big")
    ds.add_chunk("d2", "c2", "I love Paris")
    ds.add_entity("Paris", "Paris")
    ds.link_entity("c1", "Paris")
    ds.link_entity("c2", "Paris")

    added = ds.link_chunks_by_entity()
    assert added == 1
    assert ds.graph.graph.has_edge("c1", "c2")
    assert ds.events[-1].operation == "link_chunks_by_entity"
    assert ds.events[-1].params == {"added": 1}


def test_link_documents_by_entity_wrapper():
    ds = DatasetBuilder(DatasetType.TEXT)
    ds.add_document("doc1", source="s")
    ds.add_document("doc2", source="s")
    ds.add_chunk("doc1", "c1", "Paris is big")
    ds.add_chunk("doc2", "c2", "I love Paris")
    ds.add_entity("Paris", "Paris")
    ds.link_entity("c1", "Paris")
    ds.link_entity("c2", "Paris")

    added = ds.link_documents_by_entity()
    assert added == 1
    assert ds.graph.graph.has_edge("doc1", "doc2")
    assert ds.events[-1].operation == "link_documents_by_entity"
    assert ds.events[-1].params == {"added": 1}


def test_link_sections_by_entity_wrapper():
    ds = DatasetBuilder(DatasetType.TEXT)
    ds.add_document("doc", source="s")
    ds.add_section("doc", "s1")
    ds.add_section("doc", "s2")
    ds.add_chunk("doc", "c1", "Paris", section_id="s1")
    ds.add_chunk("doc", "c2", "Paris", section_id="s2")
    ds.add_entity("Paris", "Paris")
    ds.link_entity("c1", "Paris")
    ds.link_entity("c2", "Paris")

    added = ds.link_sections_by_entity()
    assert added == 1
    assert ds.graph.graph.has_edge("s1", "s2")
    assert ds.events[-1].operation == "link_sections_by_entity"
    assert ds.events[-1].params == {"added": 1}


def test_link_authors_organizations_wrapper():
    ds = DatasetBuilder(DatasetType.TEXT)
    ds.add_document("doc", source="s", author="alice", organization="acme")

    added = ds.link_authors_organizations()

    assert added == 1
    assert ds.graph.graph.has_edge("alice", "acme")
    assert ds.events[-1].operation == "link_authors_organizations"
    assert ds.events[-1].params == {"added": 1}


def test_clean_chunks_wrapper(monkeypatch):
    ds = DatasetBuilder(DatasetType.TEXT)
    ds.add_document("d", source="s")
    ds.add_chunk("d", "c1", "<b>Hello</b>\n")
    logs = []
    monkeypatch.setattr(dataset.logger, "info", lambda msg: logs.append(msg))

    changed = ds.clean_chunks()
    assert changed == 1
    assert ds.graph.graph.nodes["c1"]["text"] == "Hello"
    assert logs[-1] == "Cleaned 1 chunks"
    assert ds.history[-1] == "Cleaned 1 chunks"
    assert ds.events[-1].operation == "clean_chunks"


def test_cleanup_logging(monkeypatch):
    ds = DatasetBuilder(DatasetType.TEXT)
    ds.add_document("d", source="s")
    ds.add_chunk("d", "c1", "text")
    ds.add_chunk("d", "c2", "text")

    logs = []
    monkeypatch.setattr(dataset.logger, "info", lambda msg: logs.append(msg))

    removed = ds.deduplicate_chunks()
    assert removed == 1
    assert logs[-1] == "Removed 1 duplicate chunks"
    assert ds.history[-1] == logs[-1]
    assert ds.events[-1].operation == "deduplicate_chunks"


def test_cleanup_graph(monkeypatch):
    ds = DatasetBuilder(DatasetType.TEXT)
    ds.add_document("d", source="s")
    ds.add_chunk("d", "c1", "<b>Hello</b>")
    ds.add_chunk("d", "c2", "<b>Hello</b>")
    ds.graph.graph.nodes["c1"]["start_date"] = "1 Feb 2023"

    logs = []
    monkeypatch.setattr(dataset.logger, "info", lambda msg: logs.append(msg))

    removed, cleaned = ds.cleanup_graph()

    assert removed == 1
    assert cleaned == 1
    assert "c2" not in ds.graph.graph.nodes
    assert ds.events[-4].operation == "deduplicate_chunks"
    assert ds.events[-3].operation == "clean_chunks"
    assert ds.events[-2].operation == "normalize_dates"
    assert ds.events[-1].operation == "cleanup_graph"
    assert ds.graph.graph.nodes["c1"]["start_date"] == "2023-02-01"


def test_cleanup_graph_with_params(monkeypatch):
    ds = DatasetBuilder(DatasetType.TEXT)
    ds.add_document("d", source="s")
    ds.add_chunk("d", "c1", "<b>Hello</b>")
    ds.add_entity("e1", "IBM")
    ds.add_entity("e2", "International Business Machines")

    recorded = {}
    norm_called = 0

    def fake_resolve(threshold: float = 0.8, aliases=None):
        recorded["threshold"] = threshold
        recorded["aliases"] = aliases
        return 0

    def fake_norm():
        nonlocal norm_called
        norm_called += 1
        return 0

    monkeypatch.setattr(ds, "resolve_entities", fake_resolve)
    monkeypatch.setattr(ds, "normalize_dates", fake_norm)

    ds.cleanup_graph(
        resolve_threshold=0.9,
        resolve_aliases={"IBM": ["International Business Machines"]},
        normalize_dates=False,
    )

    assert recorded["threshold"] == 0.9
    assert recorded["aliases"] == {"IBM": ["International Business Machines"]}
    assert norm_called == 0


def test_normalize_dates_wrapper():
    ds = DatasetBuilder(DatasetType.TEXT)
    ds.graph.graph.add_node("e1", type="entity", start_date="1 Feb 2023")

    changed = ds.normalize_dates()

    assert changed == 1
    assert ds.graph.graph.nodes["e1"]["start_date"] == "2023-02-01"


def test_enrich_entity_wrapper(monkeypatch):
    ds = DatasetBuilder(DatasetType.TEXT)
    ds.add_entity("e1", "Beethoven")

    class FakeResponse:
        def __init__(self, data):
            self._data = data

        def json(self):
            return self._data

    def fake_get(url, params=None, timeout=10):
        return FakeResponse({"search": [{"id": "Q1", "description": "composer"}]})

    monkeypatch.setattr(requests, "get", fake_get)
    ds.enrich_entity("e1")
    node = ds.graph.graph.nodes["e1"]
    assert node.get("wikidata_id") == "Q1"


def test_enrich_entity_dbpedia_wrapper(monkeypatch):
    ds = DatasetBuilder(DatasetType.TEXT)
    ds.add_entity("e1", "Beethoven")

    class FakeResponse:
        def __init__(self, data):
            self._data = data

        def json(self):
            return self._data

    def fake_get(url, params=None, headers=None, timeout=10):
        return FakeResponse(
            {"results": [{"id": "http://dbpedia.org/resource/Beethoven", "description": "desc"}]}
        )

    monkeypatch.setattr(requests, "get", fake_get)
    ds.enrich_entity_dbpedia("e1")
    node = ds.graph.graph.nodes["e1"]
    assert node.get("dbpedia_uri") == "http://dbpedia.org/resource/Beethoven"
    assert node.get("description_dbpedia") == "desc"


def test_compute_centrality_wrapper():
    ds = DatasetBuilder(DatasetType.TEXT)
    ds.add_document("d", source="s")
    ds.add_chunk("d", "c1", "hello")
    ds.add_chunk("d", "c2", "world")
    ds.add_entity("e1", "A")
    ds.add_entity("e2", "B")
    ds.link_entity("c1", "e1")
    ds.link_entity("c2", "e2")
    ds.compute_centrality()
    assert "centrality" in ds.graph.graph.nodes["e1"]


def test_graph_embeddings_wrapper():
    ds = DatasetBuilder(DatasetType.TEXT)
    ds.add_document("d", source="s")
    ds.add_chunk("d", "c1", "hello")
    ds.add_chunk("d", "c2", "world")
    ds.add_entity("e1", "A")
    ds.add_entity("e2", "B")
    ds.link_entity("c1", "e1")
    ds.link_entity("c2", "e2")
    ds.compute_graph_embeddings(dimensions=8, walk_length=4, num_walks=5, seed=42, workers=1)
    assert len(ds.graph.graph.nodes["e1"]["embedding"]) == 8


def test_graphwave_embeddings_wrapper():
    ds = DatasetBuilder(DatasetType.TEXT)
    ds.add_document("d", source="s")
    ds.add_chunk("d", "c1", "hello")
    ds.add_chunk("d", "c2", "world")
    ds.add_entity("e1", "A")
    ds.add_entity("e2", "B")
    ds.link_entity("c1", "e1")
    ds.link_entity("c2", "e2")
    ds.compute_graphwave_embeddings(scales=[0.5], num_points=4)
    assert len(ds.graph.graph.nodes["e1"]["graphwave_embedding"]) == 8


def test_poincare_embeddings_wrapper():
    ds = DatasetBuilder(DatasetType.TEXT)
    ds.add_document("d", source="s")
    ds.add_chunk("d", "c1", "hello")
    ds.add_chunk("d", "c2", "world")
    ds.add_entity("e1", "A")
    ds.add_entity("e2", "B")
    ds.link_entity("c1", "e1")
    ds.link_entity("c2", "e2")
    ds.compute_poincare_embeddings(dim=2, negative=2, epochs=5)
    assert len(ds.graph.graph.nodes["e1"]["poincare_embedding"]) == 2


def test_graphsage_embeddings_wrapper():
    ds = DatasetBuilder(DatasetType.TEXT)
    ds.add_document("d", source="s")
    ds.add_chunk("d", "c1", "hello")
    ds.add_chunk("d", "c2", "world")
    ds.add_entity("e1", "A")
    ds.add_entity("e2", "B")
    ds.link_entity("c1", "e1")
    ds.link_entity("c2", "e2")
    ds.compute_graphsage_embeddings(dimensions=8, num_layers=1)
    assert len(ds.graph.graph.nodes["e1"]["graphsage_embedding"]) == 8


def test_multigeometric_embeddings_wrapper():
    ds = DatasetBuilder(DatasetType.TEXT)
    ds.add_document("d", source="s")
    ds.add_chunk("d", "c1", "hello")
    ds.add_chunk("d", "c2", "world")
    ds.add_entity("e1", "A")
    ds.add_entity("e2", "B")
    ds.link_entity("c1", "e1")
    ds.link_entity("c2", "e2")
    ds.compute_multigeometric_embeddings(
        node2vec_dim=8, graphwave_scales=[0.5], graphwave_points=4, poincare_dim=2, epochs=5
    )
    node = ds.graph.graph.nodes["e1"]
    assert len(node["embedding"]) == 8
    assert len(node["graphwave_embedding"]) == 8
    assert len(node["poincare_embedding"]) == 2
    assert any(e.operation == "compute_graphsage_embeddings" for e in ds.events)


def test_transe_embeddings_wrapper():
    ds = DatasetBuilder(DatasetType.TEXT)
    ds.add_document("d", source="s")
    ds.add_chunk("d", "c1", "hello")
    ds.add_chunk("d", "c2", "world")
    ds.graph.graph.add_edge("c1", "c2", relation="related")
    ds.compute_transe_embeddings(dimensions=8)
    assert len(ds.graph.graph.edges["c1", "c2"]["transe_embedding"]) == 8
    assert any(e.operation == "compute_transe_embeddings" for e in ds.events)


def test_fractal_dimension_wrapper():
    ds = DatasetBuilder(DatasetType.TEXT)
    ds.add_document("d", source="s")
    ds.add_chunk("d", "c1", "hello")
    ds.add_chunk("d", "c2", "world")
    dim, counts = ds.fractal_dimension([1])
    assert dim >= 0
    assert counts and counts[0][0] == 1


def test_fractalize_level_wrapper():
    ds = DatasetBuilder(DatasetType.TEXT)
    ds.add_document("d", source="s")
    ds.add_chunk("d", "c1", "hello")
    ds.add_chunk("d", "c2", "world")
    coarse, mapping = ds.fractalize_level(1)
    assert coarse.number_of_nodes() >= 1
    assert {"d", "c1", "c2"}.issubset(mapping)


def test_fractalize_optimal_wrapper():
    ds = DatasetBuilder(DatasetType.TEXT)
    ds.add_document("d", source="s")
    ds.add_chunk("d", "c1", "hello")
    ds.add_chunk("d", "c2", "world")
    coarse, mapping, r = ds.fractalize_optimal([1, 2])
    assert coarse.number_of_nodes() >= 1
    assert "d" in mapping
    assert r in {1, 2}


def test_optimize_topology_wrapper():
    ds = DatasetBuilder(DatasetType.TEXT)
    ds.add_document("d", source="s")
    ds.add_chunk("d", "c1", "a")
    ds.add_chunk("d", "c2", "b")
    ds.graph.graph.add_edge("c1", "c2", relation="perception_link")

    target = nx.cycle_graph(3)
    mapping = {i: n for i, n in enumerate(["c1", "c2", "d"])}
    target = nx.relabel_nodes(target, mapping)

    before = bottleneck_distance(ds.graph.graph.to_undirected(), target)
    dist = ds.optimize_topology(target, max_iter=5, seed=0, use_generator=True)
    after = bottleneck_distance(ds.graph.graph.to_undirected(), target)
    assert after <= before
    assert dist == pytest.approx(after, rel=1e-9)


def test_apply_perception_wrapper():
    ds = DatasetBuilder(DatasetType.TEXT)
    ds.add_document("d", source="s")
    ds.add_chunk("d", "c1", "hello")
    ds.graph.index.build()
    ds.apply_perception("c1", "HELLO", perception_id="p1", strength=0.5)
    node = ds.graph.graph.nodes["c1"]
    assert node["text"] == "HELLO"
    assert node.get("perception_id") == "p1"
    assert node.get("perception_strength") == 0.5
    assert "embedding" in node
    assert any(e.operation == "apply_perception" for e in ds.events)


def test_persistence_diagrams_wrapper():
    ds = DatasetBuilder(DatasetType.TEXT)
    ds.add_document("d", source="s")
    ds.add_chunk("d", "c1", "hello")
    diags = ds.persistence_diagrams(max_dim=1)
    assert 0 in diags
    assert diags[0].shape[1] == 2


def test_spectral_dimension_wrapper():
    ds = DatasetBuilder(DatasetType.TEXT)
    ds.add_document("d", source="s")
    ds.add_chunk("d", "c1", "hello")
    ds.add_chunk("d", "c2", "world")
    dim, traces = ds.spectral_dimension([0.1, 0.2])
    assert dim >= 0
    assert traces


def test_spectral_entropy_wrapper():
    ds = DatasetBuilder(DatasetType.TEXT)
    ds.add_document("d", source="s")
    ds.add_chunk("d", "c1", "hello")
    ds.add_chunk("d", "c2", "world")
    ent = ds.spectral_entropy()
    assert ent >= 0
    assert any(e.operation == "spectral_entropy" for e in ds.events)


def test_spectral_gap_wrapper():
    ds = DatasetBuilder(DatasetType.TEXT)
    ds.add_document("d", source="s")
    ds.add_chunk("d", "c1", "hello")
    ds.add_chunk("d", "c2", "world")
    gap = ds.spectral_gap()
    assert gap >= 0
    assert any(e.operation == "spectral_gap" for e in ds.events)


def test_laplacian_energy_wrapper():
    ds = DatasetBuilder(DatasetType.TEXT)
    ds.add_document("d", source="s")
    ds.add_chunk("d", "c1", "hello")
    ds.add_chunk("d", "c2", "world")
    energy = ds.laplacian_energy()
    assert energy >= 0
    assert any(e.operation == "laplacian_energy" for e in ds.events)


def test_laplacian_spectrum_wrapper():
    ds = DatasetBuilder(DatasetType.TEXT)
    ds.add_document("d", source="s")
    ds.add_chunk("d", "c1", "hello")
    ds.add_chunk("d", "c2", "world")
    evals = ds.laplacian_spectrum()
    assert len(evals) == ds.graph.graph.number_of_nodes()
    assert any(e.operation == "laplacian_spectrum" for e in ds.events)


def test_spectral_density_wrapper():
    ds = DatasetBuilder(DatasetType.TEXT)
    ds.add_document("d", source="s")
    ds.add_chunk("d", "c1", "hello")
    ds.add_chunk("d", "c2", "world")
    hist, edges = ds.spectral_density(bins=3)
    assert len(hist) == 3
    assert len(edges) == 4
    assert any(e.operation == "spectral_density" for e in ds.events)


def test_predict_links_wrapper():
    ds = DatasetBuilder(DatasetType.TEXT)
    ds.add_entity("e1", "Beethoven")
    ds.add_entity("e2", "Ludwig van Beethoven")
    ds.predict_links(threshold=0.4)
    assert ds.graph.graph.has_edge("e1", "e2")
    ds2 = DatasetBuilder(DatasetType.TEXT)
    ds2.add_entity("a1", "X")
    ds2.add_entity("a2", "X")
    ds2.add_document("d", source="s")
    ds2.add_chunk("d", "c1", "x")
    ds2.link_entity("c1", "a1")
    ds2.link_entity("c1", "a2")
    ds2.compute_graph_embeddings(dimensions=8, walk_length=4, num_walks=5, seed=42, workers=1)
    ds2.predict_links(threshold=0.1, use_graph_embeddings=True)
    assert "embedding" in ds2.graph.graph.nodes["a1"]


def test_gds_quality_check_wrapper(monkeypatch):
    ds = DatasetBuilder(DatasetType.TEXT, name="qc")
    fake_driver = object()
    ds.neo4j_driver = fake_driver
    called = {}

    def fake_qc(self, driver, *, dataset=None, min_component_size=2, similarity_threshold=0.95):
        called["driver"] = driver
        called["dataset"] = dataset
        return {"removed_nodes": [1]}

    monkeypatch.setattr(ds.graph.__class__, "gds_quality_check", fake_qc, raising=False)

    ds.add_document("d", source="s")
    ds.add_chunk("d", "c1", "hello")
    res = ds.gds_quality_check(driver=fake_driver)
    assert res == {"removed_nodes": [1]}
    assert called["driver"] is fake_driver
    assert called["dataset"] == "qc"
    assert any(e.operation == "gds_quality_check" for e in ds.events)


def test_consolidate_schema_wrapper():
    ds = DatasetBuilder(DatasetType.TEXT)
    ds.add_entity("e1", "A")
    ds.add_entity("e2", "B")
    ds.graph.graph.nodes["e1"]["type"] = "ENTITY"
    ds.graph.graph.add_edge("e1", "e2", relation="RELATED")

    ds.consolidate_schema()

    assert ds.graph.graph.nodes["e1"]["type"] == "entity"
    assert ds.graph.graph.edges["e1", "e2"]["relation"] == "related"


def test_entity_helpers():
    ds = DatasetBuilder(DatasetType.TEXT)
    ds.add_document("doc", source="s")
    ds.add_chunk("doc", "c1", "Paris is nice")
    ds.add_chunk("doc", "c2", "France is big")
    ds.add_entity("Paris", "Paris")
    ds.add_entity("France", "France")
    ds.link_entity("c1", "Paris")
    ds.link_entity("c2", "France")
    fid = ds.graph.add_fact("Paris", "capital_of", "France")

    assert ds.get_chunks_for_entity("Paris") == ["c1"]
    assert ds.get_facts_for_entity("France") == [fid]


def test_fact_helpers():
    ds = DatasetBuilder(DatasetType.TEXT)
    ds.add_document("doc", source="s")
    ds.add_chunk("doc", "c1", "text")
    fid = ds.graph.add_fact("A", "related", "B")
    ds.graph.graph.add_edge("c1", fid, relation="has_fact")

    assert ds.get_facts_for_chunk("c1") == [fid]
    assert ds.get_facts_for_document("doc") == [fid]
    assert ds.get_chunks_for_fact(fid) == ["c1"]
    assert set(ds.get_entities_for_fact(fid)) == {"A", "B"}


def test_find_facts_wrapper():
    ds = DatasetBuilder(DatasetType.TEXT)
    ds.add_document("d", source="s")
    ds.add_chunk("d", "c1", "text")
    f1 = ds.graph.add_fact("A", "likes", "B")
    f2 = ds.graph.add_fact("A", "likes", "C")
    ds.graph.graph.add_edge("c1", f1, relation="has_fact")
    ds.graph.graph.add_edge("c1", f2, relation="has_fact")

    assert set(ds.find_facts(subject="A", predicate="likes")) == {f1, f2}


def test_entity_lookup_helpers_wrapper():
    ds = DatasetBuilder(DatasetType.TEXT)
    ds.add_document("doc", source="s")
    ds.add_chunk("doc", "c1", "Paris is nice")
    ds.add_chunk("doc", "c2", "Berlin is big")
    ds.add_entity("Paris", "Paris")
    ds.add_entity("Berlin", "Berlin")
    ds.link_entity("c1", "Paris")
    ds.link_entity("c2", "Berlin")

    assert ds.get_entities_for_chunk("c1") == ["Paris"]
    assert set(ds.get_entities_for_document("doc")) == {"Paris", "Berlin"}
    assert ds.get_documents_for_entity("Berlin") == ["doc"]


def test_entity_pages_wrapper():
    ds = DatasetBuilder(DatasetType.TEXT)
    ds.add_document("doc", source="s")
    ds.add_chunk("doc", "c1", "Paris", page=3)
    ds.add_entity("Paris", "Paris")
    ds.link_entity("c1", "Paris")

    assert ds.get_pages_for_entity("Paris") == [3]


def test_fact_lookup_helpers_wrapper():
    ds = DatasetBuilder(DatasetType.TEXT)
    ds.add_document("doc", source="s")
    ds.add_section("doc", "sec1")
    ds.add_chunk("doc", "c1", "A is B", section_id="sec1")
    fid = ds.graph.add_fact("A", "is", "B")
    ds.graph.graph.add_edge("c1", fid, relation="has_fact")

    assert ds.get_sections_for_fact(fid) == ["sec1"]
    assert ds.get_documents_for_fact(fid) == ["doc"]
    assert ds.get_pages_for_fact(fid) == [1]


def test_extract_entities_wrapper():
    ds = DatasetBuilder(DatasetType.TEXT)
    ds.add_document("doc", source="s")
    ds.add_chunk("doc", "c1", "Albert Einstein was born in Ulm.")
    ds.extract_entities(model=None)
    ents = set(ds.get_entities_for_chunk("c1"))
    assert "Albert Einstein" in ents
    assert "Ulm" in ents


def test_conflict_detection_wrapper():
    ds = DatasetBuilder(DatasetType.TEXT)
    ds.graph.add_fact("A", "likes", "B", source="s1")
    ds.graph.add_fact("A", "likes", "C", source="s2")
    conflicts = ds.find_conflicting_facts()
    assert conflicts == [
        (
            "A",
            "likes",
            {"B": ["s1"], "C": ["s2"]},
        )
    ]


def test_mark_conflicts_wrapper():
    ds = DatasetBuilder(DatasetType.TEXT)
    f1 = ds.graph.add_fact("A", "likes", "B")
    f2 = ds.graph.add_fact("A", "likes", "C")
    ds.mark_conflicting_facts()
    assert ds.graph.graph.edges["A", "B"].get("conflict") is True
    assert ds.graph.graph.edges["A", "C"].get("conflict") is True


def test_validate_coherence_wrapper():
    ds = DatasetBuilder(DatasetType.TEXT)
    ds.graph.graph.add_node("p", type="entity", birth_date="2024-01-01")
    ds.graph.graph.add_node("c", type="entity", birth_date="2023-01-01")
    ds.graph.graph.add_edge("p", "c", relation="parent_of")

    marked = ds.validate_coherence()

    assert marked == 1
    assert ds.graph.graph.edges["p", "c"].get("inconsistent") is True


def test_get_raw_text_and_run_pipeline(monkeypatch):
    ds = DatasetBuilder(DatasetType.QA)
    ds.add_document("doc", source="s")
    ds.add_chunk("doc", "c1", "hello")
    ds.add_chunk("doc", "c2", "world")

    text = ds.get_raw_text()
    assert "hello" in text
    assert "world" in text

    calls = {}

    def fake_run(dtype, graph, **kwargs):
        calls["dtype"] = dtype
        calls["graph"] = graph
        calls["kwargs"] = kwargs
        calls["builder"] = kwargs.get("dataset_builder")
        return "ok"

    async def fake_run_async(*a, **k):
        return fake_run(*a, **k)

    monkeypatch.setattr(
        "datacreek.pipelines.run_generation_pipeline_async",
        fake_run_async,
    )

    result = ds.run_post_kg_pipeline(
        num_pairs=5,
        async_mode=True,
        batch_size=2,
        inference_batch=1,
        start_step=PipelineStep.CURATE,
    )

    assert result == "ok"
    assert calls["dtype"] == DatasetType.QA
    assert calls["graph"] is ds.graph
    assert calls["kwargs"]["num_pairs"] == 5
    assert calls["kwargs"]["async_mode"] is True
    assert calls["kwargs"]["multi_answer"] is False
    assert calls["kwargs"]["batch_size"] == 2
    assert calls["kwargs"]["start_step"] is PipelineStep.CURATE
    assert calls["builder"] is ds


def test_run_post_kg_pipeline_logs_cleanup(monkeypatch):
    ds = DatasetBuilder(DatasetType.QA)
    ds.add_document("d", source="s")
    ds.add_chunk("d", "c1", "<b>Hello</b>")
    ds.add_chunk("d", "c2", "<b>Hello</b>")

    monkeypatch.setattr("datacreek.pipelines.process_file", lambda *a, **k: {"qa_pairs": []})
    monkeypatch.setattr("datacreek.pipelines.curate_qa_pairs", lambda d, *a, **k: {"qa_pairs": []})
    monkeypatch.setattr("datacreek.pipelines.convert_format", lambda *a, **k: {})

    ds.run_post_kg_pipeline(start_step=PipelineStep.KG_CLEANUP)

    assert "c2" not in ds.graph.graph.nodes
    ops = [e.operation for e in ds.events]
    assert "deduplicate_chunks" in ops
    assert "clean_chunks" in ops


def test_run_post_kg_pipeline_logs_curation(monkeypatch):
    ds = DatasetBuilder(DatasetType.QA)
    ds.add_document("d", source="s")

    monkeypatch.setattr("datacreek.pipelines.process_file", lambda *a, **k: {"qa_pairs": []})
    monkeypatch.setattr(
        "datacreek.pipelines.curate_qa_pairs",
        lambda d, *a, **k: {
            "qa_pairs": [],
            "metrics": {"total": 2, "filtered": 1, "retention_rate": 0.5, "avg_score": 8.0},
        },
    )
    monkeypatch.setattr("datacreek.pipelines.convert_format", lambda *a, **k: {})

    ds.run_post_kg_pipeline(start_step=PipelineStep.CURATE)

    curate_events = [e for e in ds.events if e.operation == "curate"]
    assert any(
        e.params
        == {
            "total": 2,
            "filtered": 1,
            "retention_rate": 0.5,
            "avg_score": 8.0,
        }
        for e in curate_events
    )
    assert ds.events[-1].operation == "generate"


def test_run_post_kg_pipeline_extra_opts(monkeypatch):
    ds = DatasetBuilder(DatasetType.QA)
    ds.add_document("d", source="s")
    called = {}

    def fake_run(
        dtype,
        graph,
        *,
        pipeline_config_path=None,
        dedup_similarity=1.0,
        keep_ratings=False,
        dataset_builder=None,
        **kwargs,
    ):
        called["config"] = pipeline_config_path
        called["sim"] = dedup_similarity
        called["ratings"] = keep_ratings
        called["builder"] = dataset_builder
        return "ok"

    monkeypatch.setattr("datacreek.pipelines.run_generation_pipeline", fake_run)

    res = ds.run_post_kg_pipeline(
        pipeline_config_path=Path("cfg.yml"),
        dedup_similarity=0.95,
        keep_ratings=True,
    )

    assert res == "ok"
    assert called["config"] == Path("cfg.yml")
    assert called["sim"] == 0.95
    assert called["ratings"] is True
    assert called["builder"] is ds


def test_run_post_kg_pipeline_uses_redis(monkeypatch):
    ds = DatasetBuilder(DatasetType.QA, name="demo")
    client = fakeredis.FakeStrictRedis()
    ds.redis_client = client
    ds.add_document("d", source="s")

    received = {}

    def fake_run(dtype, graph, *, redis_client=None, **kwargs):
        received["client"] = redis_client
        return "ok"

    monkeypatch.setattr("datacreek.pipelines.run_generation_pipeline", fake_run)

    ds.run_post_kg_pipeline()
    assert received["client"] is client


def test_atom_and_molecule_wrappers():
    ds = DatasetBuilder(DatasetType.TEXT)
    ds.add_document("d", source="s")
    ds.add_atom("d", "a1", "hello", "NarrativeText")
    ds.add_atom("d", "a2", "world", "NarrativeText")
    ds.add_molecule("d", "m1", ["a1", "a2"])
    assert ds.get_atoms_for_document("d") == ["a1", "a2"]
    assert ds.get_molecules_for_document("d") == ["m1"]
    assert ds.graph.graph.has_edge("m1", "a1")


def test_mark_exported_records_event():
    ds = DatasetBuilder(DatasetType.TEXT)
    ds.mark_exported()
    assert ds.stage == DatasetStage.EXPORTED
    assert ds.events[-1].operation == "export_dataset"


def test_graph_information_bottleneck_wrapper():
    ds = DatasetBuilder(DatasetType.TEXT)
    ds.add_document("d", source="s")
    for i in range(4):
        ds.add_atom("d", f"a{i}", str(i), "text")
    ds.compute_graph_embeddings(dimensions=2, walk_length=2, num_walks=5, workers=1, seed=0)
    labels = {f"a{i}": i % 2 for i in range(4)}
    loss = ds.graph_information_bottleneck(labels, beta=0.5)
    assert loss > 0
    assert ds.events[-1].operation == "graph_information_bottleneck"


def test_graph_fourier_wrappers():
    ds = DatasetBuilder(DatasetType.TEXT)
    ds.add_document("d", source="s")
    ds.add_chunk("d", "c1", "x")
    ds.add_chunk("d", "c2", "y")
    signal = {n: i for i, n in enumerate(ds.graph.graph.nodes)}
    coeffs = ds.graph_fourier_transform(signal)
    recon = ds.inverse_graph_fourier_transform(coeffs)
    for val, node in zip(recon, ds.graph.graph.nodes):
        assert pytest.approx(val, rel=1e-6) == signal[node]
    assert any(e.operation == "graph_fourier_transform" for e in ds.events)
    assert any(e.operation == "inverse_graph_fourier_transform" for e in ds.events)
