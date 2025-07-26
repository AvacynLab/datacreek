import json
import contextlib
import pytest

from datacreek.generators import qa_generator as qg
from datacreek.models.qa import QAPair


class DummyClient:
    """Minimal LLM client returning predictable results."""

    config = {
        "generation": {
            "chunk_size": 10,
            "temperature": 0,
            "overlap": 0,
            "batch_size": 2,
            "chunk_method": "sliding",
            "similarity_drop": 0.0,
            "retrieval_top_k": 2,
            "summary_temperature": 0,
            "summary_max_tokens": 5,
        },
        "curate": {"threshold": 5, "batch_size": 2, "temperature": 0},
        "prompts": {
            "summary": "SUM",
            "kg_summary": "KGSUM",
            "qa_generation": "QA {num_pairs} {summary} {text}",
            "qa_rating": "RATE {pairs}",
        },
    }

    def __init__(self):
        self.calls = []

    def chat_completion(self, messages, temperature=None, max_tokens=None):
        self.calls.append(("chat", messages))
        return "SUMMARY"

    def batch_completion(self, batches, temperature=None, batch_size=None):
        self.calls.append(("batch", len(batches)))
        # return JSON strings for parse_qa_pairs
        return [
            json.dumps([{"question": f"Q{i}", "answer": f"A{i}"}]) for i, _ in enumerate(batches)
        ]

    async def async_batch_completion(self, batches, temperature=None, batch_size=None):
        return self.batch_completion(batches, temperature, batch_size)


class FakeKG:
    class SimpleGraph:
        def __init__(self):
            self.nodes = {}
        def __contains__(self, item):
            return item in self.nodes

    def __init__(self):
        self.graph = self.SimpleGraph()

    def to_text(self):
        return "KG_TEXT"

    def add_document(self, doc, source="inline", text=None):
        self.graph.nodes[doc] = {"text": text or "", "source": source}

    def add_chunk(self, doc, cid, chunk):
        self.graph.nodes[cid] = {"text": chunk, "source": "inline"}

    def search_embeddings(self, q, k=1):
        return []


def fake_split(text, **_):
    return [text[:3], text[3:6]]


def fake_parse_pairs(text):
    data = json.loads(text)
    return [QAPair(question=d["question"], answer=d["answer"]) for d in data]


def fake_parse_ratings(text, original_items=None):
    # Simply assign increasing scores
    rated = []
    for idx, it in enumerate(original_items or []):
        q = it["question"] if isinstance(it, dict) else it.question
        a = it["answer"] if isinstance(it, dict) else it.answer
        rated.append(QAPair(question=q, answer=a, rating=idx + 6))
    return rated


class DummyProgress:
    def update(self, *_, **__):
        pass

@contextlib.contextmanager
def dummy_progress(*_, **__):
    yield DummyProgress(), 1


def setup_monkeypatch(monkeypatch):
    monkeypatch.setattr(qg, "split_into_chunks", fake_split)
    import datacreek.utils.batch as batch

    def _proc_batches(_client, messages, **kwargs):
        return [fake_parse_pairs('[{"question":"Q","answer":"A"}]') for _ in messages]

    async def _aproc_batches(_client, messages, **kwargs):
        return [fake_parse_pairs('[{"question":"Q","answer":"A"}]') for _ in messages]

    monkeypatch.setattr(batch, "process_batches", _proc_batches)
    monkeypatch.setattr(batch, "async_process_batches", _aproc_batches)
    monkeypatch.setattr(qg, "parse_qa_pairs", fake_parse_pairs)
    monkeypatch.setattr(qg, "parse_ratings", fake_parse_ratings)
    monkeypatch.setattr(qg, "progress_context", dummy_progress)
    monkeypatch.setattr(qg, "get_prompt", lambda cfg, name: cfg["prompts"][name])


def test_generate_summary_with_kg(monkeypatch):
    client = DummyClient()
    setup_monkeypatch(monkeypatch)
    kg = FakeKG()
    gen = qg.QAGenerator(client, kg=kg)
    summary = gen.generate_summary("DOC", verbose=True)
    assert summary == "SUMMARY"
    assert client.calls[0][0] == "chat"
    # ensure KG text used when KG provided
    assert client.calls[0][1][1]["content"] == kg.to_text()


@pytest.mark.asyncio
async def test_generate_pairs_with_query(monkeypatch):
    client = DummyClient()
    setup_monkeypatch(monkeypatch)
    kg = FakeKG()
    kg.search_embeddings = lambda q, k=1: ["chunk-0"]
    gen = qg.QAGenerator(client, kg=kg)
    res = await gen.generate_qa_pairs_async("DOCUMENT", "SUM", num_pairs=1, query="q")
    assert len(res) == 1


def test_process_document_sync(monkeypatch):
    client = DummyClient()
    setup_monkeypatch(monkeypatch)
    gen = qg.QAGenerator(client)
    result = gen.process_document("DOCUMENT", num_pairs=2, verbose=True)
    assert len(result.qa_pairs) == 2
    assert result.summary == "SUMMARY"
    assert result.qa_pairs[0].chunk == "DOC"


@pytest.mark.asyncio
async def test_process_document_async(monkeypatch):
    client = DummyClient()
    setup_monkeypatch(monkeypatch)
    gen = qg.QAGenerator(client)
    result = await gen.process_document_async("DOC", num_pairs=1, verbose=False)
    assert len(result.qa_pairs) == 2


@pytest.mark.asyncio
async def test_rate_pairs_async(monkeypatch):
    client = DummyClient()
    setup_monkeypatch(monkeypatch)
    gen = qg.QAGenerator(client)
    pairs = [QAPair(question="Q", answer="A"), QAPair(question="Q2", answer="A2")]
    rated, metrics = await gen.rate_qa_pairs_async(pairs, "SUM")
    assert metrics["filtered"] == 2
    assert rated[0]["rating"] == 6


def test_rate_pairs_sync(monkeypatch):
    client = DummyClient()
    setup_monkeypatch(monkeypatch)
    gen = qg.QAGenerator(client)
    pairs = [QAPair(question="Q", answer="A")]
    rated, metrics = gen.rate_qa_pairs(pairs, "SUM")
    assert metrics["filtered"] == 1
