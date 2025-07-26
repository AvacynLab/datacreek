import types
import sys
import pytest

from datacreek.generators import pref_generator as pg
from datacreek.models.qa import QAPair
from datacreek.models.results import QAGenerationResult


class DummyClient:
    """Minimal LLM client used for preference tests."""

    config = {"generation": {"temperature": 0}}


class DummyQAGenerator:
    """Return a predictable list of QA pairs."""

    def __init__(self, *_, **__):
        pass

    def process_document(self, text, num_pairs=25, verbose=False):
        return self._make_result(num_pairs)

    async def process_document_async(self, text, num_pairs=25, verbose=False):
        return self._make_result(num_pairs)

    @staticmethod
    def _make_result(num_pairs):
        pairs = [
            QAPair(question=f"Q{i}", answer=f"A{i}", chunk=f"C{i}", source=f"S{i}")
            for i in range(num_pairs)
        ]
        return QAGenerationResult("SUM", pairs)


def setup_dummy(monkeypatch):
    module = types.ModuleType("datacreek.generators.qa_generator")
    module.QAGenerator = DummyQAGenerator
    monkeypatch.setitem(sys.modules, "datacreek.generators.qa_generator", module)


def test_pair_generation_sync(monkeypatch):
    setup_dummy(monkeypatch)
    gen = pg.PrefPairGenerator(DummyClient())
    result = gen.process_document("txt", num_pairs=2, verbose=True)
    assert result.summary == "SUM"
    # QAGenerator should receive twice the number of requested pairs
    assert len(result.pairs) == 2
    assert result.pairs[0]["question"] == "Q0"
    assert result.pairs[0]["chosen"] == "A0"
    assert result.pairs[0]["rejected"] == "A1"


@pytest.mark.asyncio
async def test_pair_generation_async(monkeypatch):
    setup_dummy(monkeypatch)
    gen = pg.PrefPairGenerator(DummyClient())
    res = await gen.process_document_async("txt", num_pairs=3, verbose=False)
    # Expect three pairs assembled from six QA pairs
    assert len(res.pairs) == 3
    assert res.pairs[-1]["rejected"] == "A5"


def test_list_generation(monkeypatch):
    setup_dummy(monkeypatch)
    gen = pg.PrefListGenerator(DummyClient())
    res = gen.process_document("txt", num_lists=2, list_size=2)
    assert len(res.responses) == 2
    assert res.responses[0]["answers"][1]["text"] == "A1"


def test_insufficient_pairs(monkeypatch):
    """Handle case where fewer QA pairs than requested are produced."""

    class ShortQAGenerator(DummyQAGenerator):
        @staticmethod
        def _make_result(num_pairs):
            # Always return only two pairs regardless of request
            pairs = [
                QAPair(question="Q0", answer="A0"),
                QAPair(question="Q1", answer="A1"),
            ]
            return QAGenerationResult("SUM", pairs)

    module = types.ModuleType("datacreek.generators.qa_generator")
    module.QAGenerator = ShortQAGenerator
    monkeypatch.setitem(sys.modules, "datacreek.generators.qa_generator", module)

    gen = pg.PrefListGenerator(DummyClient())
    res = gen.process_document("text", num_lists=3, list_size=2)
    # Only one list can be formed from the two pairs
    assert len(res.responses) == 1

