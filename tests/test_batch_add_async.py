from datacreek.generators.qa_generator import QAGenerator


class DummyClient:
    def __init__(self):
        self.config = {
            "prompts": {"qa_rating": "{pairs}"},
            "curate": {"batch_size": 1, "temperature": 0.1, "threshold": 0.0},
        }


async_called = {}


async def fake_async(client, messages, *, batch_size, temperature, parse_fn):
    async_called["count"] = len(messages)
    return ["{}"] * len(messages)


def test_rate_qa_pairs_async(monkeypatch):
    monkeypatch.setattr("datacreek.utils.batch.async_process_batches", fake_async)
    monkeypatch.setattr(
        "datacreek.generators.qa_generator.parse_ratings", lambda resp, orig: [{"rating": 9}]
    )

    gen = QAGenerator(DummyClient())
    pairs, metrics = gen.rate_qa_pairs([{"question": "q", "answer": "a"}], "", async_mode=True)

    assert async_called.get("count") == 1
    assert pairs == [{"rating": 9}]
