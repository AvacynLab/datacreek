import pytest

from datacreek.core.knowledge_graph import KnowledgeGraph
from datacreek.models.content_type import ContentType
from datacreek.pipelines import (
    DatasetType,
    PipelineStep,
    TrainingGoal,
    get_dataset_types_for_training,
    get_pipelines_for_training,
    get_trainings_for_dataset,
    run_generation_pipeline,
)


def test_get_trainings_for_dataset():
    qa_trainings = get_trainings_for_dataset(DatasetType.QA)
    assert TrainingGoal.SFT in qa_trainings
    assert TrainingGoal.CPT not in qa_trainings

    text_trainings = get_trainings_for_dataset(DatasetType.TEXT)
    assert text_trainings == [TrainingGoal.CPT]

    pair_trainings = get_trainings_for_dataset(DatasetType.PREF_PAIR)
    assert TrainingGoal.DPO in pair_trainings
    assert TrainingGoal.GRPO not in pair_trainings

    kg_trainings = get_trainings_for_dataset(DatasetType.KG)
    assert TrainingGoal.SFT in kg_trainings
    assert TrainingGoal.CPT not in kg_trainings

    tool_trainings = get_trainings_for_dataset(DatasetType.TOOL)
    assert TrainingGoal.SFT in tool_trainings
    assert TrainingGoal.CPT not in tool_trainings


def test_reverse_mapping():
    qa_datasets = get_dataset_types_for_training(TrainingGoal.SFT)
    assert DatasetType.QA in qa_datasets
    assert DatasetType.TEXT not in qa_datasets

    cpt_pipes = get_pipelines_for_training(TrainingGoal.CPT)
    assert len(cpt_pipes) == 1
    assert cpt_pipes[0].dataset_type == DatasetType.TEXT

    rrhf_datasets = get_dataset_types_for_training(TrainingGoal.RRHF)
    assert DatasetType.PREF_LIST in rrhf_datasets

    sft_datasets = get_dataset_types_for_training(TrainingGoal.SFT)
    assert DatasetType.KG in sft_datasets
    assert DatasetType.TOOL in sft_datasets
    assert DatasetType.CONVERSATION in sft_datasets
    assert DatasetType.MULTI_TOOL in sft_datasets


def test_pipelines_include_kg_step():
    for pipeline in get_pipelines_for_training(TrainingGoal.SFT):
        if pipeline.dataset_type != DatasetType.TEXT:
            assert pipeline.steps[1] == PipelineStep.TO_KG


def test_run_generation_pipeline(monkeypatch):
    calls = []

    def fake_generate(*args, **kwargs):
        # content_type is passed positionally
        assert args[5] == ContentType.QA
        assert kwargs.get("async_mode") is True
        assert "kg" in kwargs
        calls.append("gen")
        return {"qa_pairs": [{"question": "q", "answer": "a"}]}

    def fake_curate(data, *args, **kwargs):
        assert kwargs.get("async_mode") is True
        calls.append("curate")
        return {"qa_pairs": data["qa_pairs"], "summary": ""}

    def fake_save(data, output_path, fmt, cfg, storage_format="json"):
        calls.append("save")
        assert fmt == "jsonl"
        return "done"

    monkeypatch.setattr("datacreek.pipelines.process_file", fake_generate)
    monkeypatch.setattr("datacreek.pipelines.curate_qa_pairs", fake_curate)
    monkeypatch.setattr("datacreek.pipelines.convert_format", fake_save)

    kg = KnowledgeGraph()
    kg.add_document("d", source="s", text="text")

    result = run_generation_pipeline(DatasetType.QA, kg, verbose=True, async_mode=True)

    assert result == "done"
    assert calls == ["gen", "curate", "save"]


def test_run_generation_pipeline_invalid():
    with pytest.raises(ValueError):
        kg = KnowledgeGraph()
        kg.add_document("d", source="s", text="text")
        run_generation_pipeline("wrong", kg)


def test_run_generation_pipeline_cot(monkeypatch):
    calls = []

    def fake_generate(*args, **kwargs):
        assert args[5] == ContentType.COT
        assert "kg" in kwargs
        calls.append("gen")
        return {"cot_examples": [{"question": "q", "reasoning": "r", "answer": "a"}]}

    def fake_curate(data, *args, **kwargs):
        calls.append("curate")
        return {"qa_pairs": [{"question": "q", "answer": "a"}], "summary": ""}

    def fake_save(data, output_path, fmt, cfg, storage_format="json"):
        calls.append("save")
        return "done"

    monkeypatch.setattr("datacreek.pipelines.process_file", fake_generate)
    monkeypatch.setattr("datacreek.pipelines.curate_qa_pairs", fake_curate)
    monkeypatch.setattr("datacreek.pipelines.convert_format", fake_save)

    kg = KnowledgeGraph()
    kg.add_document("d", source="s", text="text")

    result = run_generation_pipeline(DatasetType.COT, kg)

    assert result == "done"
    assert calls == ["gen", "curate", "save"]


def test_run_generation_pipeline_vqa(monkeypatch):
    calls = []

    def fake_generate(*args, **kwargs):
        # VQA pipeline uses the special content type
        assert args[5] == ContentType.VQA_ADD_REASONING
        assert "kg" in kwargs
        calls.append("gen")
        return {"qa_pairs": [{"question": "q", "answer": "a"}]}

    def fake_curate(data, *args, **kwargs):
        calls.append("curate")
        return {"qa_pairs": data["qa_pairs"], "summary": ""}

    def fake_save(data, output_path, fmt, cfg, storage_format="json"):
        calls.append("save")
        return "done"

    monkeypatch.setattr("datacreek.pipelines.process_file", fake_generate)
    monkeypatch.setattr("datacreek.pipelines.curate_qa_pairs", fake_curate)
    monkeypatch.setattr("datacreek.pipelines.convert_format", fake_save)

    kg = KnowledgeGraph()
    kg.add_document("d", source="s", text="img")

    result = run_generation_pipeline(DatasetType.VQA, kg)

    assert result == "done"
    assert calls == ["gen", "curate", "save"]


def test_run_generation_pipeline_tool(monkeypatch):
    calls = []

    def fake_generate(*args, **kwargs):
        assert args[5] == ContentType.TOOL_CALL
        assert "kg" in kwargs
        calls.append("gen")
        return {"conversations": []}

    monkeypatch.setattr("datacreek.pipelines.process_file", fake_generate)
    monkeypatch.setattr("datacreek.pipelines.curate_qa_pairs", lambda *a, **k: {})
    monkeypatch.setattr("datacreek.pipelines.convert_format", lambda *a, **k: "done")

    kg = KnowledgeGraph()
    kg.add_document("d", source="s", text="text")

    result = run_generation_pipeline(DatasetType.TOOL, kg)

    assert result == "done"
    assert calls == ["gen"]


def test_run_generation_pipeline_conversation(monkeypatch):
    calls = []

    def fake_generate(*args, **kwargs):
        assert args[5] == ContentType.CONVERSATION
        assert "kg" in kwargs
        calls.append("gen")
        return {"conversations": []}

    monkeypatch.setattr("datacreek.pipelines.process_file", fake_generate)
    monkeypatch.setattr("datacreek.pipelines.curate_qa_pairs", lambda *a, **k: {})
    monkeypatch.setattr("datacreek.pipelines.convert_format", lambda *a, **k: "done")

    kg = KnowledgeGraph()
    kg.add_document("d", source="s", text="text")

    result = run_generation_pipeline(DatasetType.CONVERSATION, kg)

    assert result == "done"
    assert calls == ["gen"]


def test_run_generation_pipeline_multi_tool(monkeypatch):
    calls = []

    def fake_generate(*args, **kwargs):
        assert args[5] == ContentType.MULTI_TOOL
        assert "kg" in kwargs
        calls.append("gen")
        return {"conversations": []}

    monkeypatch.setattr("datacreek.pipelines.process_file", fake_generate)
    monkeypatch.setattr("datacreek.pipelines.curate_qa_pairs", lambda *a, **k: {})
    monkeypatch.setattr("datacreek.pipelines.convert_format", lambda *a, **k: "done")

    kg = KnowledgeGraph()
    kg.add_document("d", source="s", text="text")

    result = run_generation_pipeline(DatasetType.MULTI_TOOL, kg)

    assert result == "done"
    assert calls == ["gen"]


def test_run_generation_pipeline_pref_pair(monkeypatch):
    calls = []

    def fake_generate(*args, **kwargs):
        assert args[5] == ContentType.PREF_PAIR
        assert "kg" in kwargs
        calls.append("gen")
        return {"pairs": []}

    monkeypatch.setattr("datacreek.pipelines.process_file", fake_generate)
    monkeypatch.setattr("datacreek.pipelines.curate_qa_pairs", lambda *a, **k: {})
    monkeypatch.setattr("datacreek.pipelines.convert_format", lambda *a, **k: "done")

    kg = KnowledgeGraph()
    kg.add_document("d", source="s", text="text")

    result = run_generation_pipeline(DatasetType.PREF_PAIR, kg)

    assert result == "done"
    assert calls == ["gen"]


def test_run_generation_pipeline_pref_list(monkeypatch):
    calls = []

    def fake_generate(*args, **kwargs):
        assert args[5] == ContentType.PREF_LIST
        assert "kg" in kwargs
        calls.append("gen")
        return {"responses": []}

    monkeypatch.setattr("datacreek.pipelines.process_file", fake_generate)
    monkeypatch.setattr("datacreek.pipelines.curate_qa_pairs", lambda *a, **k: {})
    monkeypatch.setattr("datacreek.pipelines.convert_format", lambda *a, **k: "done")

    kg = KnowledgeGraph()
    kg.add_document("d", source="s", text="text")

    result = run_generation_pipeline(DatasetType.PREF_LIST, kg)

    assert result == "done"
    assert calls == ["gen"]


def test_run_generation_pipeline_from_kg(monkeypatch):
    calls = []

    def fake_generate(*args, **kwargs):
        assert args[5] == ContentType.FROM_KG
        assert "kg" in kwargs
        assert kwargs.get("multi_answer") is True
        calls.append("gen")
        return {"qa_pairs": []}

    monkeypatch.setattr("datacreek.pipelines.process_file", fake_generate)
    monkeypatch.setattr("datacreek.pipelines.curate_qa_pairs", lambda *a, **k: {})
    monkeypatch.setattr("datacreek.pipelines.convert_format", lambda *a, **k: "done")

    kg = KnowledgeGraph()
    kg.add_document("d", source="s", text="text")

    result = run_generation_pipeline(DatasetType.KG, kg, multi_answer=True)

    assert result == "done"
    assert calls == ["gen"]


def test_run_generation_pipeline_overrides(monkeypatch):
    received = {}

    def fake_generate(*args, **kwargs):
        received["overrides"] = kwargs.get("config_overrides")
        assert "kg" in kwargs
        return {"qa_pairs": []}

    monkeypatch.setattr("datacreek.pipelines.process_file", fake_generate)
    monkeypatch.setattr("datacreek.pipelines.curate_qa_pairs", lambda *a, **k: {})
    monkeypatch.setattr("datacreek.pipelines.convert_format", lambda *a, **k: "done")

    kg = KnowledgeGraph()
    kg.add_document("d", source="s", text="text")

    run_generation_pipeline(DatasetType.QA, kg, overrides={"foo": 1})

    assert received["overrides"] == {"foo": 1}


def test_run_generation_pipeline_unsupported(monkeypatch):
    """Ensure unsupported dataset types raise errors."""

    with pytest.raises(ValueError):
        kg = KnowledgeGraph()
        kg.add_document("d", source="s", text="text")
        run_generation_pipeline(DatasetType.TEXT, kg)
