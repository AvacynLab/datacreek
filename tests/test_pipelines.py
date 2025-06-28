import pytest
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
        assert args[5] == "qa"
        assert kwargs.get("async_mode") is True
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

    result = run_generation_pipeline(DatasetType.QA, "text", verbose=True, async_mode=True)

    assert result == "done"
    assert calls == ["gen", "curate", "save"]


def test_run_generation_pipeline_invalid():
    with pytest.raises(ValueError):
        run_generation_pipeline("wrong", "text")


def test_run_generation_pipeline_cot(monkeypatch):
    calls = []

    def fake_generate(*args, **kwargs):
        assert args[5] == "cot"
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

    result = run_generation_pipeline(DatasetType.COT, "text")

    assert result == "done"
    assert calls == ["gen", "curate", "save"]


def test_run_generation_pipeline_vqa(monkeypatch):
    calls = []

    def fake_generate(*args, **kwargs):
        # VQA pipeline uses the special content type
        assert args[5] == "vqa_add_reasoning"
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

    result = run_generation_pipeline(DatasetType.VQA, "text")

    assert result == "done"
    assert calls == ["gen", "curate", "save"]


def test_run_generation_pipeline_overrides(monkeypatch):
    received = {}

    def fake_generate(*args, **kwargs):
        received["overrides"] = kwargs.get("config_overrides")
        return {"qa_pairs": []}

    monkeypatch.setattr("datacreek.pipelines.process_file", fake_generate)
    monkeypatch.setattr("datacreek.pipelines.curate_qa_pairs", lambda *a, **k: {})
    monkeypatch.setattr("datacreek.pipelines.convert_format", lambda *a, **k: "done")

    run_generation_pipeline(DatasetType.QA, "text", overrides={"foo": 1})

    assert received["overrides"] == {"foo": 1}
