from datacreek.pipelines import (
    DatasetType,
    TrainingGoal,
    get_dataset_types_for_training,
    get_pipelines_for_training,
    get_trainings_for_dataset,
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
            assert pipeline.steps[1] == "to_kg"
