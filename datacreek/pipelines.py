from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List


class TrainingGoal(str, Enum):
    """Supported training objectives."""

    SFT = "sft"
    CPT = "cpt"
    RLAIF = "rlaif"
    GRPO = "grpo"
    PPO = "ppo"
    DPO = "dpo"
    ORPO = "orpo"
    DPO_SFT = "dpo_sft"
    RRHF = "rrhf"


class DatasetType(str, Enum):
    """Type of dataset produced by generation pipelines."""

    QA = "qa"
    COT = "cot"
    VQA = "vqa"
    TEXT = "text"
    KG = "kg"
    PREF_PAIR = "pref_pair"
    PREF_LIST = "pref_list"
    TOOL = "tool"
    CONVERSATION = "conversation"
    MULTI_TOOL = "multi_tool"


@dataclass
class GenerationPipeline:
    """Definition of a generation pipeline."""

    dataset_type: DatasetType
    steps: List[str]
    compatible_trainings: List[TrainingGoal]
    description: str


PIPELINES: Dict[DatasetType, GenerationPipeline] = {
    DatasetType.QA: GenerationPipeline(
        dataset_type=DatasetType.QA,
        steps=["ingest", "to_kg", "generate_qa", "curate", "save"],
        compatible_trainings=[
            TrainingGoal.SFT,
            TrainingGoal.DPO,
            TrainingGoal.ORPO,
            TrainingGoal.DPO_SFT,
            TrainingGoal.PPO,
            TrainingGoal.RRHF,
            TrainingGoal.RLAIF,
            TrainingGoal.GRPO,
        ],
        description="Question-answer pairs for instruction tuning and preference based training.",
    ),
    DatasetType.COT: GenerationPipeline(
        dataset_type=DatasetType.COT,
        steps=["ingest", "to_kg", "generate_cot", "curate", "save"],
        compatible_trainings=[
            TrainingGoal.SFT,
            TrainingGoal.DPO,
            TrainingGoal.ORPO,
            TrainingGoal.DPO_SFT,
            TrainingGoal.RRHF,
        ],
        description="Chain-of-thought traces to teach stepwise reasoning.",
    ),
    DatasetType.VQA: GenerationPipeline(
        dataset_type=DatasetType.VQA,
        steps=["ingest", "to_kg", "generate_vqa", "curate", "save"],
        compatible_trainings=[TrainingGoal.SFT],
        description="Visual question answering pairs.",
    ),
    DatasetType.TEXT: GenerationPipeline(
        dataset_type=DatasetType.TEXT,
        steps=["ingest", "to_kg", "save"],
        compatible_trainings=[TrainingGoal.CPT],
        description="Raw text corpus for continual pre-training.",
    ),
    DatasetType.KG: GenerationPipeline(
        dataset_type=DatasetType.KG,
        steps=["ingest", "to_kg", "generate_from_kg", "curate", "save"],
        compatible_trainings=[
            TrainingGoal.SFT,
            TrainingGoal.DPO,
            TrainingGoal.ORPO,
            TrainingGoal.DPO_SFT,
            TrainingGoal.PPO,
            TrainingGoal.RRHF,
            TrainingGoal.RLAIF,
            TrainingGoal.GRPO,
        ],
        description="Question answering data generated from a knowledge graph.",
    ),
    DatasetType.PREF_PAIR: GenerationPipeline(
        dataset_type=DatasetType.PREF_PAIR,
        steps=["ingest", "to_kg", "generate_candidates", "label_pairs", "save"],
        compatible_trainings=[
            TrainingGoal.PPO,
            TrainingGoal.DPO,
            TrainingGoal.ORPO,
            TrainingGoal.DPO_SFT,
            TrainingGoal.RLAIF,
        ],
        description="Pairwise preferences for reward-model and preference-based training.",
    ),
    DatasetType.PREF_LIST: GenerationPipeline(
        dataset_type=DatasetType.PREF_LIST,
        steps=["ingest", "to_kg", "generate_candidates", "rank_responses", "save"],
        compatible_trainings=[TrainingGoal.GRPO, TrainingGoal.RRHF],
        description="Listwise ranked responses for GRPO or RRHF.",
    ),
    DatasetType.TOOL: GenerationPipeline(
        dataset_type=DatasetType.TOOL,
        steps=["ingest", "to_kg", "generate_tool_call", "curate", "save"],
        compatible_trainings=[
            TrainingGoal.SFT,
            TrainingGoal.DPO,
            TrainingGoal.ORPO,
            TrainingGoal.DPO_SFT,
            TrainingGoal.PPO,
            TrainingGoal.RRHF,
            TrainingGoal.RLAIF,
            TrainingGoal.GRPO,
        ],
        description="Single tool-calling demonstrations with integrated results.",
    ),
    DatasetType.CONVERSATION: GenerationPipeline(
        dataset_type=DatasetType.CONVERSATION,
        steps=["ingest", "to_kg", "generate_conversation", "curate", "save"],
        compatible_trainings=[
            TrainingGoal.SFT,
            TrainingGoal.DPO,
            TrainingGoal.ORPO,
            TrainingGoal.DPO_SFT,
            TrainingGoal.PPO,
            TrainingGoal.RRHF,
            TrainingGoal.RLAIF,
            TrainingGoal.GRPO,
        ],
        description="Multi-turn conversations for dialogue training.",
    ),
    DatasetType.MULTI_TOOL: GenerationPipeline(
        dataset_type=DatasetType.MULTI_TOOL,
        steps=["ingest", "to_kg", "generate_multi_tool", "curate", "save"],
        compatible_trainings=[
            TrainingGoal.SFT,
            TrainingGoal.DPO,
            TrainingGoal.ORPO,
            TrainingGoal.DPO_SFT,
            TrainingGoal.PPO,
            TrainingGoal.RRHF,
            TrainingGoal.RLAIF,
            TrainingGoal.GRPO,
        ],
        description="Sequential multi-tool use traces for complex tasks.",
    ),
}


def get_pipeline(dataset_type: DatasetType) -> GenerationPipeline:
    """Return pipeline information for a dataset type."""

    if dataset_type not in PIPELINES:
        raise KeyError(f"Unknown dataset type: {dataset_type}")
    return PIPELINES[dataset_type]


def get_trainings_for_dataset(dataset_type: DatasetType) -> List[TrainingGoal]:
    """Return compatible trainings for a dataset type."""

    return get_pipeline(dataset_type).compatible_trainings


def get_dataset_types_for_training(goal: TrainingGoal) -> List[DatasetType]:
    """Return dataset types that can be used for the given training goal."""

    return [
        pipeline.dataset_type
        for pipeline in PIPELINES.values()
        if goal in pipeline.compatible_trainings
    ]


def get_pipelines_for_training(goal: TrainingGoal) -> List[GenerationPipeline]:
    """Return generation pipelines compatible with the given training goal."""

    return [pipeline for pipeline in PIPELINES.values() if goal in pipeline.compatible_trainings]
