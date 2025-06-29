from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List

from datacreek.core.create import process_file
from datacreek.core.curate import curate_qa_pairs
from datacreek.core.save_as import convert_format
from datacreek.models.content_type import ContentType
from datacreek.utils.config import get_format_settings, load_config, merge_configs

logger = logging.getLogger(__name__)


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


class PipelineStep(str, Enum):
    """Valid generation pipeline steps."""

    INGEST = "ingest"
    TO_KG = "to_kg"
    GENERATE_QA = "generate_qa"
    GENERATE_COT = "generate_cot"
    GENERATE_VQA = "generate_vqa"
    GENERATE_FROM_KG = "generate_from_kg"
    GENERATE_CANDIDATES = "generate_candidates"
    LABEL_PAIRS = "label_pairs"
    RANK_RESPONSES = "rank_responses"
    GENERATE_TOOL_CALL = "generate_tool_call"
    GENERATE_CONVERSATION = "generate_conversation"
    GENERATE_MULTI_TOOL = "generate_multi_tool"
    CURATE = "curate"
    SAVE = "save"


@dataclass
class GenerationPipeline:
    """Definition of a generation pipeline."""

    dataset_type: DatasetType
    steps: List[PipelineStep]
    compatible_trainings: List[TrainingGoal]
    description: str


@dataclass
class ProcessOptions:
    """Grouped parameters for ``process_file`` calls."""

    config_path: Path | None = None
    provider: str | None = None
    profile: str | None = None
    api_base: str | None = None
    model: str | None = None
    num_pairs: int | None = None
    overrides: Dict[str, Any] | None = None
    verbose: bool = False
    async_mode: bool = False


PIPELINES: Dict[DatasetType, GenerationPipeline] = {
    DatasetType.QA: GenerationPipeline(
        dataset_type=DatasetType.QA,
        steps=[
            PipelineStep.INGEST,
            PipelineStep.TO_KG,
            PipelineStep.GENERATE_QA,
            PipelineStep.CURATE,
            PipelineStep.SAVE,
        ],
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
        steps=[
            PipelineStep.INGEST,
            PipelineStep.TO_KG,
            PipelineStep.GENERATE_COT,
            PipelineStep.CURATE,
            PipelineStep.SAVE,
        ],
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
        steps=[
            PipelineStep.INGEST,
            PipelineStep.TO_KG,
            PipelineStep.GENERATE_VQA,
            PipelineStep.CURATE,
            PipelineStep.SAVE,
        ],
        compatible_trainings=[TrainingGoal.SFT],
        description="Visual question answering pairs.",
    ),
    DatasetType.TEXT: GenerationPipeline(
        dataset_type=DatasetType.TEXT,
        steps=[PipelineStep.INGEST, PipelineStep.TO_KG, PipelineStep.SAVE],
        compatible_trainings=[TrainingGoal.CPT],
        description="Raw text corpus for continual pre-training.",
    ),
    DatasetType.KG: GenerationPipeline(
        dataset_type=DatasetType.KG,
        steps=[
            PipelineStep.INGEST,
            PipelineStep.TO_KG,
            PipelineStep.GENERATE_FROM_KG,
            PipelineStep.CURATE,
            PipelineStep.SAVE,
        ],
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
        steps=[
            PipelineStep.INGEST,
            PipelineStep.TO_KG,
            PipelineStep.GENERATE_CANDIDATES,
            PipelineStep.LABEL_PAIRS,
            PipelineStep.SAVE,
        ],
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
        steps=[
            PipelineStep.INGEST,
            PipelineStep.TO_KG,
            PipelineStep.GENERATE_CANDIDATES,
            PipelineStep.RANK_RESPONSES,
            PipelineStep.SAVE,
        ],
        compatible_trainings=[TrainingGoal.GRPO, TrainingGoal.RRHF],
        description="Listwise ranked responses for GRPO or RRHF.",
    ),
    DatasetType.TOOL: GenerationPipeline(
        dataset_type=DatasetType.TOOL,
        steps=[
            PipelineStep.INGEST,
            PipelineStep.TO_KG,
            PipelineStep.GENERATE_TOOL_CALL,
            PipelineStep.CURATE,
            PipelineStep.SAVE,
        ],
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
        steps=[
            PipelineStep.INGEST,
            PipelineStep.TO_KG,
            PipelineStep.GENERATE_CONVERSATION,
            PipelineStep.CURATE,
            PipelineStep.SAVE,
        ],
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
        steps=[
            PipelineStep.INGEST,
            PipelineStep.TO_KG,
            PipelineStep.GENERATE_MULTI_TOOL,
            PipelineStep.CURATE,
            PipelineStep.SAVE,
        ],
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


def run_generation_pipeline(
    dataset_type: DatasetType,
    document_text: str,
    *,
    config_path: Path | None = None,
    provider: str | None = None,
    profile: str | None = None,
    api_base: str | None = None,
    model: str | None = None,
    num_pairs: int | None = None,
    threshold: float | None = None,
    fmt: str | None = None,
    overrides: Dict[str, Any] | None = None,
    verbose: bool = False,
    async_mode: bool = False,
) -> Any:
    """Execute the generation steps for ``dataset_type`` on ``document_text``.

    Only steps after the knowledge graph construction are executed. In practice
    this means generation, curation and formatting steps. The function returns
    the final in-memory representation of the dataset without writing files.
    When ``async_mode`` is True, generation steps use asynchronous LLM requests
    when supported by the client.

    At the moment only the QA, COT and VQA pipelines are implemented here. They
    share a common post-knowledge-graph flow consisting of generation, optional
    curation and output formatting. Other dataset types require additional
    specialised steps which are not yet supported.
    """

    try:
        pipeline = get_pipeline(dataset_type)
    except KeyError as exc:
        raise ValueError(str(exc)) from exc

    # The TEXT dataset does not include generation after the knowledge graph,
    # so it remains unsupported by this helper.
    if dataset_type == DatasetType.TEXT:
        raise ValueError("run_generation_pipeline does not support the TEXT dataset type")

    options = ProcessOptions(
        config_path=config_path,
        provider=provider,
        profile=profile,
        api_base=api_base,
        model=model,
        num_pairs=num_pairs,
        overrides=overrides,
        verbose=verbose,
        async_mode=async_mode,
    )
    data: Any = document_text

    def _save(d: Any) -> Any:
        cfg = load_config(str(config_path) if config_path else None)
        if overrides:
            cfg = merge_configs(cfg, overrides)
        fmt_cfg = get_format_settings(cfg)
        format_type = fmt or fmt_cfg.default
        return convert_format(d, None, format_type, cfg)

    handlers = {
        PipelineStep.GENERATE_QA: lambda d: process_file(
            None,
            None,
            options.config_path,
            options.api_base,
            options.model,
            ContentType.QA,
            options.num_pairs,
            options.verbose,
            async_mode=options.async_mode,
            provider=options.provider,
            profile=options.profile,
            document_text=d,
            config_overrides=options.overrides,
        ),
        PipelineStep.GENERATE_COT: lambda d: process_file(
            None,
            None,
            options.config_path,
            options.api_base,
            options.model,
            ContentType.COT,
            options.num_pairs,
            options.verbose,
            async_mode=options.async_mode,
            provider=options.provider,
            profile=options.profile,
            document_text=d,
            config_overrides=options.overrides,
        ),
        PipelineStep.GENERATE_VQA: lambda d: process_file(
            None,
            None,
            options.config_path,
            options.api_base,
            options.model,
            ContentType.VQA_ADD_REASONING,
            options.num_pairs,
            options.verbose,
            async_mode=options.async_mode,
            provider=options.provider,
            profile=options.profile,
            document_text=d,
            config_overrides=options.overrides,
        ),
        PipelineStep.GENERATE_FROM_KG: lambda d: process_file(
            None,
            None,
            options.config_path,
            options.api_base,
            options.model,
            ContentType.FROM_KG,
            options.num_pairs,
            options.verbose,
            async_mode=options.async_mode,
            provider=options.provider,
            profile=options.profile,
            document_text=d,
            config_overrides=options.overrides,
        ),
        PipelineStep.GENERATE_TOOL_CALL: lambda d: process_file(
            None,
            None,
            options.config_path,
            options.api_base,
            options.model,
            ContentType.TOOL_CALL,
            options.num_pairs,
            options.verbose,
            async_mode=options.async_mode,
            provider=options.provider,
            profile=options.profile,
            document_text=d,
            config_overrides=options.overrides,
        ),
        PipelineStep.GENERATE_CONVERSATION: lambda d: process_file(
            None,
            None,
            options.config_path,
            options.api_base,
            options.model,
            ContentType.CONVERSATION,
            options.num_pairs,
            options.verbose,
            async_mode=options.async_mode,
            provider=options.provider,
            profile=options.profile,
            document_text=d,
            config_overrides=options.overrides,
        ),
        PipelineStep.GENERATE_MULTI_TOOL: lambda d: process_file(
            None,
            None,
            options.config_path,
            options.api_base,
            options.model,
            ContentType.MULTI_TOOL,
            options.num_pairs,
            options.verbose,
            async_mode=options.async_mode,
            provider=options.provider,
            profile=options.profile,
            document_text=d,
            config_overrides=options.overrides,
        ),
        PipelineStep.GENERATE_CANDIDATES: lambda d: process_file(
            None,
            None,
            options.config_path,
            options.api_base,
            options.model,
            (
                ContentType.PREF_PAIR
                if dataset_type == DatasetType.PREF_PAIR
                else ContentType.PREF_LIST
            ),
            options.num_pairs,
            options.verbose,
            async_mode=options.async_mode,
            provider=options.provider,
            profile=options.profile,
            document_text=d,
            config_overrides=options.overrides,
        ),
        PipelineStep.LABEL_PAIRS: lambda d: d,
        PipelineStep.RANK_RESPONSES: lambda d: d,
        PipelineStep.CURATE: lambda d: curate_qa_pairs(
            d,
            None,
            threshold,
            options.api_base,
            options.model,
            options.config_path,
            options.verbose,
            options.provider,
            async_mode=options.async_mode,
        ),
        PipelineStep.SAVE: lambda d: _save(d),
    }

    for step in pipeline.steps:
        if step in {PipelineStep.INGEST, PipelineStep.TO_KG}:
            continue
        if verbose:
            logger.info("Running step %s", step.value)
        handler = handlers.get(step)
        if handler:
            data = handler(data)

    return data
