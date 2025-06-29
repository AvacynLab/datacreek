from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from datacreek.core.create import process_file
from datacreek.core.curate import curate_qa_pairs
from datacreek.core.knowledge_graph import KnowledgeGraph
from datacreek.core.save_as import convert_format
from datacreek.models.content_type import ContentType
from datacreek.utils.config import get_format_settings, load_config_with_overrides

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
    kg: KnowledgeGraph | None = None
    multi_answer: bool = False


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
    kg: KnowledgeGraph,
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
    document_text: str | None = None,
    multi_answer: bool = False,
) -> Any:
    """Execute the generation steps for ``dataset_type`` using ``kg``.

    The raw text is derived from the provided knowledge graph. Only steps after
    the knowledge graph construction are executed. The function returns the
    final in-memory representation of the dataset without writing files. When
    ``async_mode`` is True, generation steps use asynchronous LLM requests when
    supported by the client.

    ``multi_answer`` controls whether multiple answers are generated per fact
    when using the knowledge graph generator.

    All dataset types except ``TEXT`` share a common post-knowledge-graph flow
    consisting of generation, optional curation and output formatting. This
    helper executes those steps in-memory and returns the resulting dataset.
    """

    try:
        pipeline = get_pipeline(dataset_type)
    except KeyError as exc:
        raise ValueError(str(exc)) from exc

    # The TEXT dataset does not include generation after the knowledge graph,
    # so it remains unsupported by this helper.
    if dataset_type == DatasetType.TEXT:
        raise ValueError("run_generation_pipeline does not support the TEXT dataset type")

    if document_text is None:
        document_text = kg.to_text()

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
        kg=kg,
        multi_answer=multi_answer,
    )
    data: Any = document_text

    def _save(d: Any) -> Any:
        cfg = load_config_with_overrides(str(config_path) if config_path else None, overrides)
        fmt_cfg = get_format_settings(cfg)
        format_type = fmt or fmt_cfg.default
        return convert_format(d, None, format_type, cfg)

    def _generate(ct: ContentType, text: str) -> Any:
        return process_file(
            None,
            None,
            options.config_path,
            options.api_base,
            options.model,
            ct,
            options.num_pairs,
            options.verbose,
            async_mode=options.async_mode,
            provider=options.provider,
            profile=options.profile,
            document_text=text,
            kg=options.kg,
            config_overrides=options.overrides,
            multi_answer=options.multi_answer if ct is ContentType.FROM_KG else False,
        )

    def _validate(step: PipelineStep, result: Any) -> Any:
        """Basic sanity checks for generation results."""
        if step in {
            PipelineStep.GENERATE_QA,
            PipelineStep.GENERATE_FROM_KG,
        }:
            if not isinstance(result, dict) or "qa_pairs" not in result:
                raise ValueError("QA generation did not produce 'qa_pairs'")
        elif step is PipelineStep.GENERATE_COT:
            if not isinstance(result, dict) or "cot_examples" not in result:
                raise ValueError("CoT generation did not produce 'cot_examples'")
        elif step in {
            PipelineStep.GENERATE_TOOL_CALL,
            PipelineStep.GENERATE_CONVERSATION,
            PipelineStep.GENERATE_MULTI_TOOL,
        }:
            if not isinstance(result, dict) or "conversations" not in result:
                raise ValueError("Conversation generation missing 'conversations'")
        elif step is PipelineStep.GENERATE_CANDIDATES:
            if not isinstance(result, dict) or not ("pairs" in result or "responses" in result):
                raise ValueError("Candidate generation returned invalid data")
        return result

    def _curate(d: Any) -> Any:
        return curate_qa_pairs(
            d,
            None,
            threshold,
            options.api_base,
            options.model,
            options.config_path,
            options.verbose,
            options.provider,
            async_mode=options.async_mode,
            kg=options.kg,
        )

    handlers = {
        PipelineStep.GENERATE_QA: lambda d: _generate(ContentType.QA, d),
        PipelineStep.GENERATE_COT: lambda d: _generate(ContentType.COT, d),
        PipelineStep.GENERATE_VQA: lambda d: _generate(ContentType.VQA_ADD_REASONING, d),
        PipelineStep.GENERATE_FROM_KG: lambda d: _generate(ContentType.FROM_KG, d),
        PipelineStep.GENERATE_TOOL_CALL: lambda d: _generate(ContentType.TOOL_CALL, d),
        PipelineStep.GENERATE_CONVERSATION: lambda d: _generate(ContentType.CONVERSATION, d),
        PipelineStep.GENERATE_MULTI_TOOL: lambda d: _generate(ContentType.MULTI_TOOL, d),
        PipelineStep.GENERATE_CANDIDATES: lambda d: _generate(
            (
                ContentType.PREF_PAIR
                if dataset_type == DatasetType.PREF_PAIR
                else ContentType.PREF_LIST
            ),
            d,
        ),
        PipelineStep.LABEL_PAIRS: lambda d: d,
        PipelineStep.RANK_RESPONSES: lambda d: d,
        PipelineStep.CURATE: _curate,
        PipelineStep.SAVE: _save,
    }

    for step in pipeline.steps:
        if step in {PipelineStep.INGEST, PipelineStep.TO_KG}:
            continue
        if verbose:
            logger.info("Running step %s", step.value)
        handler = handlers.get(step)
        if handler:
            result = handler(data)
            if step.name.startswith("GENERATE"):
                data = _validate(step, result)
            else:
                data = result

    return data
