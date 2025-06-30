from __future__ import annotations

import asyncio
import json
import logging
import time
import traceback
from dataclasses import asdict, dataclass, is_dataclass
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:  # pragma: no cover - for type checkers only
    from datacreek.core.dataset import DatasetBuilder

from datacreek.core.create import process_file, process_file_async
from datacreek.core.curate import curate_qa_pairs, curate_qa_pairs_async
from datacreek.core.knowledge_graph import KnowledgeGraph
from datacreek.core.save_as import convert_format
from datacreek.models.content_type import ContentType
from datacreek.models.results import (
    ConversationResult,
    COTGenerationResult,
    PrefListResult,
    PrefPairResult,
    QAGenerationResult,
)
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
    KG_CLEANUP = "kg_cleanup"
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
    batch_size: int | None = None
    inference_batch: int | None = None
    checkpoint_dir: Path | None = None


# Expected dataclass types for generation results
STEP_DATACLASSES = {
    PipelineStep.GENERATE_QA: QAGenerationResult,
    PipelineStep.GENERATE_FROM_KG: QAGenerationResult,
    PipelineStep.GENERATE_COT: COTGenerationResult,
    PipelineStep.GENERATE_TOOL_CALL: ConversationResult,
    PipelineStep.GENERATE_CONVERSATION: ConversationResult,
    PipelineStep.GENERATE_MULTI_TOOL: ConversationResult,
}

# Expected top-level fields for validation
STEP_FIELDS = {
    PipelineStep.GENERATE_QA: {"qa_pairs"},
    PipelineStep.GENERATE_FROM_KG: {"qa_pairs"},
    PipelineStep.GENERATE_COT: {"cot_examples"},
    PipelineStep.GENERATE_TOOL_CALL: {"conversations"},
    PipelineStep.GENERATE_CONVERSATION: {"conversations"},
    PipelineStep.GENERATE_MULTI_TOOL: {"conversations"},
    PipelineStep.GENERATE_CANDIDATES: {"pairs", "responses"},
}


def _validate_step_result(
    dataset_type: DatasetType, step: PipelineStep, result: Any
) -> Dict[str, Any]:
    """Return ``result`` as a dict and verify it matches ``step`` expectations."""

    step_name = step.value
    expected_cls = STEP_DATACLASSES.get(step)
    if step is PipelineStep.GENERATE_CANDIDATES:
        expected_cls = PrefPairResult if dataset_type == DatasetType.PREF_PAIR else PrefListResult

    if is_dataclass(result):
        if expected_cls and not isinstance(result, expected_cls):
            raise ValueError(
                f"{step_name}: expected {expected_cls.__name__}, got {type(result).__name__}"
            )
        result_dict = asdict(result)
    else:
        result_dict = result

    expected_fields = STEP_FIELDS.get(step)
    if expected_fields:
        if not isinstance(result_dict, dict):
            raise ValueError(f"{step_name}: expected mapping, got {type(result).__name__}")
        if step is PipelineStep.GENERATE_CANDIDATES:
            if not ("pairs" in result_dict or "responses" in result_dict):
                raise ValueError(
                    f"{step_name}: expected candidate pairs, got {type(result).__name__}"
                )
        else:
            missing = [f for f in expected_fields if f not in result_dict]
            if missing:
                raise ValueError(f"{step_name}: missing fields {', '.join(missing)}")
        for f in expected_fields:
            if f in result_dict:
                if not isinstance(result_dict[f], list):
                    raise ValueError(f"{step_name}: field '{f}' must be a list")
                if f == "qa_pairs":
                    for item in result_dict[f]:
                        if is_dataclass(item):
                            item_dict = asdict(item)
                        elif isinstance(item, dict):
                            item_dict = item
                        else:
                            raise ValueError(f"{step_name}: qa_pairs items must be mappings")
                        if (
                            "question" not in item_dict
                            or "answer" not in item_dict
                            or not str(item_dict["question"]).strip()
                            or not str(item_dict["answer"]).strip()
                        ):
                            raise ValueError(f"{step_name}: invalid QA pair item")
                if f == "cot_examples":
                    for item in result_dict[f]:
                        if is_dataclass(item):
                            item_dict = asdict(item)
                        elif isinstance(item, dict):
                            item_dict = item
                        else:
                            raise ValueError(f"{step_name}: cot_examples items must be mappings")
                        required = {"question", "reasoning", "answer"}
                        if (
                            any(k not in item_dict for k in required)
                            or not str(item_dict["question"]).strip()
                            or not str(item_dict["answer"]).strip()
                            or not str(item_dict["reasoning"]).strip()
                        ):
                            raise ValueError(f"{step_name}: invalid COT example item")

        if step is PipelineStep.GENERATE_CANDIDATES:
            if dataset_type == DatasetType.PREF_PAIR:
                pairs = result_dict.get("pairs", [])
                if not isinstance(pairs, list):
                    raise ValueError(f"{step_name}: field 'pairs' must be a list")
                for item in pairs:
                    if not isinstance(item, dict):
                        raise ValueError(f"{step_name}: pairs items must be mappings")
                    if (
                        "question" not in item
                        or "chosen" not in item
                        or "rejected" not in item
                        or not str(item["question"]).strip()
                        or not str(item["chosen"]).strip()
                        or not str(item["rejected"]).strip()
                    ):
                        raise ValueError(f"{step_name}: invalid pair item")
            else:
                responses = result_dict.get("responses", [])
                if not isinstance(responses, list):
                    raise ValueError(f"{step_name}: field 'responses' must be a list")
                for resp in responses:
                    if not isinstance(resp, dict):
                        raise ValueError(f"{step_name}: responses items must be mappings")
                    if not str(resp.get("question", "")).strip():
                        raise ValueError(f"{step_name}: response question is empty")
                    answers = resp.get("answers")
                    if not isinstance(answers, list) or not answers:
                        raise ValueError(f"{step_name}: response answers must be a non-empty list")
                    for ans in answers:
                        if not isinstance(ans, dict):
                            raise ValueError(f"{step_name}: answers items must be mappings")
                        if not str(ans.get("text", "")).strip():
                            raise ValueError(f"{step_name}: answer text is empty")

    return result_dict


def _serialize(data: Any) -> Any:
    """Recursively convert dataclasses to dictionaries for JSON dumping."""
    if is_dataclass(data):
        return asdict(data)
    if isinstance(data, dict):
        return {k: _serialize(v) for k, v in data.items()}
    if isinstance(data, list):
        return [_serialize(x) for x in data]
    return data


class PipelineExecutionError(RuntimeError):
    """Raised when a generation pipeline step fails with additional context."""

    def __init__(self, step: PipelineStep, original_exception: Exception):
        self.step = step
        self.original_exception = original_exception
        self.traceback = "".join(
            traceback.format_exception(
                type(original_exception), original_exception, original_exception.__traceback__
            )
        )
        super().__init__(f"{step.value} failed: {original_exception}")


PIPELINES: Dict[DatasetType, GenerationPipeline] = {
    DatasetType.QA: GenerationPipeline(
        dataset_type=DatasetType.QA,
        steps=[
            PipelineStep.INGEST,
            PipelineStep.TO_KG,
            PipelineStep.KG_CLEANUP,
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
            PipelineStep.KG_CLEANUP,
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
            PipelineStep.KG_CLEANUP,
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
            PipelineStep.KG_CLEANUP,
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
            PipelineStep.KG_CLEANUP,
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
            PipelineStep.KG_CLEANUP,
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
            PipelineStep.KG_CLEANUP,
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
            PipelineStep.KG_CLEANUP,
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
            PipelineStep.KG_CLEANUP,
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


async def _run_generation_pipeline_impl(
    dataset_type: DatasetType,
    kg: KnowledgeGraph,
    *,
    dataset_builder: "DatasetBuilder | None" = None,
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
    resolve_threshold: float = 0.8,
    resolve_aliases: dict[str, list[str]] | None = None,
    use_async_handlers: bool = False,
    batch_size: int | None = None,
    inference_batch: int | None = None,
    start_step: PipelineStep | None = None,
    checkpoint_dir: Path | None = None,
) -> Any:
    """Shared implementation for sync and async generation pipelines."""

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

    cfg = load_config_with_overrides(str(config_path) if config_path else None, overrides)
    fmt_cfg = get_format_settings(cfg)

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
        batch_size=batch_size,
        inference_batch=inference_batch,
        checkpoint_dir=checkpoint_dir,
    )
    if checkpoint_dir:
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

    data: Any = document_text
    if start_step and checkpoint_dir:
        idx = pipeline.steps.index(start_step)
        if idx > 0:
            prev_step = pipeline.steps[idx - 1]
            cp_file = checkpoint_dir / f"{prev_step.value}.json"
            if cp_file.exists():
                data = json.loads(cp_file.read_text())

    async def _identity(d: Any) -> Any:
        return d

    async def _save(d: Any) -> Any:
        format_type = fmt or fmt_cfg.default
        return await asyncio.to_thread(convert_format, d, None, format_type, cfg)

    async def _generate(ct: ContentType, text: str) -> Any:
        if use_async_handlers:
            return await process_file_async(
                None,
                None,
                options.config_path,
                options.api_base,
                options.model,
                ct,
                options.num_pairs,
                options.verbose,
                provider=options.provider,
                profile=options.profile,
                document_text=text,
                kg=options.kg,
                config_overrides=options.overrides,
                multi_answer=options.multi_answer if ct is ContentType.FROM_KG else False,
            )
        return await asyncio.to_thread(
            process_file,
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

    async def _curate(d: Any) -> Any:
        if use_async_handlers:
            result = await curate_qa_pairs_async(
                d,
                None,
                threshold,
                options.api_base,
                options.model,
                options.config_path,
                options.verbose,
                options.provider,
                kg=options.kg,
                batch_size=options.batch_size,
                inference_batch=options.inference_batch,
            )
        else:
            result = await asyncio.to_thread(
                curate_qa_pairs,
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
                batch_size=options.batch_size,
                inference_batch=options.inference_batch,
            )

        if dataset_builder is not None:
            metrics = None
            if isinstance(result, dict):
                metrics = result.get("metrics")
            elif hasattr(result, "metrics"):
                metrics = result.metrics
            if metrics:
                if is_dataclass(metrics):
                    metrics_dict = asdict(metrics)
                else:
                    metrics_dict = metrics
                dataset_builder._record_event(
                    "curate",
                    "Curated %d/%d QA pairs"
                    % (
                        metrics_dict.get("filtered", 0),
                        metrics_dict.get("total", 0),
                    ),
                    **metrics_dict,
                )

        return result

    async def _kg_cleanup(d: Any) -> Any:
        if dataset_builder is not None:
            removed, cleaned = dataset_builder.cleanup_graph(
                resolve_threshold=resolve_threshold,
                resolve_aliases=resolve_aliases,
            )
        else:
            removed = kg.deduplicate_chunks()
            cleaned = kg.clean_chunk_texts()
            kg.resolve_entities(
                threshold=resolve_threshold, aliases=resolve_aliases
            )
        if verbose:
            logger.info("  KG cleanup - removed:%d cleaned:%d", removed, cleaned)
        return d

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
        PipelineStep.LABEL_PAIRS: _identity,
        PipelineStep.RANK_RESPONSES: _identity,
        PipelineStep.KG_CLEANUP: _kg_cleanup,
        PipelineStep.CURATE: _curate,
        PipelineStep.SAVE: _save,
    }

    start_idx = 0
    if start_step and start_step in pipeline.steps:
        start_idx = pipeline.steps.index(start_step)
    for step in pipeline.steps[start_idx:]:
        if step in {PipelineStep.INGEST, PipelineStep.TO_KG}:
            continue
        if verbose:
            logger.info("Running step %s", step.value)
        handler = handlers.get(step)
        if handler:
            start = time.perf_counter()
            try:
                result = await handler(data)
            except Exception as exc:
                logger.exception("Step %s failed", step.value)
                raise PipelineExecutionError(step, exc) from exc
            duration = time.perf_counter() - start
            if step.name.startswith("GENERATE"):
                data = _validate_step_result(dataset_type, step, result)
            else:
                data = result
            if options.checkpoint_dir:
                cp_file = Path(options.checkpoint_dir) / f"{step.value}.json"
                cp_file.write_text(json.dumps(_serialize(data)))
            if verbose:
                logger.info("Finished %s in %.2fs", step.value, duration)
                if isinstance(data, dict):
                    if "qa_pairs" in data:
                        logger.info("  Pairs: %d", len(data["qa_pairs"]))
                    if "conversations" in data:
                        logger.info("  Conversations: %d", len(data["conversations"]))
                    if step is PipelineStep.CURATE and "metrics" in data:
                        m = data["metrics"]
                        logger.info(
                            "  Curation metrics - total:%d filtered:%d retention:%.2f avg:%.1f",
                            m.get("total", 0),
                            m.get("filtered", 0),
                            m.get("retention_rate", 0.0),
                            m.get("avg_score", 0.0),
                        )

    return data


def run_generation_pipeline(
    dataset_type: DatasetType,
    kg: KnowledgeGraph,
    *,
    dataset_builder: "DatasetBuilder | None" = None,
    batch_size: int | None = None,
    inference_batch: int | None = None,
    resolve_threshold: float = 0.8,
    resolve_aliases: dict[str, list[str]] | None = None,
    start_step: PipelineStep | None = None,
    checkpoint_dir: Path | None = None,
    **kwargs: Any,
) -> Any:
    """Execute the generation steps synchronously.

    Parameters
    ----------
    dataset_builder:
        If provided, knowledge graph cleanup will log events on this builder.
    resolve_threshold:
        Similarity threshold used when merging entities during knowledge graph
        cleanup.
    resolve_aliases:
        Optional alias mapping passed to :meth:`DatasetBuilder.resolve_entities`.
    checkpoint_dir:
        Directory to store intermediate results for resuming later.

    Raises
    ------
    PipelineExecutionError
        If any step fails.
    """

    return asyncio.run(
        _run_generation_pipeline_impl(
            dataset_type,
            kg,
            use_async_handlers=False,
            dataset_builder=dataset_builder,
            batch_size=batch_size,
            inference_batch=inference_batch,
            resolve_threshold=resolve_threshold,
            resolve_aliases=resolve_aliases,
            start_step=start_step,
            checkpoint_dir=checkpoint_dir,
            **kwargs,
        )
    )


async def run_generation_pipeline_async(
    dataset_type: DatasetType,
    kg: KnowledgeGraph,
    *,
    dataset_builder: "DatasetBuilder | None" = None,
    batch_size: int | None = None,
    inference_batch: int | None = None,
    resolve_threshold: float = 0.8,
    resolve_aliases: dict[str, list[str]] | None = None,
    start_step: PipelineStep | None = None,
    checkpoint_dir: Path | None = None,
    **kwargs: Any,
) -> Any:
    """Asynchronous counterpart to :func:`run_generation_pipeline`.

    Parameters
    ----------
    dataset_builder:
        If provided, knowledge graph cleanup will log events on this builder.
    resolve_threshold:
        Similarity threshold used when merging entities during knowledge graph
        cleanup.
    resolve_aliases:
        Optional alias mapping passed to :meth:`DatasetBuilder.resolve_entities`.
    checkpoint_dir:
        Directory to store intermediate results for resuming later.

    Raises
    ------
    PipelineExecutionError
        If any step fails.
    """

    kwargs["async_mode"] = True
    return await _run_generation_pipeline_impl(
        dataset_type,
        kg,
        use_async_handlers=True,
        dataset_builder=dataset_builder,
        batch_size=batch_size,
        inference_batch=inference_batch,
        resolve_threshold=resolve_threshold,
        resolve_aliases=resolve_aliases,
        start_step=start_step,
        checkpoint_dir=checkpoint_dir,
        **kwargs,
    )
