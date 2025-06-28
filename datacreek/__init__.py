"""Datacreek package."""

from typing import TYPE_CHECKING

__version__ = "0.0.2"

# Avoid heavy imports at module load time. Relevant classes and functions
# are available via submodules such as ``datacreek.pipelines`` or
# ``datacreek.core``.

__all__: list[str] = [
    "__version__",
    "DatasetBuilder",
    "DatasetType",
    "GenerationPipeline",
    "TrainingGoal",
    "get_pipeline",
    "get_trainings_for_dataset",
    "get_dataset_types_for_training",
    "get_pipelines_for_training",
    "ingest_file",
    "to_kg",
    "ingest_into_dataset",
    "extract_facts",
    "KnowledgeGraph",
    "GenerationSettings",
]

if TYPE_CHECKING:  # pragma: no cover - used for type checking only
    from .config_models import GenerationSettings
    from .core.dataset import DatasetBuilder
    from .core.ingest import ingest_into_dataset, process_file as ingest_file, to_kg
    from .core.knowledge_graph import KnowledgeGraph
    from .pipelines import (
        PIPELINES,
        DatasetType,
        GenerationPipeline,
        TrainingGoal,
        get_dataset_types_for_training,
        get_pipeline,
        get_pipelines_for_training,
        get_trainings_for_dataset,
    )
    from .utils.fact_extraction import extract_facts


def __getattr__(name: str):
    if name == "DatasetBuilder":
        from .core.dataset import DatasetBuilder as _DB

        return _DB
    if name in {"ingest_file", "to_kg", "ingest_into_dataset"}:
        from .core.ingest import (
            ingest_into_dataset as _ingest_into_dataset,
            process_file as _ingest_file,
            to_kg as _to_kg,
        )

        return {
            "ingest_file": _ingest_file,
            "to_kg": _to_kg,
            "ingest_into_dataset": _ingest_into_dataset,
        }[name]
    if name == "KnowledgeGraph":
        from .core.knowledge_graph import KnowledgeGraph as _KG

        return _KG
    if name == "GenerationSettings":
        from .config_models import GenerationSettings as _GS

        return _GS
    if name in {
        "DatasetType",
        "GenerationPipeline",
        "TrainingGoal",
        "get_pipeline",
        "get_trainings_for_dataset",
        "get_dataset_types_for_training",
        "get_pipelines_for_training",
    }:
        from .pipelines import (
            DatasetType as _DT,
            GenerationPipeline as _GP,
            TrainingGoal as _TG,
            get_pipeline as _gp,
            get_trainings_for_dataset as _gtfd,
            get_dataset_types_for_training as _gdtft,
            get_pipelines_for_training as _gpft,
        )

        mapping = {
            "DatasetType": _DT,
            "GenerationPipeline": _GP,
            "TrainingGoal": _TG,
            "get_pipeline": _gp,
            "get_trainings_for_dataset": _gtfd,
            "get_dataset_types_for_training": _gdtft,
            "get_pipelines_for_training": _gpft,
        }

        return mapping[name]
    if name == "extract_facts":
        from .utils.fact_extraction import extract_facts as _ef

        return _ef
    raise AttributeError(name)
