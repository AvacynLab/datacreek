# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
"""Datacreek: tools for preparing synthetic data for LLM fine-tuning."""

__version__ = "0.0.2"

from .config_models import GenerationSettings
from .core.dataset import DatasetBuilder
from .core.ingest import ingest_into_dataset
from .core.ingest import process_file as ingest_file
from .core.ingest import to_kg
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

__all__ = [
    "GenerationPipeline",
    "TrainingGoal",
    "DatasetType",
    "PIPELINES",
    "get_pipeline",
    "get_trainings_for_dataset",
    "get_dataset_types_for_training",
    "get_pipelines_for_training",
    "KnowledgeGraph",
    "DatasetBuilder",
    "ingest_file",
    "to_kg",
    "ingest_into_dataset",
    "extract_facts",
    "GenerationSettings",
]
